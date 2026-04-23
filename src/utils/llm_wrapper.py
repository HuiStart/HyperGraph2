"""
Unified LLM wrapper supporting local (ollama) and remote (OpenAI-compatible) APIs.

All prompts, temperatures, max_tokens, and output formats are configurable via configs/llm.yaml.
"""

import os
import re
from typing import Any

import requests
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMWrapper:
    """Unified wrapper for local and remote LLM APIs."""

    def __init__(self, config_path: str = "configs/llm.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.provider = self.config.get("default_provider", "ollama")
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize the selected provider."""
        if self.provider == "ollama":
            self.ollama_config = self.config.get("ollama", {})
            self.base_url = self.ollama_config.get("base_url", "http://localhost:11434")
            # Default to cloud model
            model_cfg = self.ollama_config.get("models", {}).get("cloud_default", {})
            self.model_name = model_cfg.get("name", "qwen3.5:4b")
            self.temperature = model_cfg.get("temperature", 0.4)
            self.top_p = model_cfg.get("top_p", 0.95)
            self.max_tokens = model_cfg.get("max_tokens", 7000)
        elif self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package is required for remote API. Install with: pip install openai")
            openai_cfg = self.config.get("openai", {})
            self.base_url = openai_cfg.get("base_url", "https://api.openai.com/v1")
            api_key = openai_cfg.get("api_key", "")
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, "")
            model_cfg = openai_cfg.get("models", {}).get("default", {})
            self.model_name = model_cfg.get("name", "gpt-4o-mini")
            self.temperature = model_cfg.get("temperature", 0.4)
            self.top_p = model_cfg.get("top_p", 0.95)
            self.max_tokens = model_cfg.get("max_tokens", 7000)
            self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def set_model(self, model_key: str) -> None:
        """Switch to a different model configuration.

        Args:
            model_key: Key in config (e.g., 'local_default', 'cloud_default').
        """
        if self.provider == "ollama":
            model_cfg = self.ollama_config.get("models", {}).get(model_key, {})
            if not model_cfg:
                raise ValueError(f"Model config '{model_key}' not found in ollama config")
            self.model_name = model_cfg.get("name", self.model_name)
            self.temperature = model_cfg.get("temperature", self.temperature)
            self.top_p = model_cfg.get("top_p", self.top_p)
            self.max_tokens = model_cfg.get("max_tokens", self.max_tokens)
        else:
            openai_cfg = self.config.get("openai", {})
            model_cfg = openai_cfg.get("models", {}).get(model_key, {})
            if model_cfg:
                self.model_name = model_cfg.get("name", self.model_name)
                self.temperature = model_cfg.get("temperature", self.temperature)
                self.max_tokens = model_cfg.get("max_tokens", self.max_tokens)

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """Generate text from LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            **kwargs: Override temperature, max_tokens, etc.

        Returns:
            Generated text string.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        if self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, temperature, max_tokens, top_p)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens, top_p)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_ollama(self, prompt: str, system_prompt: str | None,
                         temperature: float, max_tokens: int, top_p: float) -> str:
        """Generate via ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def _generate_openai(self, prompt: str, system_prompt: str | None,
                         temperature: float, max_tokens: int, top_p: float) -> str:
        """Generate via OpenAI-compatible API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise

    def generate_json(self, prompt: str, system_prompt: str | None = None, **kwargs) -> dict[str, Any]:
        """Generate and parse JSON output from LLM.

        Falls back to regex extraction if direct JSON parsing fails.
        """
        text = self.generate(prompt, system_prompt, **kwargs)

        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            import json
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}. Returning raw text wrapper.")
            return {"raw_text": text, "parsed": False}

    def get_scoring_prompt(self, mode: str = "fast", reviewer_num: int = 4) -> str:
        """Get the system prompt for scoring mode.

        Args:
            mode: 'fast', 'standard', or 'best'.
            reviewer_num: Number of reviewers to simulate.

        Returns:
            Formatted system prompt.
        """
        prompt_cfg = self.config.get("prompt_modes", {}).get("scoring", {})
        template = prompt_cfg.get("system_prompt_template", "")

        mode_instructions = self.config.get("mode_instructions", {})
        mode_text = mode_instructions.get(mode, "")
        mode_text = mode_text.format(reviewer_num=reviewer_num)

        reviewer_instruction = ""
        if mode in ("standard", "best"):
            reviewer_instruction = mode_text
            mode_text = mode_instructions.get(mode, "").split("\n")[0]

        return template.format(
            mode_instruction=mode_text,
            reviewer_instruction=reviewer_instruction
        )

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Chat-style generation with message history.

        Args:
            messages: List of {'role': 'system'|'user'|'assistant', 'content': str}
            **kwargs: Override generation parameters.

        Returns:
            Assistant response text.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        if self.provider == "ollama":
            # Ollama doesn't have native chat API in the same way, simulate
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")
            prompt = "\n".join(prompt_parts) + "Assistant:"
            return self._generate_ollama(prompt, None, temperature, max_tokens, top_p)
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
