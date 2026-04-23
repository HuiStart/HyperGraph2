"""Debug script: run one sample through LLM and print raw output."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.baseline import FastModeScorer
from src.utils.llm_wrapper import LLMWrapper

llm = LLMWrapper('configs/llm.yaml')
llm.set_model('cloud_default')
scorer = FastModeScorer(llm)

with open('data/processed/test_2024_processed.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

sample = samples[0]
print(f"Sample ID: {sample['id']}")
print(f"Title: {sample['title'][:80]}")
print(f"Paper context length: {len(sample['paper_context'])}")
print()

result = scorer.score(sample['paper_context'], sample['title'])
print("=== RAW OUTPUT ===")
print(result['raw_output'])
print()
print("=== SCORES ===")
print(result['scores'])
print()
print("=== PARSED ===")
print(result['parsed'])
