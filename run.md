# 项目进度汇报
## 完成状态
### 已实现的模块
| 模块 | 文件 | 状态 |
| ---- | ---- | ---- |
| 项目骨架 | requirements.txt, configs/, 目录结构 | ✅ |
| LLM Wrapper | src/utils/llm_wrapper.py | ✅ 支持 ollama + OpenAI |
| 输出解析器 | src/utils/parser.py | ✅ 兼容 \boxed_review{} / \boxed_simreviewers{} |
| 数据适配器 | src/adapters/deepreview_adapter.py | ✅ 自动提取数字评分、计算多评审人均值 |
| LaTeX 解析 | src/preprocess/latex_parser.py | ✅ 章节/图表/引用提取 |
| Rubric | src/rubric/fixed_rubric.py | ✅ DeepReview 固定4维度 |
| 官方评估 | src/evaluation/metrics.py, official_eval.py | ✅ MSE/MAE/Spearman/Decision F1/Pairwise Acc |
| Baseline 1/2/3 | src/scoring/baseline.py | ✅ Fast/Standard/Best Mode |
| 证据抽取 | src/evidence/extractor.py | ✅ 5类学术论文证据 |
| 超图构建 | src/graph/builder.py, patterns.py, retrieval.py, consistency.py | ✅ 5类模式超边 + 一致性检查 |
| 多智能体 | src/agents/*.py, workflow.py | ✅ 5个Agent + LangGraph编排 |
| CLI | src/cli/main.py | ✅ |
| 运行脚本 | scripts/run_baseline.py, run_ours.py, run_ablation.py | ✅ |
| README | README.md | ✅ |

### 与 DeepReview 官方标准的关键对齐
根据对 `Research/` 下官方代码的分析，以下 `design.md` 假设已被修正：
1. **数据格式**：实际输入是 paper_context（LaTeX 全文），不是通用 input_text + code_blocks
2. **评分维度**：固定为 Rating(1-10)、Soundness(1-5)、Presentation(1-5)、Contribution(1-5)，无灵活权重
3. **评估指标**：主报告指标与 evaluate.py 完全一致（MSE、MAE、Spearman、Decision Acc、Decision F1 macro、Pairwise Acc）。Pearson/QWK/RMSE 保留为扩展指标
4. **输出格式**：LLM 输出为自然语言文本，通过 \boxed_review{} / \boxed_simreviewers{} 包裹，解析逻辑与官方 get_reviewer_score() 对齐
5. **证据类型**：适配学术论文场景（论点/方法/实验/引用/一致性），移除代码/测试相关假设
6. **Baseline**：直接映射为 DeepReviewer 官方 Fast/Standard/Best 三种模式

## 验证结果
官方评估流程已在 `sample.json` 上跑通：
```bash
python src/cli/main.py evaluate --official --input "Research/evaluate/DeepReview/sample.json" --output-dir "experiments/deepreview_baseline"
```

三种模式的预测均被正确解析，评估指标已输出到：

- `experiments/deepreview_baseline/results_fast.md`
- `experiments/deepreview_baseline/results_standard.md`
- `experiments/deepreview_baseline/results_best.md`

### 快速开始

```bash
# 1. 预处理
python src/cli/main.py preprocess --input "Research/evaluate/DeepReview/sample.json"

# 2. 评估官方预测
python src/cli/main.py evaluate --official

# 3. 运行证据抽取
python src/cli/main.py evidence

# 4. 运行 baseline（需配置 ollama 或 API）
python scripts/run_baseline.py --mode fast --samples 10

# 5. 运行我们的方法
python scripts/run_ours.py --samples 10 --use-llm-evidence
```

### 下一步建议

1. **接入完整 DeepReview 数据集**：当前 `sample.json` 只有3条样本，需要完整数据集才能获得稳定指标
2. **配置本地模型**：修改 `configs/llm.yaml` 中的 ollama 地址和模型名，然后运行 baseline 生成实际预测
3. **运行消融实验**：使用 `scripts/run_ablation.py` 对比各模块贡献
4. **Side-by-Side 评估**：实现双盲胜率对比（参考 `win_rate_evaluate.py` 的 prompt 模板）

如果你希望我现在继续：接入完整数据集、运行实际 baseline 生成预测、或者实现 Side-by-Side 评估，请告诉我。



## 4. 运行项目

### 第一步：预处理数据



```bash
python src/cli/main.py preprocess --input "Research/evaluate/DeepReview/sample.json" --output "data/processed/deepreview_processed.json"
```

### 第二步：评估官方已有预测（无需调用 LLM）



```bash
python src/cli/main.py evaluate --official --input "Research/evaluate/DeepReview/sample.json" --output-dir "experiments/deepreview_baseline"
```

### 第三步：运行 Baseline 生成新预测



```bash
# 使用 cloud_default 模型（deepseek-r1:14b-16k）
# 先测试 1 条样本确认能跑通
python scripts/run_baseline.py --mode fast --samples 1 --llm-config configs/llm.yaml

# 如果正常，再跑完整数据集
python scripts/run_baseline.py --mode fast --llm-config configs/llm.yaml
python scripts/run_baseline.py --mode standard --llm-config configs/llm.yaml
python scripts/run_baseline.py --mode best --llm-config configs/llm.yaml
```

### 第四步：运行我们的增强方法



```bash
python scripts/run_ours.py --use-llm-evidence --llm-config configs/llm.yaml
```

### 第五步：运行消融实验



```bash
python scripts/run_ablation.py --variant full
python scripts/run_ablation.py --variant no_hg
python scripts/run_ablation.py --variant no_evidence
python scripts/run_ablation.py --variant no_risk
```

## 5. 显存与长文本注意事项

| 问题                       | 解决方案                                              |
| -------------------------- | ----------------------------------------------------- |
| 论文太长（40k tokens）     | 代码会自动截断到模型支持的 context_window             |
| 14B 模型单卡 16GB 可能紧张 | Ollama 会自动 offload 到 CPU，或降低 `context_window` |
| Best Mode 需要检索         | 默认使用本地 LLM fallback（不依赖 OpenScholar API）   |

## 6. 验证输出

运行完成后，检查结果文件：



```bash
ls experiments/
# deepreview_baseline/    # 官方预测评估结果
# baseline_fast.json      # Fast mode 新生成的预测
# baseline_standard.json  # Standard mode 预测
# baseline_best.json      # Best mode 预测
# ours_results.json       # 我们的方法结果
# ablation_*.json         # 消融实验结果
```

如果需要我把运行脚本改成后台批处理形式（nohup + 日志重定向），或者添加多进程并行加速，请告诉我。







## 已完成的代码修改

1. **`src/adapters/deepreview_adapter.py`**：支持 CSV 自动识别，提取 `inputs` 里的论文全文和 `rating`/`decision` 作为 ground truth
2. **`src/utils/parser.py`**：修复评分解析时的 float 转换容错
3. **`scripts/evaluate_predictions.py`**：新增，用于对比"LLM 新生成的预测"与 ground truth

------

## 服务器完整运行流程

### 前置准备



```bash
cd d:/Experiment/HyperGraph2

# 1. 确保 ollama 已启动
ollama serve

# 2. 拉取模型（双卡16GB用 deepseek-r1:14b-16k）
ollama pull deepseek-r1:14b-16k

# 3. 建议开 tmux，防止 SSH 断开
tmux new -s deepreview
```

### 修改模型配置（`configs/llm.yaml`）



```yaml
ollama:
  base_url: "http://localhost:11434"
  models:
    cloud_default:
      name: "deepseek-r1:14b-16k"
      temperature: 0.4
      max_tokens: 35000
      context_window: 16384
```

### Step 1: 预处理 CSV → JSON

提取论文内容和 ground truth，保存为统一格式：



```bash
# 2024 数据（652条）
python -m src.cli.main preprocess \
    --input data/deepreview/test_2024.csv \
    --output data/processed/test_2024_processed.json

# 2025 数据（634条）
python -m src.cli.main preprocess \
    --input data/deepreview/test_2025.csv \
    --output data/processed/test_2025_processed.json
```

### Step 2: 运行 Baseline（调用 LLM 生成预测）

**Fast Mode**（单审稿人，较快）：



```bash
python scripts/run_baseline.py \
    --mode fast \
    --input data/processed/test_2024_processed.json \
    --output experiments/baseline_fast_2024.json \
    --llm-config configs/llm.yaml
```

**Standard Mode**（4个模拟审稿人，慢）：



```bash
python scripts/run_baseline.py \
    --mode standard \
    --input data/processed/test_2024_processed.json \
    --output experiments/baseline_standard_2024.json \
    --llm-config configs/llm.yaml
```

**Best Mode**（检索增强，最慢）：



```bash
python scripts/run_baseline.py \
    --mode best \
    --input data/processed/test_2024_processed.json \
    --output experiments/baseline_best_2024.json \
    --llm-config configs/llm.yaml
```

### Step 3: 运行 Our Method（多 Agent 增强方法）



```bash
python scripts/run_ours.py \
    --input data/processed/test_2024_processed.json \
    --output experiments/ours_2024.json \
    --llm-config configs/llm.yaml
```

### Step 4: 评估（对比 Ground Truth）



```bash
# 评估 Fast Mode
python scripts/evaluate_predictions.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/baseline_fast_2024.json \
    --output experiments/metrics_fast_2024.json

# 评估 Standard Mode
python scripts/evaluate_predictions.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/baseline_standard_2024.json \
    --output experiments/metrics_standard_2024.json

# 评估 Our Method
python scripts/evaluate_predictions.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/ours_2024.json \
    --output experiments/metrics_ours_2024.json
```

------

## 快速测试（先跑 5 条验证）

在跑完整数据集前，强烈建议先测试 5 条：



```bash
# 预处理 5 条
python -m src.cli.main preprocess \
    --input data/deepreview/test_2024.csv \
    --output data/processed/test_5.json \
    --max-samples 5

# Fast Mode 测试
python scripts/run_baseline.py \
    --mode fast \
    --input data/processed/test_5.json \
    --output experiments/test_5_fast.json \
    --llm-config configs/llm.yaml

# 评估测试
python scripts/evaluate_predictions.py \
    --ground-truth data/processed/test_5.json \
    --predictions experiments/test_5_fast.json
```

确认输出正常、模型响应可解析后，再去掉 `--max-samples` 跑全量。

------

## 关键提醒

| 问题         | 方案                                                         |
| ------------ | ------------------------------------------------------------ |
| **运行时间** | 652 条 × Fast Mode 约 15-30 小时，Standard/Best 可能数倍。务必用 tmux/screen |
| **显存**     | `deepseek-r1:14b-16k` 双卡 16GB 应该够用。如 OOM 换 `qwen3.5:4b` |
| **超时**     | `src/utils/llm_wrapper.py` 默认 timeout=300，Standard/Best 输出很长，建议改 600+ |
| **断点续跑** | 代码每 5 条自动保存，但中断后会从头开始。如需断点续跑告诉我，我帮你写 |

需要我帮你写**断点续跑脚本**或者**多进程并行加速**吗？652 条串行跑太慢了。



## 1. Soundness/Presentation/Contribution 为 null 的原因

LLM 输出的 `\boxed_review{}` 里有时候只写了 Rating 和 Decision，漏了其他维度。原来的代码只从 boxed 块里提取，缺了就变成 null。

**已修复**：`src/scoring/baseline.py` 的 `FastModeScorer` 现在增加了 fallback，如果 boxed 块里找不到，会从 raw text 的其他位置再搜一遍。



```python
# 改进后的逻辑
soundness = self._extract_number(meta.get("soundness"))
if soundness is None:
    soundness = self._extract_from_raw(raw_output, "soundness")
```

**需要重新跑 baseline 才能生效**（之前的输出文件里的 null 不会自动修复）：



```bash
python scripts/run_baseline.py \
    --mode fast \
    --input data/processed/test_2024_processed.json \
    --output experiments/baseline_fast_2024_v2.json \
    --llm-config configs/llm.yaml
```

## 2. 生成 true/pred 对比表格

新增脚本 `scripts/generate_comparison_table.py`，输出你截图里的那种 markdown 表格：

```bash
python scripts/generate_comparison_table.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/baseline_fast_2024_v2.json \
    --output results_table.md \
    --max-samples 20
```
