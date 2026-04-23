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
python src/cli/main.py evaluate --official \
    --input "Research/evaluate/DeepReview/sample.json" \
    --output-dir "experiments/deepreview_baseline"
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