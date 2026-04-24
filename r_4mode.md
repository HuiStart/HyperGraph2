四个模式分别用这些命令跑：

1. Fast Mode（最快，1 次 LLM 调用）


python scripts/run_and_evaluate.py \
    --input data/processed/test_2024_processed.json \
    --mode fast \
    --output-pred experiments/pred_fast.json \
    --output-metrics experiments/metrics_fast.json
2. Standard Mode（模拟 4 个审稿人）


python scripts/run_and_evaluate.py \
    --input data/processed/test_2024_processed.json \
    --mode standard \
    --output-pred experiments/pred_standard.json \
    --output-metrics experiments/metrics_standard.json
3. Best Mode（检索增强，最慢）


python scripts/run_and_evaluate.py \
    --input data/processed/test_2024_processed.json \
    --mode best \
    --output-pred experiments/pred_best.json \
    --output-metrics experiments/metrics_best.json
4. Our Method（证据驱动多智能体）


python scripts/run_ours.py \
    --input data/processed/test_2024_processed.json \
    --output experiments/pred_ours.json \
    --use-llm-evidence
只想跑 2 条测试一下：
加 --max-samples 2 即可：


python scripts/run_and_evaluate.py \
    --input data/processed/test_2024_processed.json \
    --mode fast \
    --max-samples 2
跑完后对比指标：


python scripts/evaluate_full.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/pred_fast.json
你先把 Fast 跑通（确认输出有值而不是全 -），再跑其他模式。Fast 通了其他的基本没问题。


# 处理数据
python -m src.cli.main preprocess \
    --input data/deepreview/test_2024.csv \
    --output data/processed/test_2024_processed.json

# run
python scripts/run_ours.py \
    --input data/processed/test_2024_processed.json \
    --output experiments/pred_ours.json \
    --use-llm-evidence \
    -n 2

同步代码后，建议先用小批量（如 10 条）测试，重点关注 Soundness 的 true/pred 对比：
python scripts/run_ours.py \
    --input data/processed/test_2024_processed.json \
    --output experiments/ours_results.json \
    --samples 10
跑完后用诊断脚本查看：

python scripts/diagnose_scores.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/ours_results.json
如果 Soundness 的 Spearman 转正（> 0），说明修复有效，再跑全量。