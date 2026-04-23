## 一、数据集准备

将完整的 DeepReview 数据集放到项目目录：

```bash
# 假设你的完整数据集文件名为 deepreview_full.json
cp /path/to/your/deepreview_full.json d:/Experiment/HyperGraph2/data/raw/deepreview_full.json
```

**数据格式要求**：必须与 `sample.json` 完全一致，每条样本包含：

- `id`, `title`, `paper_context` (LaTeX 文本)
- `review` (多个人类评审，含 rating/content/soundness 等)
- `pred_fast_mode`, `pred_standard_mode`, `pred_best_mode` (AI 预测，可选)

------

## 二、服务器环境配置

### 1. 安装 Ollama 并拉取模型



```bash
# 安装 ollama（如未安装）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型（双卡16GB推荐 deepseek-r1:14b-16k）
ollama pull deepseek-r1:14b-16k

# 验证模型是否可用
ollama list
ollama run deepseek-r1:14b-16k "Hello"
```

### 2. 配置 Ollama 允许远程访问（如从其他机器调用）

```bash
# 编辑 ollama 服务配置
sudo systemctl edit ollama.service

# 添加环境变量
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"

# 重启服务
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 3. 安装 Python 依赖

```bash
cd d:/Experiment/HyperGraph2
pip install -r requirements.txt  # 或手动安装
pip install pyyaml requests scikit-learn scipy numpy networkx
```

------

## 三、修改配置文件

### 1. `configs/llm.yaml` — 模型配置

```yaml
# 如果 ollama 在本地跑，保持 localhost
# 如果在另一台机器，改为服务器 IP
ollama:
  base_url: "http://localhost:11434"
  models:
    # 云服务器用 cloud_default
    cloud_default:
      name: "deepseek-r1:14b-16k"
      temperature: 0.4
      top_p: 0.95
      max_tokens: 35000
      context_window: 16384
```

**关键**：代码默认使用 `local_default` 模型。需要修改 `src/utils/llm_wrapper.py` 或在运行时切换模型。

### 2. 修改 `src/utils/llm_wrapper.py` 默认加载云模型（可选）

编辑第 35 行，将 `local_default` 改为 `cloud_default`：



```python
model_cfg = self.ollama_config.get("models", {}).get("cloud_default", {})
```

或者运行时在每个 scorer 中调用 `llm.set_model("cloud_default")`。

更简单的做法：修改 baseline.py 第 428 行：

```python
llm = LLMWrapper(llm_config)
llm.set_model("cloud_default")  # 添加这行
```

### 3. `configs/deepreview.yaml` — 数据集路径



```yaml
dataset:
  name: "deepreview"
  raw_path: "data/raw/deepreview_full.json"  # 改为你的完整数据集
  processed_path: "data/processed/deepreview_processed.json"
```

------

## 四、运行步骤

**建议使用 tmux/screen 保持会话**，因为完整数据集可能需要运行数小时甚至数天。



```bash
tmux new -s deepreview
cd d:/Experiment/HyperGraph2
```

### Step 1: 预处理数据



```bash
python -m src.cli.main preprocess \
    --input data/raw/deepreview_full.json \
    --output data/processed/deepreview_processed.json
```

输出：`Processed N samples -> data/processed/deepreview_processed.json`

### Step 2: 运行 Baseline（三种模式）

**Fast Mode**（单审稿人，最快）：



```bash
python scripts/run_baseline.py \
    --mode fast \
    --input data/processed/deepreview_processed.json \
    --output experiments/baseline_fast_full.json \
    --llm-config configs/llm.yaml
```

**Standard Mode**（4个模拟审稿人，较慢）：



```bash
python scripts/run_baseline.py \
    --mode standard \
    --input data/processed/deepreview_processed.json \
    --output experiments/baseline_standard_full.json \
    --llm-config configs/llm.yaml
```

**Best Mode**（检索增强，最慢）：



```bash
python scripts/run_baseline.py \
    --mode best \
    --input data/processed/deepreview_processed.json \
    --output experiments/baseline_best_full.json \
    --llm-config configs/llm.yaml
```

### Step 3: 运行 Our Enhanced Method



```bash
python scripts/run_ours.py \
    --input data/processed/deepreview_processed.json \
    --output experiments/ours_full.json \
    --llm-config configs/llm.yaml
```

### Step 4: 评估（对比 Ground Truth）

对于 **内置预测**（数据集里已有 pred_fast_mode / pred_standard_mode / pred_best_mode）：



```bash
python -m src.cli.main evaluate \
    --input data/raw/deepreview_full.json \
    --official \
    --output-dir experiments/evaluation_official
```

对于 **新生成的预测**，需要写一个简单的评估脚本把预测结果和 processed 数据合并后评估。示例：



```python
import json
from src.evaluation.official_eval import evaluate_predictions

# 加载 processed 数据（含 ground_truth）
with open("data/processed/deepreview_processed.json") as f:
    samples = json.load(f)

# 加载预测结果
with open("experiments/baseline_fast_full.json") as f:
    preds = json.load(f)

# 合并
for i, sample in enumerate(samples):
    if i < len(preds):
        sample["pred_scores"] = preds[i].get("scores", {})

# 评估
metrics = evaluate_predictions(samples, pred_field="pred_scores")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
```

------

## 五、关键注意事项

| 问题            | 解决方案                                                     |
| --------------- | ------------------------------------------------------------ |
| **显存不足**    | 双卡16GB跑 `deepseek-r1:14b-16k` 应该可以，但如果 OOM，换更小的模型或降低 `context_window` |
| **运行时间**    | 完整数据集（假设 1000 条）× Fast Mode 约需 10-30 小时。建议先跑 `--samples 10` 验证 |
| **中断恢复**    | 代码每 5 条样本自动保存，中断后重新运行会覆盖。如需断点续跑，需手动修改脚本跳过已完成样本 |
| **Ollama 超时** | `src/utils/llm_wrapper.py` 中 timeout=300，Standard/Best Mode 输出很长，建议改为 600+ |
| **解析失败**    | 查看 `experiments/logs/deepreview.log` 定位问题              |

### 快速验证命令（先跑 5 条测试）



```bash
python scripts/run_baseline.py --mode fast --samples 5
python scripts/run_ours.py --samples 5
```

确认输出正常、评分可解析后，再去掉 `--samples` 跑全量。

------

## 六、目录结构（运行后）



```
experiments/
├── baseline_fast_full.json
├── baseline_standard_full.json
├── baseline_best_full.json
├── ours_full.json
├── evidence_results.json
└── logs/
    └── deepreview.log
```

需要我帮你写一个**自动断点续跑脚本**或者**多进程并行加速脚本**吗？完整数据集串行跑会非常慢。





## 一、关于 `raw_path` / `processed_path` 的作用

这两行配置在 `configs/deepreview.yaml` 里，但代码中**并没有自动读取这个 yaml 来决定路径**。它们只是文档性质的配置，实际路径由 CLI 命令的 `--input` 参数决定。

- `raw_path`: 原始数据位置。之前指向 `data/raw/sample.json`（3条样本的JSON），现在改为 `data/deepreview/test_2024.csv`
- `processed_path`: 预处理后的中间文件（统一内部格式）。代码会把 CSV/JSON 转换成标准格式后存到这里，后续 baseline/evaluate 都读这个文件

**之前的问题**：代码只支持 JSON，你的数据集是 CSV 格式（`data/deepreview/test_2024.csv` 和 `test_2025.csv`），直接跑会报错。

**已修复**：修改了 `src/adapters/deepreview_adapter.py`，现在 `load_and_adapt` 会自动检测文件扩展名，`.csv` 会走 CSV 解析逻辑，`.json` 走原来的 JSON 逻辑。

------

## 二、你的数据集格式分析

| 文件                            | 条数 | 列                                                           |
| ------------------------------- | ---- | ------------------------------------------------------------ |
| `data/deepreview/test_2024.csv` | 652  | inputs, outputs, year, id, mode, rating, decision, reviewer_comments |
| `data/deepreview/test_2025.csv` | 634  | 同上                                                         |

- `inputs`: 包含 system prompt + user prompt（LaTeX 论文全文）
- `outputs`: 包含 AI 生成的完整评审结果（\boxed_simreviewers{} 格式）
- `rating`: 多个人类评审者的评分列表，如 `[5,6,6,5]`
- `decision`: Accept / Reject

**注意**：这个数据集里的 `outputs` 已经是 AI 预测结果，可以直接提取评分做评估，**不需要再调 LLM**。

------

## 三、服务器运行命令

### 方式 1：直接评估已有 AI 预测（最快，无需 GPU/LLM）

适合验证数据集质量，或对比已有预测效果：

```bash
cd d:/Experiment/HyperGraph2

# 评估 2024 数据
python scripts/run_csv_eval.py --input data/deepreview/test_2024.csv --output experiments/eval_2024.json

# 评估 2025 数据
python scripts/run_csv_eval.py --input data/deepreview/test_2025.csv --output experiments/eval_2025.json

# 只评估前 10 条测试
python scripts/run_csv_eval.py --input data/deepreview/test_2024.csv --max-samples 10
```

**实测结果**：

- 2024（652条）：Rating MSE=0.0006, Decision Accuracy=1.0
- 2025（634条）：Rating MSE≈0, Decision Accuracy=0.79

### 方式 2：预处理为 JSON（供后续 baseline/ours 使用）



```bash
# 预处理 2024
python -m src.cli.main preprocess \
    --input data/deepreview/test_2024.csv \
    --output data/processed/test_2024_processed.json

# 预处理 2025
python -m src.cli.main preprocess \
    --input data/deepreview/test_2025.csv \
    --output data/processed/test_2025_processed.json
```

### 方式 3：用我们的方法重新生成预测（需要 LLM）

如果你不想用数据集里已有的 outputs，想用自己的方法重新跑：

**先确保服务器 Ollama 已启动并拉取了模型：**



```bash
ollama pull deepseek-r1:14b-16k
ollama serve
```

**修改 `configs/llm.yaml`** 确认模型配置正确，然后在 `src/scoring/baseline.py` 第 428 行后添加：



```python
llm.set_model("cloud_default")
```

**运行 baseline（Fast Mode，单审稿人）：**



```bash
python scripts/run_baseline.py \
    --mode fast \
    --input data/processed/test_2024_processed.json \
    --output experiments/baseline_fast_2024.json \
    --llm-config configs/llm.yaml
```

**运行 Our Enhanced Method：**



```bash
python scripts/run_ours.py \
    --input data/processed/test_2024_processed.json \
    --output experiments/ours_2024.json \
    --llm-config configs/llm.yaml
```

------

## 四、关键注意事项

| 问题             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| CSV 字段过大     | 已修复，`adapt_csv_and_adapt` 自动扩展了 `csv.field_size_limit` |
| 评分解析容错     | 已修复，`get_average_scores` 遇到非数字内容会跳过而不是崩溃  |
| 2025 有 1 条跳过 | `test_2025.csv` 有 1 条 outputs 解析失败，属于正常数据噪声   |
| 完整数据量       | 2024+2025 共 1286 条，Fast Mode 预计需 20-50 小时，建议开 tmux |
| 显存             | `deepseek-r1:14b-16k` 双卡 16GB 够用，如 OOM 换 `qwen3.5:4b` |

------

需要我帮你写一个**合并 2024+2025 的脚本**，或者**断点续跑脚本**（防止服务器中断后从头开始）吗？