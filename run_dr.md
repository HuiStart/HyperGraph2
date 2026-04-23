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