# 项目模块架构与数据流文档

## 一、完整模块清单

### 1. 数据适配层 `src/adapters/`
| 文件 | 功能 |
| ---- | ---- |
| deepreview_adapter.py | 将 DeepReview 原始 sample.json 转换为项目内部统一格式，提取数字评分，计算 ground_truth 均值，处理多评审人结构 |

---

### 2. 预处理层 `src/preprocess/`
|      |                                                              |
| ---- | ---- |
|      | 解析 LaTeX 全文结构，提取章节（abstract、introduction、methods、results 等） |

---

### 3. 证据抽取层 `src/evidence/`
| 文件 | 功能 |
| ---- | ---- |
| extractor.py | EvidenceExtractor：规则抽取（正则匹配 LaTeX 结构）+ LLM 抽取证据；EvidenceMerger：证据合并与去重。证据类型包括 claim、method、experiment、reference、consistency，以及负面证据（missing/weak） |

---

### 4. 评分标准层 `src/rubric/`
| 文件 | 功能 |
| ---- | ---- |
| base.py | Rubric 基类接口 |
| fixed_rubric.py | DeepReview 固定 4 维度（Rating 1-10，Soundness/Presentation/Contribution 1-4/1-5） |

---

### 5. 超图层 `src/graph/`
| 文件 | 功能 |
| ---- | ---- |
| builder.py | 超图构建（节点编码、超边生成） |
| patterns.py | 模式超边定义（SoundnessPattern、ContributionPattern、PresentationPattern、ConsistencyPattern、RiskPattern） |
| retrieval.py | 图检索（按维度找相关证据） |
| consistency.py | 跨章节一致性检查 |

---

### 6. Agent 层 `src/agents/`
| 文件 | 功能 |
| ---- | ---- |
| evidence_agent.py | 证据抽取 Agent（调用 EvidenceExtractor） |
| scoring_agent.py | 核心：单维度评分 Agent，含 chain-of-thought 推理 + Ceiling Caps 防膨胀 |
| arbitration_agent.py | 仲裁 Agent：多维度评分聚合，高分歧时用 LLM 仲裁，低分歧用 median |
| explanation_agent.py | 解释生成 Agent（基于模板引用证据生成评审解释） |
| risk_agent.py | 风险判断 Agent（判断是否需要人工复核） |
| workflow.py | 编排器：LangGraph 风格工作流，串接 Evidence → Scoring(x4) → Arbitration → Risk |

---

### 7. 评分实现层 `src/scoring/`
| 文件 | 功能 |
| ---- | ---- |
| baseline.py | Baseline 1/2/3 实现：FastModeScorer、StandardModeScorer（多审稿人模拟）、BestModeScorer（检索增强） |
| aggregation.py | 分数聚合工具（均值、median、加权） |

---

### 8. 评估层 `src/evaluation/`
| 文件 | 功能 |
| ---- | ---- |
| metrics.py | 核心指标：MSE、MAE、Spearman、Decision Acc/F1(macro)、Pairwise Acc |
| official_eval.py | 与官方 evalate.py 对齐的评估流程 |

---

### 9. 工具层 `src/utils/`
| 文件 | 功能 |
| ---- | ---- |
| llm_wrapper.py | 统一 LLM 接口，支持 ollama 本地模型和 OpenAI 远程 API，4 种模式（evidence/scoring/arbitration/explanation） |
| parser.py | 解析 \boxed_review{} / \boxed_simreviewers{}，提取分数，round_to_step(0.05) |
| logger.py | 日志工具 |

---

### 10. CLI 入口 `src/cli/`
| 文件 | 功能 |
| ---- | ---- |
| main.py | 命令行入口 |

---

### 11. 运行脚本 `scripts/`
| 文件 | 功能 |
| ---- | ---- |
| run_baseline.py | 运行 Baseline（Fast/Standard/Best） |
| run_ours.py | 运行我们的方法（Evidence + Multi-Agent + Arbitration） |
| run_ablation.py | 消融实验 |
| evaluate_full.py | 统一评估脚本：逐样本对比 + 汇总指标 |
| evaluate_predictions.py | 单独评估 predictions |
| generate_comparison_table.py | 生成对比表格 |
| diagnose_scores.py | 诊断分数分布、长度相关性、Spearman 分析 |
| run_and_evaluate.py | 运行并自动评估 |
| run_csv_eval.py / check_csv.py | CSV 格式评估 |
| debug_llm.py | LLM 调试 |

---

### 12. 配置 `configs/`
| 文件 | 功能 |
| ---- | ---- |
| llm.yaml | LLM 接口配置（ollama/openai、temperature=0.4、max_tokens、4 种 prompt 模式） |
| deepreview.yaml | DeepReview 专用配置（维度定义、scale 范围） |

---

## 二、数据流概览
```plain
sample.json → Adapter → paper_context + ground_truth
    ↓
LaTeXParser → 章节结构
    ↓
EvidenceExtractor → 结构化证据列表
    ↓
Workflow 编排：
    EvidenceAgent → DimensionScoringAgent(x4) → ArbitrationAgent → RiskAgent
    ↓
    ExplanationAgent（生成解释）
    ↓
evaluate_full.py → MSE/MAE/Spearman/Decision F1/Pairwise Acc