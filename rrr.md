写好了，就一个命令：
```bash
python scripts/evaluate_full.py \
    --ground-truth data/processed/test_2024_processed.json \
    --predictions experiments/baseline_fast_2024.json \
    --output experiments/full_eval.json
```
只跑前 5 条测试：
```bash
python scripts/evaluate_full.py -g data/processed/test_5.json -p experiments/baseline_fast_2024.json -n 5
```

**输出效果：**
```
====================================================================================================
Sample   ID              Rating       Soundness    Presentation   Contribution   Decision        
====================================================================================================
1/652    w7BwaDHppp      5.50/7.00    2.75/3.00    2.75/3.00      2.25/3.00      Accept/Accept   
2/652    u4CQHLTfg5      3.00/7.00    2.67/3.00    2.00/3.00      1.67/3.00      Reject/Accept   
...
====================================================================================================
Matched: 652, Skipped: 0

============================================================
Final Metrics
============================================================
| Rating MSE | 3.2456 |
| Rating MAE | 1.5234 |
...
| Decision Accuracy | 0.7896 |
| Decision F1 (macro) | 0.7234 |
```


如果你跑的 predictions 文件 pred 列还全是 `-`，把那条命令的输出贴给我，我帮你定位是 LLM 没调通还是 parser 问题。