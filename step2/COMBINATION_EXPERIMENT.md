# 事实重提＋旧正向提示：预定开发实验

已完成：288条轨迹、727次actor调用，全部重放通过。组合21/48，事实基线18/48；编号对照仅4/48，不能视为无害上下文变化。完整结果与限制见[结果解读](results/facts-positive-v1/interpretation.md)。以下保留运行前制定的条件和比较口径。

使用同一ToolShed开发集48个失败点（24个任务的两个历史变体），Qwen2.5-0.5B、FP16、greedy、三步、每步96 tokens。
全部六组重新生成，共288条轨迹；沿用相同目标判据和停止规则，不与旧批次混合计分。

| 条件 | 文本 |
|---|---|
| abstract | 原FTB抽象，复跑参考 |
| positive_history | 原历史正向提示，复跑参考 |
| facts_control | 原facts_only，重新运行的事实基线 |
| facts_retry | 相同事实块＋Retry index: N |
| facts_diagnosis | 相同事实块＋原旧提示第一行：工具、错误类别、次数 |
| facts_positive | 相同事实块＋旧render_card输出全文，逐字追加 |

N沿用旧正向提示的matching_failures：最近成功调用后，相同工具与错误类别的累计失败次数。
它不是整个任务的全局尝试次数。两种附加信息控制使用相同的N，不额外泄漏失败参数。

事实区每次仍调用原facts_only生成器，独立预算256 tokens，保持同一历史下的事实文本一致。
总上限统一增加至384 tokens，以容纳原样旧提示。不会为给附加文本腾位置而删减事实；超出则报错。
实际提示长度不相等，所以结果不能证明等长度条件下的纯语义效应；长度与调用成本另行保存。
本实验限定已有本地tokenizer口径，不直接迁移到API字节预算。

预定主比较：facts_positive vs facts_retry 的 strict_success。
辅助比较：组合vs事实基线、组合vs仅旧提示、组合vs诊断头。
单独看失败重复下降不足以说明恢复成功；同时核对相同输入重现、首步有效、目标通过与调用量。
对独立任务聚类bootstrap 10,000次，所有比较属探索，不把两个历史变体当独立样本。

运行：

```powershell
Set-Location C:\Projects\AI\PE\step2
python -m unittest discover -s tests -v
python scripts/pe.py run --config configs/facts_positive_v1.json
python scripts/pe.py analyze --output results/facts-positive-v1
python scripts/audit_combinations.py
```

新增模式不会覆盖旧结果；源码指纹更新后，旧批次继续保留用于阅读与分析，重新运行应使用新配置/输出目录。
若本批组合有收益，再在新的自然失败与独立任务世界确认。这个开发集已经多次使用，不承担最终证明作用。
