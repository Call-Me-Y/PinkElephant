# 概率探针：复用 FTB 的测量思路

在同一个失败点固定错误动作 a_fail、参考正确动作 a_gold 和两个其他错误动作，分别计算它们在不同反馈上下文下的 log P。
候选动作只用于离线评分，不作为选择题或答案发送给任务模型。监控器仍然只读取已经执行的错误和公共工具 schema。

配置为 `configs/probability_pilot.json`，入口为 `scripts/probe_probability.py`。
目前实现直接读取 Transformers 权重；现有 GGUF Chat API 后端尚未实现指定输入动作的概率评分。
模型开源不代表服务接口会返回输入 log-probs，GGUF 需要另行实现其运行时的逐 token 评分适配。
GGUF 量化版和 Transformers 版本也不能混作同一个概率模型，比较各条件时必须固定权重、量化、模板与精度。

先使用已经下载的 Qwen2.5-0.5B-Instruct 做一个失败点的真实评分：

```powershell
Set-Location C:\Projects\AI\PE\step1
python scripts/probe_probability.py --limit 1 --conditions pre_clean verbatim positive_history --output results/probability-check-v1
```

完整 pilot 使用48个失败点、七个上下文条件、四个候选，共1344次候选前向评分，没有逐 token 采样。
已经在本机GPU上完成一批，结果位于 `results/probability-pilot-cuda-v1/report.md`。
复跑和汇总命令（已有完整结果会跳过评分）：

```powershell
python scripts/probe_probability.py --config configs/probability_pilot_cuda.json
python scripts/summarize_probability.py --output results/probability-pilot-cuda-v1
```

本批次与先前CPU单点使用同一权重文件，但硬件数值结果有差异，分别保存，比较条件时均使用本批GPU分数。
原CPU配置仍可用于其他机器：

```powershell
python scripts/probe_probability.py
```

上下文条件包含原五组，增加 `pre_clean`（无失败记录的成功前缀）和 `neutral`（记录失败动作但用中性观察替代报错）。
history=1 时，`pre_clean` 对应 FTB 的动作尝试之前；history=2 时它仍是两次失败之前的干净前缀，
所以此时的 gain 测量两条历史的总变化，不应称为“第二次失败的单次增益”。
neutral 条件使用 FTB 环境的中性观察；copy / semantic 是这一具体对照下的操作性分解，不是对认知机制的唯一证明。

评分实现使用完整前向，按因果位移取出候选 token 对应的 logits，用 FP32 log_softmax 后求和。
保持 chat template、generation prompt 和候选 tokenization 固定；不把思维过程作为隐藏条件先生成。
当前计量为候选 token 串前缀的概率，不含 EOS/换行停止符，不声称是“输出该完整动作后立刻停止”的概率。
`all_candidate_tokens_greedy` 仅表示该候选各 token 均为逐步 argmax，不验证其后的停止行为。
没有实现 KV 缓存加速，优先检查数值正确性；后续复用 FTB 缓存时需核对与完整前向结果一致。

结果 `scores.jsonl` 保存逐候选原始分数、实际上下文和候选集合，`metrics.csv` 输出：

| 指标 | 定义 / 解读 |
|---|---|
| failed_logp | 失败动作的 log P；相同动作在条件间越低，倾向越弱 |
| gain_vs_clean | log P_fail(clean) − log P_fail(condition)；正值表示相对干净前缀受抑制 |
| suppression_vs_verbatim | log P_fail(verbatim) − log P_fail(condition)；正值表示相对原始回显受抑制 |
| gold_minus_fail_logp | log P_gold − log P_fail；越高越偏向参考修复，不能跨任务直接等同能力分数 |
| p_repeat_candidate_set | 在四个固定候选上用 logsumexp 归一化得到的失败候选份额；不是自由生成的实际重复率 |
| copy_delta / semantic_delta | neutral−clean / verbatim−neutral；对 raw 条件满足 −gain = copy_delta + semantic_delta |

优先比较相同失败点、同一模型的条件差值。正确/错误动作长度不同会影响候选间的序列分数，
要同时查看 token 数、每 token log P 以及固定候选内的条件变化，不把长度归一化分数当成真实序列概率。
两个干扰动作在每个条件下保持不变；归一化结果受这套候选集合选择影响。

最小研究流程为：先看原始回显是否提高失败动作概率，再看正向卡片是否抑制失败并提高正确相对错误的优势，
最后用已有多步 runner 检查实际动作与任务成功。前两项成立也不能替代第三项。
主历史比较仍限定 history=2 的 positive_history vs positive_current；history=1 两者输入本来相同。
逐行分数不是独立样本。评分脚本输出逐项指标；新增汇总脚本验证完整配对，并按任务聚类，
输出10,000次bootstrap的均值与差值区间，以及错误类型子组。区间未做多重比较校正，pilot结果为探索性证据。

