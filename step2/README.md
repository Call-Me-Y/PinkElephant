# Step2：根据证据与恢复进展生成反馈

追加实验见[事实重提＋旧正向提示](COMBINATION_EXPERIMENT.md)：包含重试编号、诊断头和原样组合三种对照。

本目录是与 `step1` 并列的第二阶段项目。[研究计划](PLAN.md)说明主假设、对照和7–8周安排，
[当前状态](STATUS.md)说明实际完成内容。代码和本阶段产物存放在此；只读复用step1的环境适配和推理后端。
模型仍读取公共的 `C:/Projects/AI/models`，不重复下载。

## 本版已实现

- 从用户目标解析ToolShed的六种已知表达，提取明确要求。
- 只从成功查询/读取与可确认的执行结果登记相关事实，记录来源编号；失败参数不自动变成事实。
- 识别尚缺证据和下一子目标；任意工具返回ok不会清空全部未完成要求。
- 历史区分未执行上次建议、执行但无可观察进展、取得可观察进展。
- 六组对照共享工具失败触发、模型和动作预算。
- 严格目标判据在开跑前固定：沿用step1补充核对函数，本阶段按该判据停止，保留上游分数。
- 逐动作保存提示、证据、卡片与环境结果，独立重放和任务聚类配对汇总。
- 从初始状态运行和收集每任务第一个模型自产失败的入口。

任务解析与下一步选择目前含手写领域规则；它们提供的领域知识可能解释收益。
本版没有实现直接LLM建议强基线、学习式选择器、通用目标解析、合法调用后的目标偏离触发器或AppWorld适配。

## 项目结构

```text
step2/
  PLAN.md / STATUS.md
  configs/                 开发、自然失败收集、API、本地9B和确认模板
  src/pe_feedback/
    goal.py                仅解析公开用户目标
    evidence.py            有来源的事实、失效处理和要求满足证据
    feedback.py            事实重提 / 目标引导 / 历史响应卡片
    models.py              现有本地权重或兼容Chat API
    experiment.py          同前缀续跑、从零运行、自然失败采集
    analysis.py            目标判定重放、任务聚类区间
    bridge.py              只读接入step1/FTB
  scripts/pe.py            统一入口
  tests/                   来源、表面不变性、证据敏感性、历史与重放检查
  schemas/ / prompts/ / templates/
  data/                    后续自然失败与独立确认数据
  results/                 本地结果，Git忽略
```

## 先看生成的卡片

使用现有Python环境即可，纯规则部分只依赖已有numpy：

```powershell
Set-Location C:\Projects\AI\PE\step2
python -m unittest discover -s tests -v
python scripts/pe.py demo
python scripts/audit_cards.py
```

输出 `results/cards/cards.md` 和 `cards.json`，覆盖48个开发失败点的三种新反馈。
这是规则输出，不是模型效果实验；给actor的文本与完整内部证据元数据分别保存。

## 对照条件

| 条件 | 给actor的失败信息 |
|---|---|
| drop | 删除失败动作和错误观察 |
| abstract | FTB原版抽象反馈 |
| positive_history | step1的类别模板+历史计数 |
| facts_only | 只重提目标要求和相关已确认事实 |
| grounded | 加入诊断、下一子目标和完成检查 |
| adaptive | 再根据上次建议的执行/进展调整表述 |

第一阶段所有新卡片只在最近一步工具失败时触发。目标偏离的检测与额外提醒机会，后续再单独验证。
不支持的目标表达回退到FTB抽象，明确记录supported=false，不猜隐藏答案。
沿用的目标核对仍未穷尽操作顺序和副作用等语义要求，不能等同完整人工审查。

## 使用已有0.5B模型

```powershell
python scripts/pe.py run --config configs/development.json --limit 2 --output results/connection-check-v1
python scripts/pe.py analyze --output results/connection-check-v1
python scripts/pe.py run --config configs/development.json
python scripts/pe.py analyze --output results/development-v1
```

完整开发集：48检查点×6条件=288轨迹，最多864次actor调用。GPU batch=4，FP16主干、FP32分数，greedy，三步。
依赖已有torch/transformers，不训练新权重。两条连通检查是同一任务的两个历史变体，不能作为统计结论。
step2按补充目标判据停止，与step1按上游判据停止的旧数据不能直接混合作配对比较。
代码、配置或依赖变化后选新输出目录；完整批次复跑会跳过完成项。

新卡片预算为256个模型token，超过预算时用成功观察编号指向完整值，不截断正文后冒充完整正文。
这是初始上限，80–120 token的压缩与长度消融留待后续；原有上下文和FTB基线的长度另外记录。

## 收集真正的模型自产失败

```powershell
python scripts/pe.py run --config configs/native_collect.json
python scripts/pe.py collect --source results/native-collection-v1/trajectories.jsonl
python scripts/pe.py run --config configs/native_development.json
python scripts/pe.py analyze --output results/native-development-v1
```

收集从初始世界开始运行的abstract基线，每任务固定取第一个工具失败，不按哪个方案能修复来选样本。
采集器拒绝将注入失败后的续跑轨迹称作“从零自然失败”。当前任务池仍来自已有开发世界，不是独立确认。
确认集模板 `confirm.example.json` 指向尚不存在的数据文件；须先冻结方案，再另建任务世界和固定数据。

## 本地9B和服务商API

沿用已有llama.cpp服务，可使用 `configs/local_server.example.json`；先核对服务模型ID及thinking参数。
兼容Chat API用 `configs/api.example.json`，环境变量如下：

```powershell
$env:PE_BASE_URL = 'https://YOUR-ENDPOINT/v1'
$env:PE_MODEL = 'YOUR-MODEL-ID'
$env:PE_API_KEY = 'YOUR-KEY'
python scripts/pe.py run --config configs/api.example.json --limit 2 --output results/api-check-v1
```

本程序不读取根目录.env，密钥只用于请求头，不写配置或日志；localhost请求沿用绕过代理逻辑。
API没有本地tokenizer时，新增卡片按1800个UTF-8字节限制，明确记录单位，不把它当成精确token预算。
跨服务比较应先统一预算口径。此阶段不需要log-probability；尚未实现监控器的额外LLM调用。

## 研究中的重要边界

监控器接口只接收goal字符串与已执行轨迹，拒绝把payload、参考动作或隐藏世界当事实来源。
evaluator为环境与任务判据所需可读私有数据；它的结果不用于在测试时挑选“最佳反馈”。
读取后源文件被成功修改/删除会使旧正文证据失效；第一版不保证修复误改源文件造成的不可逆损失。
公开成功回执不足以判断的目标会保守保持未完成；多重同名目标记录为歧义。
仅对六种已知目标表达有效；泛化到AppWorld时需要新的公开观测适配和强基线。
若facts_only与完整方法持平，应把收益归于事实重提，不能声称学到了更强的恢复策略。
