# PinkElephant 最小实验

下一阶段研究建议见[根据证据与恢复进展生成反馈](FEEDBACK_RESEARCH_PLAN.md)；该方案尚未实现，
与已完成的pilot结果分开记录。

[直接解码干预与小型排斥训练实验](INTERVENTIONS.md)：包含FTB字符串禁用、错误触发的局部约束、
三档回显增量对比解码，以及独立数据训练的SFT/UL输出头对照。

新增[概率探针入口与指标说明](PROBABILITY.md)：可对已有开源模型直接做 teacher-forced 候选动作评分，
与下面的实际动作评测配合使用。

目标：比较不同反馈方式之后，智能体**实际执行的下一步动作**与三步内的任务结果。
首版采用 FTB 的 ToolShed 环境，加一个基于错误历史的规则监控器。无需 log-probability，
无需训练模型，也不依赖额外的监控模型调用。

研究与周计划见 [PLAN.md](PLAN.md)，已完成的检查和实际运行边界见 [STATUS.md](STATUS.md)，
操作记录和结果写作模板见 `templates/`。
本目录独立存放项目代码，只读复用 `../literature/feedback-that-backfires` 的源码。
与 Sentry 使用同一个本地模型服务地址；FTB ToolShed 可直接在 Windows 跑，AppWorld 后续继续使用既有 WSL 环境。

| 条件 | 输入给任务模型的失败信息 | 用途 |
|---|---|---|
| verbatim | 失败动作 + 原始报错，FTB 原版 | 标准回显 |
| abstract | FTB 原版 abstract_failure | 最近邻基线；原模板包含 not repeatable 字样 |
| positive_current | 最新错误的规则化正向引导，移除所有失败动作和报错 | 检查正向引导 |
| positive_history | 同一引导，根据当前未解决窗口中同类失败计数加强要求 | 待验证方案 |
| verbatim_positive | 原始回显 + 与 positive_history 完全相同的引导 | 检查保留引导时动作隔离的影响 |

`positive_current` 与 `positive_history` 在只有一次失败时生成完全相同的输入；第二次同类失败
之后才有区别。首版“利用历史”仅包括同类失败计数和提示强度切换，没有跨任务经验学习。
工具名、schema 字段名和格式规范可见；失败调用的具体参数值和完整字符串不进入卡片。

## 已建立的目录

```text
step1/
  PLAN.md                       最小研究问题、判据、7–8 周安排
  configs/                      smoke / CPU / 本地服务 / API / 确认 / 自由运行
  data/                         冻结失败点与来源 manifest（含私有参考，不作为模型输入）
  schemas/                      纠错卡片 JSON schema
  prompts/                      监控器输入输出约定
  src/pinkelephant/
    ftb.py                      只读复用上游环境、错误扰动、消息模板
    dataset.py                  生成并验证相同失败状态
    feedback.py                 错误历史 → 正向纠错卡片
    backends.py                 本地 Transformers / 兼容 Chat API / 流程替身
    experiment.py               配对续跑、自由运行、逐步保存
    analysis.py                 动作指标、任务聚类区间、配对差值
  scripts/pe.py                  统一入口
  tests/                        失败状态重放、隔离、计数、指标检查
  templates/                    实验登记与结果报告模板
  results/                      本地产物，Git 忽略
```

## 先检查流程，不调用模型

在 PowerShell 中使用已有的 Python 3.12 环境：

```powershell
Set-Location C:\Projects\AI\PE\step1
python scripts/pe.py prepare --config configs/smoke.json
python scripts/pe.py prepare --config configs/local_pilot.json
python -m unittest discover -s tests -v
python scripts/pe.py run --config configs/smoke.json
python scripts/pe.py analyze --output results/smoke-v1
```

只依赖已有的 numpy；若环境缺失，可安装 `requirements.txt`。`smoke` 使用知道参考解的流程替身，
结果标记为 **plumbing_only**，不能充当模型效果证据。所有真实模型后端都收不到参考后续动作。

## 使用已经下载的小模型验证真实推理

```powershell
python scripts/pe.py run --config configs/cpu_quickcheck.json --limit 1
python scripts/pe.py analyze --output results/cpu-quickcheck-v1
```

使用已有的 `C:/Projects/AI/models/Qwen2.5-0.5B-Instruct`，不下载权重。
需要已有 torch / transformers。这里仅用一个失败点，五个条件各生成一个真实动作，只检查模型连通；
不适合做统计结论或三步恢复结论。`cpu_check.json` 另提供最多三步的检查，但CPU会更慢。
`MODEL_DIR` 可覆盖模型所在的公共目录。

## 使用本地 Qwen3.5-9B GGUF 服务跑第一张结果表

沿用现有启动脚本，在单独的 PowerShell 窗口运行：

```powershell
& C:\Projects\AI\PE\_shared\runs\sentry-appworld\start_local_model.ps1
```

确认服务加载的模型与 `/v1/models` 返回 ID 后，设置模型别名：

```powershell
$env:PE_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:PE_MODEL = 'Qwen3.5-9B'
python scripts/pe.py run --config configs/local_pilot.json --limit 2 --output results/local-check-v1
python scripts/pe.py run --config configs/local_pilot.json
python scripts/pe.py analyze --output results/local-pilot-v1
```

本地客户端自动绕过 localhost 的系统代理。GGUF 由现有服务加载，实验进程无需再次占用显存。
配置显式关闭 thinking；请先确认当前服务接受 `chat_template_kwargs.enable_thinking` 并保存服务启动参数。
若该服务版本接口不同，只调整配置后另建运行目录。出现空回复会停止，避免把缺失回复算成研究结果。
8G 显存下沿用已验证的服务参数；显存不足时优先减少 context 或 GPU offload，再重新登记配置。

首轮为 24 个独立任务 × 两种失败历史 × 五种条件 = **240 条续跑轨迹**，
每条最多三次模型调用，最多 **720 次调用**。首轮 temperature=0，每个条件只跑一次；
这不能估计采样随机性。先跑两个失败点检查输出格式和 token 预算，再跑完整数据。

## 使用服务商 API

复用相同数据和 runner。配置支持返回文本动作的兼容 `/chat/completions` 接口，
未实现各服务商的专用协议。选择已经可用的指令模型服务，将准确模型 ID 固定进实验登记表。

```powershell
$env:PE_BASE_URL = 'https://YOUR-ENDPOINT/v1'
$env:PE_MODEL = 'YOUR-EXACT-MODEL-ID'
$env:PE_API_KEY = 'YOUR-KEY'
python scripts/pe.py run --config configs/api_pilot.example.json --limit 2 --output results/api-check-v1
python scripts/pe.py run --config configs/api_pilot.example.json
python scripts/pe.py analyze --output results/api-pilot-v1
```

密钥仅从环境变量读取，不写入配置、manifest 或请求日志。此程序不自动读取根目录 `.env`。
先确认服务支持 temperature、输出长度字段与 thinking 设置。模板默认不发送 seed；支持时可以开启。
某些服务要求 `max_completion_tokens`，可修改 `token_limit_field`。实际回复中的 model、
system_fingerprint、usage、finish_reason 随轨迹保存。所有调用顺序在条件间固定随机化。
程序没有自动重试；报错会停止，已完成的轨迹保留。

## 自由运行和确认实验

```powershell
python scripts/pe.py run --config configs/free_pilot.json
python scripts/pe.py analyze --output results/free-pilot-v1
```

这轮不注入失败，从初始状态开始，让模型自行产生错误，24 个任务、五条件、每任务最多八步，
最多 960 次调用。任务仍由受控集选出，属于探索性的同分布验证，不能视为独立确认。

`confirm.example.json` 使用另一个世界 seed：60 个独立任务、两个历史深度、五条件、三个采样，
最多 5,400 次调用。**先根据 pilot 冻结模板、预算和判据，再生成并运行确认集。**
确认集没有沿用具体任务世界，但仍使用同一组任务模板，不能据此宣称跨领域泛化。

## 每次运行会保存什么

- `manifest.json`：配置、解析后的模型地址/ID、数据和源码指纹、上游版本。
- `trajectories.jsonl`：每步真实提示、原始模型文本、实际执行动作、环境反馈、卡片与审计。
- `requests.jsonl` / `errors.jsonl`：请求尝试与出错位置；费用以提供商账单为准。
- `summary.csv` / `paired_contrasts.csv`：按历史深度汇总指标、任务聚类 bootstrap 与配对差值。
- `report.md` / `cost_and_audit.json` / `manual_review.csv`：可汇报结果、成本与人工样例审查。

同一个结果目录可断点继续；配置、数据、上游或本项目源码变化后会拒绝混写。
中断时未完成的单条轨迹会重新跑，请求日志保留已发出的调用，调用上限按尝试次数累计。
真实模型实验目录中包含完整的内部失败轨迹，用于研究审计；“隔离”约束针对给任务模型的反馈通道。

