# 首版监控器的输入输出约定

实现为确定性规则，不调用另一个模型。后续换成 LLM 监控器时沿用此约定。

输入：当前任务已经执行的轨迹、环境报告的错误类型、所有模型都可见的工具 schema。
输入不包含：gold_actions、参考后续步骤、任务隐藏成功判据、尚未观察到的世界实体值。

在最近一次成功调用之后，统计相同 `(已解析工具名, error_type)` 的失败次数。
一次失败提供字段/格式修复要求；至少两次提供“核对字段与格式，未知值先查询”的加强提示。
成功工具调用清空当前待处理窗口；这只是简单的窗口规则，不代表完整任务已经恢复。

输出 `schemas/correction_card.schema.json` 对应卡片。只允许工具名、schema 字段名、错误类别、
次数与一般修复要求。当前错误动作中的字符串、数值、完整调用及其语义改写均不作为模板插值。
可以保留来自公开工具文档的 YYYY-MM-DD、HH:MM 等格式要求。

示例（示意模板，不是实测结果）：

> Recovery guidance | tool: create_event | category: missing_argument | matching failures: 2
> Supply every required field in the chosen tool's signature, using the request and observed results.
> Required fields to complete: attendees. First reconcile the required fields and their formats with
> the tool signature; use a lookup if a required value is still unknown.

这里包含正确的约束，但没有提供完整正确调用、缺失字段的具体值或参考答案。
对当前窗口内的历史只做计数与提示强度调整，暂未学习可迁移的修复规则。

