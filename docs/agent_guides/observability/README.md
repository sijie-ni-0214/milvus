# Observability - AI Agent Guides

This directory contains observability guides for AI agents working on Milvus.
Use these guides before changing logging, metrics, tracing, or related
configuration.

## Guides

| Guide | Use When |
|---|---|
| [mlog - AI Agent Logging Guide](logging.md) | Adding or changing application logs. Covers `mlog` usage, context requirements, fields, levels, and logging rules. |
| [Collection / VChannel 级 Prometheus 指标审计](collection-level-metrics-audit.md) | 审查直接 collection / VChannel 标签、区分 PChannel、核对基数风险和清理覆盖情况。 |
| [Collection / VChannel 级 Prometheus 指标降基数模式](collection-level-metrics-mode.md) | 配置 `full` / `aggregate` 模式，并逐项核对 71 个受影响指标修改前后的行为。 |
| [WAL Tracing](../streaming-system/wal/tracing.md) | Understanding or changing WAL trace span semantics across append, consume, transaction, broadcast, and replication paths. |

## Rules of Thumb

- Use `mlog` for all Milvus logs. Do not use `zap`, the old `pkg/log` package,
  the standard `log` package, or `fmt.Println` for runtime logging.
- Keep observability hot paths cheap. Avoid payload logging and
  high-cardinality metric labels.
- When debugging, start from the narrowest available evidence such as trace ID,
  request time window, node, collection, channel, or error message.
- Preserve compatibility for metric names, label sets, config keys, and log
  field names unless the task explicitly requires a breaking change.
- Add or update focused tests when changing observability behavior.
