# Kafka consumer runbook

## Consumer lag rising

Distinguish the three causes before acting; they have opposite fixes.

1. **Skew** -- lag concentrated on one or two partitions. Adding consumers does
   nothing: a partition is consumed by exactly one member.
   `kafka-consumer-groups --bootstrap-server $BROKERS --describe --group <group>`
   Fix the partition key, then drain the hot partition with a dedicated
   backfill consumer.
2. **Rebalance storm** -- group state cycling `PreparingRebalance`/`Stable`,
   `CommitFailedException` in the logs. Processing time per batch has exceeded
   `max.poll.interval.ms`. Lower `max.poll.records` first; it takes effect on
   restart and is reversible.
3. **Genuine throughput ceiling** -- lag rising evenly across partitions with a
   stable group. Scale consumers up to, but not beyond, the partition count.

## Consumer group commands

- State: `kafka-consumer-groups --describe --group <group> --state`
- Per-partition lag: `kafka-consumer-groups --describe --group <group>`
- Reset to a timestamp (destructive, requires the group to be stopped):
  `kafka-consumer-groups --reset-offsets --to-datetime <ts> --group <group> --topic <topic> --execute`
