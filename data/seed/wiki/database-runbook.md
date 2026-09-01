# orders-db operations runbook

## Connection pool exhausted

Symptom: `ERR_5012` / `HikariPool-1 - Connection is not available, request timed
out after 30000ms`, HTTP 503 from the calling service, pool active count pinned
at max with a growing pending queue.

Pool exhaustion is almost never a pool sizing problem. It is a symptom: something
is holding connections open. Work outward from the database.

1. Find sessions waiting on locks and who is blocking them:
   `SELECT pid, now() - query_start AS duration, wait_event_type, left(query, 80) FROM pg_stat_activity WHERE wait_event_type = 'Lock' ORDER BY duration DESC;`
2. Identify the blocker:
   `SELECT pid, pg_blocking_pids(pid) FROM pg_stat_activity WHERE cardinality(pg_blocking_pids(pid)) > 0;`
3. If the blocker is a maintenance statement (`VACUUM FULL`, `CREATE INDEX`
   without `CONCURRENTLY`, `ALTER TABLE`), terminate it. Application traffic is
   never the right thing to kill first.
   `SELECT pg_terminate_backend(<pid>);`
4. Raising the pool ceiling is a mitigation, not a fix. It buys headroom for the
   retry surge; revert it once the blocker is gone.

Escalate to the database on-call if the blocking statement is application
traffic rather than maintenance -- that is a query plan problem, not a lock
problem.

## Safe schema changes

- `CREATE INDEX` on any table over 10M rows must use `CONCURRENTLY`.
- `VACUUM FULL` is banned on production. Use `VACUUM (ANALYZE)` for statistics
  and `pg_repack` inside the Sunday maintenance window to reclaim space.
- The migration role carries `statement_timeout = 5min`. Do not raise it.

## WAL volume filling up

Check replication slots first -- an inactive slot pins every WAL segment it has
not consumed:
`SELECT slot_name, active, pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)) AS retained FROM pg_replication_slots;`

Dropping an orphaned slot releases the retained WAL immediately:
`SELECT pg_drop_replication_slot('<slot>');`

## Deadlocks

Deadlocks mean two writers take row locks in inconsistent order. The fix is
always lock ordering, never retries. Everything writing to `orders` must sort
its updates by `order_id` ascending.
