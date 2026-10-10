# Runbook: queue database failover

## Symptoms

- The `orbit_queue_lag_seconds` alert fires (lag above 120 seconds).
- Workers log `connection refused` for the queue database.

## Steps

1. Confirm the primary is down: `make db-status ENV=production`.
2. Promote the replica: `make db-promote ENV=production`. This takes about 30 seconds.
3. Restart the workers so they reconnect: `make restart-workers ENV=production`.
4. Open an incident ticket and link the alert.

Do not promote the replica if replication lag was above 60 seconds when the primary
failed: messages could be sent twice. Escalate to the database team instead.
