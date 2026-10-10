# Configuration

Nimbus reads its configuration from environment variables.

## Schedule

`NIMBUS_SCHEDULE` sets when backups run, as a cron expression. The default is
`0 2 * * *`, every night at 02:00 local time.

## Retention

`NIMBUS_RETENTION_DAYS` sets how long snapshots are kept. The default is 30 days.
Snapshots older than the retention period are pruned every Sunday at 04:00.

## Bandwidth

`NIMBUS_MAX_UPLOAD_MBPS` limits the upload rate of an agent in megabits per second.
The default is 0, which means unlimited.

## Encryption

Snapshots are encrypted with AES-256-GCM before they leave the agent. The encryption
key is derived from the passphrase in `NIMBUS_PASSPHRASE`. If the passphrase is lost,
the snapshots cannot be recovered by anyone, including the vault administrator.
