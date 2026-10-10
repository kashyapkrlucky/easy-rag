# Troubleshooting

## Error N-104: repository locked

Another backup of the same agent is still running. Wait for it to finish; a stale lock
is released automatically after 6 hours.

## Error N-207: version mismatch

The agent is newer than the vault. Upgrade the vault first, then retry.

## Error N-311: passphrase rejected

The passphrase in `NIMBUS_PASSPHRASE` does not match the one used for earlier snapshots.
Restore the original passphrase; snapshots cannot be re-encrypted with a new one.

## Slow backups

The first backup copies everything and can take hours. Later backups only send changed
blocks of 4 MB. If backups stay slow, check `NIMBUS_MAX_UPLOAD_MBPS`.
