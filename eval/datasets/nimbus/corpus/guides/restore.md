# Restoring files

## Restoring a snapshot

List snapshots with `nimbus-agent snapshots`, then restore one with
`nimbus-agent restore <SNAPSHOT_ID> --target /tmp/restore`. Restores never overwrite
existing files unless `--overwrite` is passed.

## Partial restores

To restore a single folder, add `--path` followed by the folder inside the snapshot,
for example `--path /home/alex/projects`.
