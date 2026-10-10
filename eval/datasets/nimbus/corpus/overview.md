---
title: Nimbus Overview
---
# Nimbus Overview

Nimbus is a self-hosted backup service for small teams. It copies selected folders to an
encrypted repository on a schedule and keeps a history of snapshots that can be restored
individually.

## Components

Nimbus has two processes. The **nimbus-agent** runs on every machine that is backed up
and reads the files. The **nimbus-vault** stores snapshots and serves restores. Agents
connect to the vault over TLS on port 7443.

## Supported platforms

The agent runs on Linux (x86_64 and arm64) and macOS 13 or newer. The vault runs on
Linux only and needs at least 2 GB of RAM.
