# Installation

## Installing the vault

Download the vault package and start it with `nimbus-vault serve --data /srv/nimbus`.
On first start the vault prints a one-time enrollment token, valid for 30 minutes.

## Installing an agent

Install the agent package, then enroll it with
`nimbus-agent enroll --vault https://backup.example.internal:7443 --token <TOKEN>`.
An enrolled agent appears in the vault dashboard within one minute.

## Upgrading

Upgrade the vault before the agents. A vault accepts agents up to two minor versions
older than itself; agents newer than the vault are rejected with error N-207.
