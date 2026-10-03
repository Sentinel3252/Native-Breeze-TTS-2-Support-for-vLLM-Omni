# Security policy

This project is in development. There are no supported stable release branches
or promised security response times. Security fixes target the current source
tree; older snapshots may require upgrading.

## Reporting a vulnerability

If the hosting repository has GitHub private vulnerability reporting enabled,
use **Security → Report a vulnerability**. Do not assume that a private route
exists until the repository provides it.

If no private contact is configured, open a minimal issue requesting a secure
contact route. Omit exploit instructions, credentials, private model data,
and recordings. For a confirmed upstream vulnerability, use that upstream
project's security reporting channel.

A useful private report includes the affected revision, environment, impact,
and smallest reproducer. Maintainers should acknowledge and coordinate the
fix and disclosure through the private channel.

## Deployment boundary

The development launch examples bind to `127.0.0.1`. Network-facing deployments
need authentication, request limits, and appropriate access controls for
checkpoint and reference files. Review checkpoint code before enabling remote
code execution. Do not commit API keys, downloaded checkpoints, or private
audio to the repository.
