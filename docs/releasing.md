# Source releases

The local workspace has an unversioned project root and two nested Git
checkouts. Adding those directories directly to a new root repository can
record embedded repositories without their source contents. Publish a clean
source export instead, preserving the local checkouts and their changes.

## Export a reviewable source archive

From the project root, with Python 3.10+ and Git available:

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python scripts/package_source.py
```

The packager writes `dist/breeze-tts-omni-source.zip` and a SHA-256 sidecar.
It refuses to overwrite existing artifacts; use `--output` for another name.
It collects tracked upstream source from each nested checkout, includes the
untracked integration files declared in `project-manifest.json`, and reads
their working-tree contents. It preserves component licenses, including the
reference model-license text. No Git metadata, downloaded weights, recordings,
credentials, cache directories, or local build results are included.

The embedded `SOURCE_INVENTORY.json` records content hashes and upstream bases.
The archive is deterministic for identical file contents, inventory, and source
selection. Untracked upstream files outside the integration manifest are
excluded; add intentional new integration files to the manifest before export.
The packager rejects symlinks, missing tracked files, and changed upstream
bases rather than silently creating an incomplete release.

For an already-exported tree without nested `.git` metadata, the packager
uses and verifies the embedded source inventory. A combined root Git checkout
is also supported: it derives tracked files beneath each component directory.

## Publish from the exported tree

Extract the archive into a new directory. Inspect its inventory and licenses,
then run the standalone checks again. If creating a public source repository,
initialize Git in that exported directory and commit the complete tree:

```bash
git init -b main
git add .
git commit -m "Initialize Breeze TTS 2 native integration"
```

Set the remote to the repository you own and publish when ready. This guide
does not create a remote repository or publish anything automatically.
The root `.github/` workflow and issue templates become active in that repository;
nested upstream workflows remain reference files.

## Before tagging a release

- Record checkpoint and dependency revisions with the validation report.
- Update the README support matrix to match the evidence.
- Preserve all upstream licenses and notices.
- Review the archive inventory and excluded local material.
- Configure a private security-reporting route and maintainer contact on the host.
- Describe compatibility changes, verification, and known limitations in release notes.

## Preserve the version series

The current development baseline is `0.1.0-dev`, recorded in
`project-manifest.json` and both READMEs. This names a source baseline; no
root-repository tag or formal release has been created. See the
[version roadmap](roadmap.md) and [change history](changelog.md).

Before the next code milestone, preserve a named source snapshot:

```bash
python scripts/package_source.py --output dist/breeze-tts-omni-0.1.0-dev-source.zip
```

Retain the matching `.sha256` sidecar and validation record. Existing archives
must remain unchanged; choose a new snapshot suffix if that filename already
exists. After importing an export into a root Git repository, commit the
baseline and create an annotated `v0.1.0-dev` tag there. Subsequent releases use
their own tags and archives; do not duplicate the source tree into version folders.

Update the manifest version, READMEs, change history and validation record
together when adopting a new development baseline or releasing a milestone.
Do not mark planned functionality as implemented in the change history.
During the `0.x` series, document API/configuration compatibility changes in
each release, including any migration steps.

Until GPU validation passes, publish source snapshots with a development or
pre-release designation. `0.1.0` requires the basic-support gates in the roadmap;
later version numbers do not waive those gates. No release date, measured
performance or stable compatibility contract is currently claimed.
