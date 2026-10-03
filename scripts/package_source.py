# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Breeze TTS Omni project
"""Export complete source snapshots and the integration overlay without Git metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "README.zh-CN.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
    "CODE_OF_CONDUCT.md", "THIRD_PARTY_NOTICES.md", "project-manifest.json",
    ".gitignore", ".gitattributes",
)
OMIT_DIRS = {".git", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
             "checkpoints", "outputs", "results", "dist", "build", ".codex_deps", ".codex_deps5"}
OMIT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".wav", ".pcm", ".flac",
                 ".mp3", ".pem", ".key", ".pyc", ".pyo", ".log"}


def safe_source(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe source path: {name}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source outside project: {name}")
    if any((root / Path(*relative.parts[:n])).is_symlink() for n in range(1, len(relative.parts) + 1)):
        raise ValueError(f"Symlink sources are not supported: {name}")
    if not path.is_file():
        raise ValueError(f"Missing source: {name}")
    return path


def excluded(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(set(path.parts) & OMIT_DIRS) or path.suffix.lower() in OMIT_SUFFIXES
            or any(part.startswith((".venv-", ".env")) for part in path.parts))


def git(root: Path, component: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-c",
         f"safe.directory={component.as_posix()}", "-C", str(component), *args],
        stderr=subprocess.PIPE,
    )


def collect_sources(root: Path) -> tuple[dict[str, int], dict]:
    manifest = json.loads((root / "project-manifest.json").read_text(encoding="utf-8-sig"))
    files = dict.fromkeys(ROOT_FILES, 0o644)
    for folder in ("docs", ".github", "scripts", "tests", "examples"):
        for path in (root / folder).rglob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and not excluded(name):
                files[name] = 0o755 if path.suffix == ".sh" else 0o644

    inventory_path = root / "SOURCE_INVENTORY.json"
    previous = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else None
    for name, upstream in manifest["upstream"].items():
        component = root / name
        try:
            git_root = Path(git(root, component, "rev-parse", "--show-toplevel").decode().strip()).resolve()
            if git_root == component.resolve():
                head = git(root, component, "rev-parse", "HEAD").decode().strip()
                if head != upstream["base_commit"]:
                    raise ValueError(f"{name} HEAD differs from manifest; update and review the upstream base")
            records = git(root, component, "ls-files", "--stage", "-z").split(b"\0")
            for record in records:
                if not record:
                    continue
                info, raw_path = record.split(b"\t", 1)
                mode, _, stage = info.split()
                if stage != b"0" or mode not in {b"100644", b"100755"}:
                    raise ValueError(f"Unmerged, symlink, or embedded source entry: {raw_path!r}")
                path = name + "/" + raw_path.decode("utf-8")
                if not excluded(path):
                    files[path] = 0o755 if mode == b"100755" else 0o644
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            if previous is None:
                raise ValueError(f"Cannot inventory {name}: Git metadata or exported source inventory required") from exc
            if previous["upstream"] != manifest["upstream"]:
                raise ValueError("Exported source inventory and manifest upstream bases differ")
            for path, entry in previous["files"].items():
                if path.startswith(name + "/") and not excluded(path):
                    source = safe_source(root, path)
                    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                        raise ValueError(f"Exported source changed without Git metadata: {path}")
                    files[path] = entry["mode"]
    for path in manifest["integration_files"]:
        if excluded(path):
            raise ValueError(f"Integration manifest includes excluded material: {path}")
        files.setdefault(path, 0o644)
    for path in files:
        safe_source(root, path)
    return files, manifest


def package(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise FileExistsError(f"Release artifact already exists: {output}")
    files, manifest = collect_sources(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {"project": manifest["name"], "upstream": manifest["upstream"], "files": {}}
    created = False
    try:
        # Fixed metadata makes repeated exports of identical content deterministic.
        with output.open("xb") as handle:
            created = True
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, mode in sorted(files.items()):
                    content = safe_source(root, name).read_bytes()
                    inventory["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(), "mode": mode}
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.create_system = 3
                    entry.external_attr = (0o100000 | mode) << 16
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, content)
                entry = zipfile.ZipInfo("SOURCE_INVENTORY.json", date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{digest}  {output.name}\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise
    return {"archive": str(output), "source_files": len(files), "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/breeze-tts-omni-source.zip")
    args = parser.parse_args()
    try:
        report = package(ROOT, args.output)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Source export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
