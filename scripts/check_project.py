# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Breeze TTS Omni project
"""Check project documentation and source syntax without runtime dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "README.zh-CN.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
    "CODE_OF_CONDUCT.md", "THIRD_PARTY_NOTICES.md", "project-manifest.json",
    ".github/workflows/project-checks.yml", ".github/PULL_REQUEST_TEMPLATE.md",
    "docs/getting-started.md", "docs/api.md", "docs/architecture.md",
    "docs/validation.md", "docs/roadmap.md", "docs/releasing.md",
    "examples/speech_client.py", "scripts/package_source.py",
)


def local_path(value: str) -> Path:
    path = ROOT / value
    if not value or "\\" in value or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Invalid project path: {value}")
    return path


def main() -> int:
    errors = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Missing required file: {name}")
    try:
        manifest = json.loads((ROOT / "project-manifest.json").read_text(encoding="utf-8-sig"))
        integration = manifest["integration_files"]
        if len(integration) != len(set(integration)):
            errors.append("Duplicate integration files in manifest")
        for component in ("vllm-omni", "breeze-tts"):
            if not re.fullmatch(r"[0-9a-f]{40}", manifest["upstream"][component]["base_commit"]):
                errors.append(f"Invalid upstream base revision: {component}")
        for name in integration:
            if not local_path(name).is_file():
                errors.append(f"Missing integration file: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"Invalid project manifest: {exc}")
        integration = []

    docs = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    docs += sorted((ROOT / ".github").rglob("*.md"))
    # Upstream docs retain their own conventions; only validate project-owned docs here.
    for path in docs:
        source = path.read_text(encoding="utf-8-sig")
        source = re.sub(r"```.*?```", "", source, flags=re.DOTALL)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", source):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            if not destination.is_relative_to(ROOT) or not destination.exists():
                errors.append(f"Broken local link in {path.relative_to(ROOT)}: {target}")
        if source.count("\n# ") > 0:
            errors.append(f"Multiple top-level headings: {path.relative_to(ROOT)}")

    sources = []
    for directory in ("scripts", "examples", "tests"):
        sources.extend((ROOT / directory).rglob("*.py"))
    sources.extend(ROOT / name for name in integration if name.endswith(".py"))
    for path in sorted(set(sources)):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"Invalid Python source: {exc}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"Project checks passed: {len(docs)} documents, {len(set(sources))} Python files, "
          f"{len(integration)} integration files. Inference was not executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
