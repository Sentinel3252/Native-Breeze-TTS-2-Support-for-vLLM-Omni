# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Breeze TTS Omni project
"""Exercise source exports from a minimal already-exported snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.package_source import ROOT_FILES, package, safe_source


class SourcePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ROOT_FILES:
            (self.root / name).write_text("source\n", encoding="utf-8")
        self.manifest = {
            "name": "fixture", "upstream": {
                "vllm-omni": {"base_commit": "a" * 40},
                "breeze-tts": {"base_commit": "b" * 40},
            }, "integration_files": ["vllm-omni/native.py"],
        }
        self.write_manifest()
        self.inventory = {"upstream": self.manifest["upstream"], "files": {}}
        for name in ("vllm-omni/core.py", "breeze-tts/LICENSE", "breeze-tts/MODEL_LICENSE"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original source\n")
            self.inventory["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": 0o644}
        (self.root / "SOURCE_INVENTORY.json").write_text(json.dumps(self.inventory), encoding="utf-8")
        (self.root / "vllm-omni/native.py").write_bytes(b"native = True\n")
        for name in ("examples/.env", "examples/checkpoint.safetensors", "examples/audio.wav",
                     "examples/__pycache__/bad.pyc", "examples/.git/config"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded")
        self.output = self.root / "dist/source.zip"
        self.no_git = patch("scripts.package_source.git", side_effect=FileNotFoundError("No Git metadata"))
        self.no_git.start()
        self.addCleanup(self.no_git.stop)

    def write_manifest(self):
        (self.root / "project-manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_archive_contains_native_overlay_licenses_and_hash_inventory(self):
        report = package(self.root, self.output)
        with zipfile.ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertIn("vllm-omni/native.py", names)
            self.assertIn("breeze-tts/MODEL_LICENSE", names)
            self.assertFalse(any(name.startswith("examples/") for name in names))
            inventory = json.loads(archive.read("SOURCE_INVENTORY.json"))
            for name, entry in inventory["files"].items():
                self.assertEqual(entry["sha256"], hashlib.sha256(archive.read(name)).hexdigest())
        self.assertEqual(report["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertIn(report["sha256"], self.output.with_name("source.zip.sha256").read_text())

    def test_identical_source_yields_identical_archive(self):
        first = package(self.root, self.output)
        second = package(self.root, self.output.with_name("second.zip"))
        self.assertEqual(first["sha256"], second["sha256"])

    def test_existing_artifact_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            package(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_missing_integration_file_blocks_export(self):
        self.manifest["integration_files"].append("vllm-omni/missing.py")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing source"):
            package(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_unversioned_upstream_source_blocks_export(self):
        (self.root / "vllm-omni/core.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            package(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_manifest_cannot_include_checkpoint_material(self):
        self.manifest["integration_files"].append("examples/checkpoint.safetensors")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "excluded material"):
            package(self.root, self.output)

    def test_paths_cannot_escape_source_tree(self):
        for name in ("../outside.py", "/outside.py", "C:/outside.py", "..\\outside.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_source(self.root, name)


if __name__ == "__main__":
    unittest.main()
