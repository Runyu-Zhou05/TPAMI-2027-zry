"""Dependency-free checks that all imported verl source packages are shipped."""

from __future__ import annotations

import ast
import subprocess
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SourceTreeContractTest(unittest.TestCase):
    def test_every_verl_models_import_resolves_to_source(self):
        modules = set()
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("verl.models"):
                    modules.add(node.module)
                elif isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names if alias.name.startswith("verl.models"))
        self.assertGreaterEqual(len(modules), 18)
        missing = []
        for module in sorted(modules):
            relative = ROOT.joinpath(*module.split("."))
            if not (relative.with_suffix(".py").is_file() or (relative / "__init__.py").is_file()):
                missing.append(module)
        self.assertEqual(missing, [])

    def test_fve_ignore_rule_does_not_hide_python_package(self):
        source = ROOT / "verl" / "models" / "transformers" / "monkey_patch.py"
        visible = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", str(source.relative_to(ROOT))]
        )
        self.assertNotEqual(visible.returncode, 0, "verl/models is still ignored by .gitignore")
        generated = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-q", "models/generated-weight.bin"]
        )
        self.assertEqual(generated.returncode, 0, "root generated models directory is no longer ignored")

    def test_required_fve_modules_and_version_are_present(self):
        required = [
            ROOT / "verl/models/__init__.py",
            ROOT / "verl/models/registry.py",
            ROOT / "verl/models/transformers/monkey_patch.py",
            ROOT / "verl/models/transformers/qwen2_5_vl.py",
            ROOT / "verl/models/transformers/qwen2_vl.py",
            ROOT / "verl/models/weight_loader_registry.py",
        ]
        self.assertTrue(all(path.is_file() for path in required))
        self.assertEqual((ROOT / "verl/version/version").read_text().strip(), "0.4.1.dev")

    def test_setuptools_will_package_restored_modules(self):
        from setuptools import find_packages

        packages = set(find_packages(where=ROOT))
        self.assertIn("verl.models", packages)
        self.assertIn("verl.models.transformers", packages)
        self.assertIn("verl.models.mcore.qwen2_5_vl", packages)


if __name__ == "__main__":
    unittest.main()
