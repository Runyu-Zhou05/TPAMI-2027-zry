"""Dependency-free contracts for weak and strong MMBench ContrastiveQA generation."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "build_benchmarks" / "makecontrastive_mmbench.py"
PREPROCESSOR = ROOT / "verl" / "preprocess.py"


def generator_tree() -> ast.Module:
    return ast.parse(GENERATOR.read_text())


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function: {name}")


class HardContrastiveGeneratorContractTest(unittest.TestCase):
    def test_weak_paths_remain_unchanged_and_strong_paths_are_independent(self):
        tree = generator_tree()
        self.assertEqual(
            literal_assignment(tree, "AUGMENTATION_CONFIGS"),
            {"weak": "augcfg.yaml", "strong": "augcfg_strong.yaml"},
        )
        self.assertEqual(
            literal_assignment(tree, "TARGET_PATHS"),
            {
                "weak": "datasets/MMBench-ContrastiveQA",
                "strong": "datasets/MMBench-ContrastiveQA_strong",
            },
        )

    def test_cli_defaults_to_weak_and_only_accepts_declared_modes(self):
        tree = generator_tree()
        argument = next(
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "--augmentation"
        )
        keywords = {keyword.arg: keyword.value for keyword in argument.keywords}
        self.assertEqual(ast.literal_eval(keywords["default"]), "weak")
        self.assertIsInstance(keywords["choices"], ast.Name)
        self.assertEqual(keywords["choices"].id, "AUGMENTATION_CONFIGS")

    def test_selected_mode_reaches_every_augmentation_and_output_selection(self):
        tree = generator_tree()
        augment_fn = function(tree, "augment")
        process_fn = function(tree, "process_split")

        self.assertEqual(ast.literal_eval(augment_fn.args.defaults[-1]), "weak")
        config_calls = [
            call
            for call in ast.walk(augment_fn)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "get_augmentation_config_path"
        ]
        self.assertEqual(len(config_calls), 1)
        self.assertEqual(config_calls[0].args[0].id, "augmentation")

        augmentation_calls = [
            call
            for call in ast.walk(process_fn)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "augment"
        ]
        self.assertEqual(len(augmentation_calls), 4)
        for call in augmentation_calls:
            keyword = next(keyword for keyword in call.keywords if keyword.arg == "augmentation")
            self.assertIsInstance(keyword.value, ast.Name)
            self.assertEqual(keyword.value.id, "augmentation")

        target_calls = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "get_target_path"
        ]
        self.assertTrue(
            any(
                call.args
                and isinstance(call.args[0], ast.Attribute)
                and isinstance(call.args[0].value, ast.Name)
                and call.args[0].value.id == "args"
                and call.args[0].attr == "augmentation"
                for call in target_calls
            )
        )

    def test_preprocess_preserves_strong_dataset_identity_as_data_source(self):
        target_paths = literal_assignment(generator_tree(), "TARGET_PATHS")
        strong_source = target_paths["strong"]
        self.assertIn("ContrastiveQA_strong", strong_source)

        preprocess_tree = ast.parse(PREPROCESSOR.read_text())
        data_source_values = [
            value
            for node in ast.walk(preprocess_tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "data_source"
        ]
        self.assertTrue(
            any(isinstance(value, ast.Name) and value.id == "data_source" for value in data_source_values),
            "preprocess must preserve the selected dataset path in each row's data_source",
        )


if __name__ == "__main__":
    unittest.main()
