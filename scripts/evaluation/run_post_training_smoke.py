#!/usr/bin/env python3
"""Export an FVE checkpoint and run the validated MMBench smoke in one command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "scripts/evaluation/export_fve_checkpoint.py"
MMBENCH_SMOKE = ROOT / "scripts/evaluation/vlmeval_mmbench_smoke.py"
DEFAULT_OUTPUT_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv")


def executable(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Executable does not exist: {path}")
    return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", required=True, help="One physical GPU index for MMBench smoke")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("FVE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--training-python", default=os.environ.get("FVE_PYTHON", sys.executable))
    parser.add_argument("--eval-python", default=None)
    parser.add_argument("--source-tsv", type=Path, default=None)
    args = parser.parse_args()

    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in args.run_id):
        raise SystemExit("--run-id may contain only letters, digits, '.', '_' and '-'")
    if "," in args.gpu or not args.gpu.strip().isdigit():
        raise SystemExit("--gpu must be exactly one physical GPU index")

    output_root = args.output_root.expanduser().resolve()
    training_python = executable(args.training_python)
    eval_python = executable(
        args.eval_python or output_root / "tools" / "vlmevalkit" / "env" / "bin" / "python"
    )
    result_dir = output_root / "evaluation" / "post_training" / args.run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    export_result = result_dir / "export_result.json"

    subprocess.run(
        [
            training_python,
            str(EXPORTER),
            "--checkpoint",
            str(args.checkpoint.expanduser().resolve()),
            "--python",
            training_python,
            "--result-json",
            str(export_result),
        ],
        cwd=ROOT,
        check=True,
    )
    export_manifest = json.loads(export_result.read_text())
    exported_model = Path(export_manifest["exported_model"])

    smoke_dir = result_dir / "mmbench_smoke"
    smoke_command = [
        eval_python,
        str(MMBENCH_SMOKE),
        "--model-path",
        str(exported_model),
        "--lmu-data-root",
        str(output_root / "inputs" / "vlmeval"),
        "--output-dir",
        str(smoke_dir),
    ]
    if args.source_tsv is not None:
        smoke_command.extend(["--source-tsv", str(args.source_tsv.expanduser().resolve())])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with (result_dir / "mmbench_smoke.log").open("w", buffering=1) as log:
        subprocess.run(
            smoke_command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    smoke_manifest_path = smoke_dir / "smoke_manifest.json"
    if not smoke_manifest_path.is_file():
        raise SystemExit(f"MMBench smoke did not produce its manifest: {smoke_manifest_path}")
    smoke_manifest = json.loads(smoke_manifest_path.read_text())
    manifest = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "export": export_manifest,
        "gpu": args.gpu.strip(),
        "mmbench_smoke": smoke_manifest,
        "run_id": args.run_id,
    }
    manifest_path = result_dir / "post_training_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
