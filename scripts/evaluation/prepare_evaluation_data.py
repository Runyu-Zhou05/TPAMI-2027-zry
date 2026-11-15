#!/usr/bin/env python3
"""Download and checksum the six paper evaluation datasets for VLMEvalKit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "MMBench_DEV_EN_V11": {
        "url": "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv",
        "md5": "30c05be8f2f347a50be25aa067248184",
    },
    "SEEDBench_IMG": {
        "url": "https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv",
        "md5": "68017231464752261a2526d6ca3a10c0",
    },
    "VStarBench": {
        "url": "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv",
        "md5": "b18854d7075574be06b631cd5f7d2d6a",
    },
    "RealWorldQA": {
        # The OpenCompass certificate has expired as of 2026-11-15. curl is
        # allowed to bypass TLS only for this file because the official MD5 is
        # pinned and checked before publication.
        "url": "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv",
        "md5": "4de008f55dc4fd008ca9e15321dc44b7",
        "curl_insecure": True,
    },
    "BLINK": {
        "url": "https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv",
        "md5": "3b6649b6a662184ea046908e5506260e",
    },
    "MME-RealWorld-Lite": {
        "url": "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv",
        "md5": "4c17057d7d3b6c4a0d4397c3dae0881c",
    },
}


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download_one(name: str, root: Path) -> dict:
    spec = DATASETS[name]
    target = root / f"{name}.tsv"
    if target.is_file() and digest(target, "md5") == spec["md5"]:
        status = "reused"
    else:
        temporary = target.with_suffix(".tsv.download")
        temporary.unlink(missing_ok=True)
        print(f"Downloading {name} from {spec['url']}", flush=True)
        curl = shutil.which("curl")
        if curl is not None:
            curl_command = [curl, "--fail", "--location", "--retry", "3"]
            if spec.get("curl_insecure", False):
                curl_command.append("--insecure")
            subprocess.run(
                [*curl_command, "--output", str(temporary), spec["url"]],
                check=True,
            )
        elif spec.get("curl_insecure", False):
            raise SystemExit(f"curl is required to download checksum-pinned {name}")
        else:
            urllib.request.urlretrieve(spec["url"], temporary)
        actual_md5 = digest(temporary, "md5")
        if actual_md5 != spec["md5"]:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"{name} MD5 mismatch: expected {spec['md5']}, got {actual_md5}")
        os.replace(temporary, target)
        status = "downloaded"
    return {
        "bytes": target.stat().st_size,
        "md5": digest(target, "md5"),
        "path": str(target),
        "sha256": digest(target, "sha256"),
        "status": status,
        "url": spec["url"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmu-data-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = args.lmu_data_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.benchmarks))
    files = {}
    for name in selected:
        target = root / f"{name}.tsv"
        if args.verify_only:
            if not target.is_file():
                raise SystemExit(f"Missing evaluation dataset: {target}")
            actual_md5 = digest(target, "md5")
            if actual_md5 != DATASETS[name]["md5"]:
                raise SystemExit(f"{name} MD5 mismatch: expected {DATASETS[name]['md5']}, got {actual_md5}")
            files[name] = {
                "bytes": target.stat().st_size,
                "md5": actual_md5,
                "path": str(target),
                "sha256": digest(target, "sha256"),
                "status": "verified",
                "url": DATASETS[name]["url"],
            }
        else:
            files[name] = download_one(name, root)

    manifest_path = root / "six_benchmark_manifest.json"
    existing_files = {}
    if manifest_path.is_file():
        existing_files = json.loads(manifest_path.read_text()).get("benchmarks", {})
    existing_files.update(files)
    manifest = {
        "benchmarks": existing_files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "MMBench": "MMBench_DEV_EN_V11 (paper reports DEV-EN)",
            "MME": "MME-RealWorld-Lite, not standard MME",
            "SEEDBench": "SEEDBench_IMG, nine image-understanding dimensions",
            "VStar": "VStarBench standard VLMEvalKit MCQ evaluation",
        },
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
