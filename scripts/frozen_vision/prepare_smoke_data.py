#!/usr/bin/env python3
"""Create deterministic Rotation FVE smoke datasets outside the checkout."""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

from fve_runtime import DEFAULT_OUTPUT_ROOT, ROOT, resolve_path


def write_prefix(source, output, rows):
    parquet = pq.ParquetFile(source)
    available_rows = parquet.metadata.num_rows
    if available_rows < rows:
        raise SystemExit(
            f"Cannot build {output.name}: requested {rows} rows but {source} contains only {available_rows}"
        )
    batch = next(parquet.iter_batches(batch_size=rows))
    table = pa.Table.from_batches([batch]).slice(0, rows)
    if len(table) != rows:
        raise RuntimeError(f"Internal smoke-data row-count mismatch: requested {rows}, built {len(table)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")
    print(f"wrote {len(table)} rows to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("FVE_DATA_ROOT", str(ROOT / "our_datasets")))
    parser.add_argument("--output-root", default=os.environ.get("FVE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    args = parser.parse_args()

    source = resolve_path(args.data_root) / "MMBench_RotationQA_90degree" / "train.parquet"
    output = resolve_path(args.output_root) / "smoke_data"
    # Ten examples support every currently allowed quick-smoke GPU count.
    write_prefix(source, output / "quick.parquet", 10)
    write_prefix(source, output / "production_shape.parquet", 512)


if __name__ == "__main__":
    main()
