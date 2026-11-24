import argparse
import os
import random
from typing import Any

from PIL import Image

from image_utils import decode_image


DEFAULT_GRID_SIZE = 2
DEFAULT_SEED = 0
SUPPORTED_GRID_SIZES = (2, 3)

ORDER_EXAMPLES = {
    2: '3,2,1,4',
    3: '8,9,3,2,1,4,7,5,6',
}


def build_query_template(grid_size: int) -> str:
    """Build the JigsawQA prompt for one supported grid size."""
    if grid_size not in SUPPORTED_GRID_SIZES:
        raise ValueError(f"Unsupported jigsaw grid size: {grid_size}")
    patch_count = grid_size * grid_size
    return '<image>' * patch_count + f'''

The provided images represent {patch_count} parts of an original image, divided into a {grid_size}x{grid_size} grid.

Your task is to determine the correct order of these parts to reconstruct the original image. Starting from the top-left corner, proceed row by row, from left to right and top to bottom, to arrange the parts.

The output should be a string of numbers, separated by a comma, where each number corresponds to the original position of the patches in the restored image. For instance, "{ORDER_EXAMPLES[grid_size]}" would indicate the positions of the patches in the correct order.

Before providing the final result, you must reason through the puzzle step by step. Consider the relative placement of each part and how they fit together.

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>order</answer>'''


def output_directory(grid_size: int) -> str:
    """Return the dataset directory without creating it."""
    if grid_size not in SUPPORTED_GRID_SIZES:
        raise ValueError(f"Unsupported jigsaw grid size: {grid_size}")
    return f"datasets/MMBench_JigsawQA_{grid_size}x{grid_size}"


def inverse_order_answer(order: list[int] | tuple[int, ...]) -> str:
    """Encode where each original patch occurs in the shuffled image list."""
    if sorted(order) != list(range(len(order))):
        raise ValueError("Jigsaw order must be a permutation starting at zero")
    inverse_order = [0] * len(order)
    for shuffled_position, original_position in enumerate(order):
        inverse_order[original_position] = shuffled_position + 1
    return ','.join(map(str, inverse_order))


def build_jigsaw_datum(
    datum: dict[str, Any],
    order: list[int] | tuple[int, ...],
    grid_size: int,
    min_size: int = 28,
) -> dict[str, Any] | None:
    """Create one JigsawQA record from an explicit patch permutation."""
    patch_count = grid_size * grid_size
    if grid_size not in SUPPORTED_GRID_SIZES:
        raise ValueError(f"Unsupported jigsaw grid size: {grid_size}")
    if len(order) != patch_count:
        raise ValueError(f"A {grid_size}x{grid_size} grid needs {patch_count} order entries")
    answer = inverse_order_answer(order)

    orig: Image.Image = decode_image(datum['image'])
    patch_width = orig.width // grid_size
    patch_height = orig.height // grid_size
    if patch_width < min_size or patch_height < min_size:
        return None

    patches = []
    for row in range(grid_size):
        for column in range(grid_size):
            right = (column + 1) * patch_width
            lower = (row + 1) * patch_height
            patches.append(orig.crop((column * patch_width, row * patch_height, right, lower)))

    return {
        'index': datum['index'],
        'source': datum['source'],
        'split': datum['split'],
        'images': [patches[index] for index in order],
        'query': build_query_template(grid_size),
        'answer': answer,
    }


def process_batch(
    base,
    start: int,
    end: int,
    prefix: str,
    processor_id: int,
    min_size: int = 28,
    grid_size: int = DEFAULT_GRID_SIZE,
    rng: random.Random | None = None,
):
    results = []
    random_source = rng if rng is not None else random
    print(f"Processor {processor_id}: Starting batch from {start} to {end}")

    for row_index in range(start, end):
        datum = base[row_index]
        order = list(range(grid_size * grid_size))
        random_source.shuffle(order)
        new_datum = build_jigsaw_datum(datum, order, grid_size, min_size)
        if new_datum is not None:
            results.append(new_datum)

    print(f"Processor {processor_id}: Start saving {len(results)} items")
    save_dir = output_directory(grid_size)
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}_{processor_id:05d}.parquet'
    save_path = os.path.join(save_dir, parquet_name)

    if results:
        import datasets

        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        print(f"Processor {processor_id}: Saved {len(results)} items to {save_path}")
    else:
        print(f"Processor {processor_id}: No results to save")


def process_split(
    base,
    prefix: str,
    num_processors: int = 1,
    grid_size: int = DEFAULT_GRID_SIZE,
    rng: random.Random | None = None,
):
    total = len(base)
    batch_size = (total + num_processors - 1) // num_processors
    print(f"Processing {prefix} split: {total} items with {num_processors} batches, batch size: {batch_size}")

    jobs = []
    for i in range(num_processors):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            continue
        jobs.append((base, start, end, base.split, i))
    print(f"Created {len(jobs)} batches for {prefix} split")

    results = []
    print("Starting batch processing...")
    for i, (base, start, end, split_name, processor_id) in enumerate(jobs):
        print(f"Processing batch {i + 1}/{len(jobs)}: items {start} to {end}")
        result = process_batch(
            base,
            start,
            end,
            split_name,
            processor_id,
            grid_size=grid_size,
            rng=rng,
        )
        results.append(result)
        print(f"Batch {i + 1} completed: {result}")

    print(f"Successfully completed processing {prefix} split")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MMBench JigsawQA data")
    parser.add_argument(
        '--grid-size',
        type=int,
        choices=SUPPORTED_GRID_SIZES,
        default=DEFAULT_GRID_SIZE,
        help="Number of rows and columns in the puzzle (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rng = random.Random(DEFAULT_SEED)

    import datasets

    for split in ['test', 'validation']:
        print(f"Processing {split} split...")

        dataset_path = 'datasets/MMBench/en'
        print(f"Loading dataset from: {dataset_path}")

        base = datasets.load_dataset(dataset_path, split=split)
        print(f"Successfully loaded {split} split with {len(base)} items")

        process_split(
            base,
            split,
            num_processors=1,
            grid_size=args.grid_size,
            rng=rng,
        )
        print(f"Completed processing {split} split")


if __name__ == '__main__':
    main()
