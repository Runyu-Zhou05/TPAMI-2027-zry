import argparse
import math
import os
import random
from typing import Any

from PIL import Image

from image_utils import decode_image


DEFAULT_GRANULARITY = 90
DEFAULT_SEED = 42
SUPPORTED_GRANULARITIES = (45, 90)


def build_query_template(granularity: int) -> str:
    """Build the RotationQA prompt for one supported angle granularity."""
    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported rotation granularity: {granularity}")
    return f'''<image><image>

These are two images. The second image is a rotated version of the first image. 
Please determine how many degrees the second image has been rotated **counter-clockwise** relative to the first image.
The granuality of rotation is {granularity} degrees, meaning the second image could have been rotated by 0, {granularity}, {2 * granularity}, ..., {360 - granularity} degrees.

You must reason step-by-step and then provide the final answer.

The output **must strictly follow** this format:
<think> your reasoning here </think> <answer>number_of_degrees</answer>'''


def possible_angles(granularity: int) -> tuple[int, ...]:
    """Return all valid answers for a rotation granularity."""
    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported rotation granularity: {granularity}")
    return tuple(range(0, 360, granularity))


def output_directory(granularity: int) -> str:
    """Return the dataset directory without creating it."""
    possible_angles(granularity)
    return f'datasets/MMBench_RotationQA_{granularity}degree'


def rotatedRectWithMaxArea(w, h, angle):
    # borrowed from https://stackoverflow.com/a/16778797/13490627
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(wr), int(hr)


def build_rotated_datum(
    datum: dict[str, Any], angle: int, granularity: int, min_size: int = 28
) -> dict[str, Any] | None:
    """Create one RotationQA record from an explicit, validated angle."""
    if angle not in possible_angles(granularity):
        raise ValueError(f"Angle {angle} is invalid for granularity {granularity}")

    img: Image.Image = decode_image(datum['image'])
    rotated_img: Image.Image = img.rotate(angle, expand=True)
    wr, hr = rotatedRectWithMaxArea(img.width, img.height, math.radians(angle))
    rotated_img = rotated_img.crop((
        (rotated_img.width - wr) / 2,
        (rotated_img.height - hr) / 2,
        (rotated_img.width + wr) / 2,
        (rotated_img.height + hr) / 2,
    ))

    if rotated_img.width < min_size or rotated_img.height < min_size:
        return None

    return {
        'index': datum['index'],
        'source': datum['source'],
        'split': datum['split'],
        'height': rotated_img.height,
        'width': rotated_img.width,
        'images': [img, rotated_img],
        'query': build_query_template(granularity),
        'answer': str(angle),
    }


def process_batch(
    base,
    start: int,
    end: int,
    prefix: str,
    processor_id: int,
    seed: int,
    min_size: int = 28,
    granularity: int = DEFAULT_GRANULARITY,
):
    results = []
    rng = random.Random(seed)
    angles = possible_angles(granularity)

    for j in range(start, end):
        if str(j - start) == os.environ.get('EARLY_EXIT', str(end + 100)):
            break

        datum = base[j]
        angle = rng.choice(angles)
        new_datum = build_rotated_datum(datum, angle, granularity, min_size)
        if new_datum is not None:
            results.append(new_datum)

    save_dir = output_directory(granularity)
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
    seed: int,
    num_processors: int = 10,
    granularity: int = DEFAULT_GRANULARITY,
):
    total = len(base)
    batch_size = (total + num_processors - 1) // num_processors

    jobs = []
    for i in range(num_processors):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            continue
        jobs.append((start, end, prefix, i))

    print("Starting batch processing...")
    for i, (start, end, split_name, processor_id) in enumerate(jobs):
        print(f"Processing batch {i + 1}/{len(jobs)}: items {start} to {end}")
        process_batch(
            base,
            start,
            end,
            split_name,
            processor_id,
            seed,
            granularity=granularity,
        )
        print(f"Batch {i + 1} completed")
    print(f"Successfully completed processing {prefix} split")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MMBench RotationQA data")
    parser.add_argument(
        '--granularity',
        type=int,
        choices=SUPPORTED_GRANULARITIES,
        default=DEFAULT_GRANULARITY,
        help="Rotation step in degrees (default: 90)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

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
            seed=DEFAULT_SEED,
            granularity=args.granularity,
        )
        print(f"Completed processing {split} split")


if __name__ == '__main__':
    main()
