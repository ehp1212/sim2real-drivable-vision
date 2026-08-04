from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "office": PROJECT_ROOT / "sdg/outputs/office/run_001",
    "hospital": PROJECT_ROOT / "sdg/outputs/hospital/run_001",
    "warehouse": PROJECT_ROOT / "sdg/outputs/warehouse/run_001",
    "real_room": PROJECT_ROOT / "sdg/outputs/real_room/run_001",
}

OUTPUT_ROOT = PROJECT_ROOT / "dataset/drivable"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


@dataclass(frozen=True)
class Sample:
    environment: str
    stem: str
    image_path: Path
    mask_path: Path


def collect_files(
    directory: Path,
    allowed_extensions: set[str],
) -> dict[str, Path]:
    """Collect files indexed by filename stem."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: {directory}"
        )

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        if path.stem in files:
            raise RuntimeError(
                f"Duplicate filename stem '{path.stem}' in {directory}"
            )

        files[path.stem] = path

    return files


def collect_samples(
    environment: str,
    source_root: Path,
) -> list[Sample]:
    """Match every RGB image with its semantic mask."""

    images = collect_files(
        source_root / "rgb",
        IMAGE_EXTENSIONS,
    )

    masks = collect_files(
        source_root / "masks",
        {".png"},
    )

    image_stems = set(images)
    mask_stems = set(masks)

    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)

    if missing_masks:
        raise RuntimeError(
            f"{environment}: {len(missing_masks)} image(s) have no mask. "
            f"Examples: {missing_masks[:5]}"
        )

    if missing_images:
        raise RuntimeError(
            f"{environment}: {len(missing_images)} mask(s) have no image. "
            f"Examples: {missing_images[:5]}"
        )

    return [
        Sample(
            environment=environment,
            stem=stem,
            image_path=images[stem],
            mask_path=masks[stem],
        )
        for stem in sorted(image_stems)
    ]


def split_samples(
    samples: list[Sample],
    rng: random.Random,
) -> dict[str, list[Sample]]:
    """Create a deterministic 80/10/10 split."""

    shuffled = samples.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * SPLIT_RATIOS["train"])
    val_count = int(total * SPLIT_RATIOS["val"])

    return {
        "train": shuffled[:train_count],
        "val": shuffled[
            train_count : train_count + val_count
        ],
        "test": shuffled[
            train_count + val_count :
        ],
    }


def create_output_structure() -> None:
    """Create YOLO semantic dataset directories."""

    for data_type in ("images", "masks"):
        for split in SPLIT_RATIOS:
            directory = OUTPUT_ROOT / data_type / split
            directory.mkdir(parents=True, exist_ok=True)

            gitkeep = directory / ".gitkeep"
            gitkeep.touch(exist_ok=True)


def generated_files_exist() -> bool:
    """Check whether a prepared dataset already exists."""

    for data_type in ("images", "masks"):
        for split in SPLIT_RATIOS:
            directory = OUTPUT_ROOT / data_type / split

            if not directory.exists():
                continue

            for path in directory.iterdir():
                if path.name != ".gitkeep":
                    return True

    return False


def clear_generated_files() -> None:
    """Remove generated dataset files but preserve .gitkeep."""

    for data_type in ("images", "masks"):
        for split in SPLIT_RATIOS:
            directory = OUTPUT_ROOT / data_type / split

            if not directory.exists():
                continue

            for path in directory.iterdir():
                if path.name == ".gitkeep":
                    continue

                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def copy_sample(
    sample: Sample,
    split: str,
) -> tuple[Path, Path]:
    """Copy one image-mask pair into the prepared dataset."""

    destination_stem = (
        f"{sample.environment}_{sample.stem}"
    )

    image_destination = (
        OUTPUT_ROOT
        / "images"
        / split
        / f"{destination_stem}{sample.image_path.suffix.lower()}"
    )

    mask_destination = (
        OUTPUT_ROOT
        / "masks"
        / split
        / f"{destination_stem}.png"
    )

    shutil.copy2(
        sample.image_path,
        image_destination,
    )

    shutil.copy2(
        sample.mask_path,
        mask_destination,
    )

    return image_destination, mask_destination


def write_dataset_yaml() -> Path:
    """Write the Ultralytics semantic dataset configuration."""

    yaml_path = OUTPUT_ROOT / "drivable.yaml"

    yaml_content = f"""# YOLO26 semantic segmentation dataset
path: {OUTPUT_ROOT.resolve().as_posix()}

train: images/train
val: images/val
test: images/test

masks_dir: masks

names:
  0: background
  1: drivable
  2: obstacle
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    return yaml_path


def prepare_dataset(
    seed: int,
    overwrite: bool,
) -> None:
    create_output_structure()

    if generated_files_exist():
        if not overwrite:
            raise RuntimeError(
                "Prepared dataset already contains files. "
                "Run again with --overwrite to rebuild it."
            )

        clear_generated_files()

    rng = random.Random(seed)

    summary = {
        split: {
            environment: 0
            for environment in SOURCES
        }
        for split in SPLIT_RATIOS
    }

    manifest_rows: list[dict[str, str]] = []

    for environment, source_root in SOURCES.items():
        samples = collect_samples(
            environment,
            source_root,
        )

        splits = split_samples(samples, rng)

        for split, split_samples_list in splits.items():
            for sample in split_samples_list:
                image_destination, mask_destination = copy_sample(
                    sample,
                    split,
                )

                summary[split][environment] += 1

                manifest_rows.append(
                    {
                        "environment": environment,
                        "split": split,
                        "source_stem": sample.stem,
                        "image": str(
                            image_destination.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "mask": str(
                            mask_destination.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    }
                )

    manifest_path = OUTPUT_ROOT / "split_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "environment",
                "split",
                "source_stem",
                "image",
                "mask",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_path = write_dataset_yaml()

    print("\nDataset preparation complete.\n")

    for split in ("train", "val", "test"):
        split_total = sum(summary[split].values())

        print(f"[{split}] total={split_total}")

        for environment in SOURCES:
            print(
                f"  {environment:<10}: "
                f"{summary[split][environment]}"
            )

    print(f"\nDataset root: {OUTPUT_ROOT}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Manifest:     {manifest_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the SDG output as an Ultralytics "
            "semantic segmentation dataset."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing prepared dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    prepare_dataset(
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
