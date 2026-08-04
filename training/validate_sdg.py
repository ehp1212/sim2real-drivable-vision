from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_SOURCES = {
    "office": PROJECT_ROOT / "sdg/outputs/office/run_001",
    "hospital": PROJECT_ROOT / "sdg/outputs/hospital/run_001",
    "warehouse": PROJECT_ROOT / "sdg/outputs/warehouse/run_001",
    "real_room": PROJECT_ROOT / "sdg/outputs/real_room/run_001",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_CLASSES = {0, 1, 2}


def collect_images(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def validate_environment(name: str, root: Path) -> tuple[list[str], dict[int, int]]:
    errors: list[str] = []
    class_pixels = {0: 0, 1: 0, 2: 0}

    rgb_dir = root / "rgb"
    mask_dir = root / "masks"

    if not rgb_dir.is_dir():
        return [f"{name}: missing directory {rgb_dir}"], class_pixels

    if not mask_dir.is_dir():
        return [f"{name}: missing directory {mask_dir}"], class_pixels

    rgb_files = collect_images(rgb_dir)
    mask_files = collect_images(mask_dir)

    missing_masks = sorted(set(rgb_files) - set(mask_files))
    missing_rgb = sorted(set(mask_files) - set(rgb_files))

    if missing_masks:
        errors.append(
            f"{name}: {len(missing_masks)} RGB files have no mask: "
            f"{missing_masks[:5]}"
        )

    if missing_rgb:
        errors.append(
            f"{name}: {len(missing_rgb)} masks have no RGB image: "
            f"{missing_rgb[:5]}"
        )

    matched_stems = sorted(set(rgb_files) & set(mask_files))

    invalid_frames = 0

    for stem in matched_stems:
        rgb_path = rgb_files[stem]
        mask_path = mask_files[stem]

        try:
            with Image.open(rgb_path) as rgb_image:
                rgb_image.load()
                rgb_size = rgb_image.size

            with Image.open(mask_path) as mask_image:
                mask_image.load()
                mask_size = mask_image.size
                mask = np.asarray(mask_image)

        except (UnidentifiedImageError, OSError) as error:
            errors.append(f"{name}/{stem}: unreadable image: {error}")
            invalid_frames += 1
            continue

        if rgb_size != mask_size:
            errors.append(
                f"{name}/{stem}: size mismatch "
                f"RGB={rgb_size}, mask={mask_size}"
            )
            invalid_frames += 1
            continue

        if mask.ndim != 2:
            errors.append(
                f"{name}/{stem}: mask is not single-channel, "
                f"shape={mask.shape}"
            )
            invalid_frames += 1
            continue

        values, counts = np.unique(mask, return_counts=True)
        found_classes = set(int(value) for value in values)

        unexpected = found_classes - ALLOWED_CLASSES

        if unexpected:
            errors.append(
                f"{name}/{stem}: unexpected mask values "
                f"{sorted(unexpected)}"
            )
            invalid_frames += 1
            continue

        for value, count in zip(values, counts):
            class_pixels[int(value)] += int(count)

    print(f"\n[{name}]")
    print(f"  RGB files:       {len(rgb_files)}")
    print(f"  Mask files:      {len(mask_files)}")
    print(f"  Matched pairs:   {len(matched_stems)}")
    print(f"  Invalid frames:  {invalid_frames}")

    total_pixels = sum(class_pixels.values())

    if total_pixels > 0:
        for class_id, class_name in (
            (0, "background"),
            (1, "drivable"),
            (2, "obstacle"),
        ):
            ratio = class_pixels[class_id] / total_pixels * 100.0
            print(
                f"  Class {class_id} {class_name:<10}: "
                f"{ratio:6.2f}%"
            )

    return errors, class_pixels


def main() -> None:
    all_errors: list[str] = []
    global_pixels = {0: 0, 1: 0, 2: 0}

    for environment, root in DATA_SOURCES.items():
        errors, class_pixels = validate_environment(environment, root)
        all_errors.extend(errors)

        for class_id, pixel_count in class_pixels.items():
            global_pixels[class_id] += pixel_count

    print("\n[Total]")
    total_pixels = sum(global_pixels.values())

    if total_pixels > 0:
        for class_id, class_name in (
            (0, "background"),
            (1, "drivable"),
            (2, "obstacle"),
        ):
            ratio = global_pixels[class_id] / total_pixels * 100.0
            print(
                f"  Class {class_id} {class_name:<10}: "
                f"{ratio:6.2f}%"
            )

    if all_errors:
        print(f"\nValidation failed with {len(all_errors)} error(s):")

        for error in all_errors[:30]:
            print(f"  - {error}")

        raise SystemExit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()