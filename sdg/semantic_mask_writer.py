import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_IDS = {
    "background": 0,
    "drivable": 1,
    "obstacle": 2,
}


class SemanticMaskWriter:
    """
    Synchronous exporter for RGB images and 3-class masks.

    Mask values:
        0 = background
        1 = drivable
        2 = obstacle
    """

    def __init__(
        self,
        output_dir: str,
        save_preview: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.save_preview = save_preview

        self.rgb_dir = self.output_dir / "rgb"
        self.mask_dir = self.output_dir / "masks"
        self.preview_dir = self.output_dir / "previews"

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        if self.save_preview:
            self.preview_dir.mkdir(parents=True, exist_ok=True)

        with (
            self.output_dir / "class_mapping.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "0": "background",
                    "1": "drivable",
                    "2": "obstacle",
                },
                file,
                indent=2,
            )

        print(f"[SDG] Output directory: {self.output_dir}")

    @staticmethod
    def _label_text(label_info) -> str:
        """Convert different idToLabels formats into searchable text."""

        if isinstance(label_info, str):
            return label_info.lower()

        try:
            return json.dumps(label_info).lower()
        except TypeError:
            return str(label_info).lower()

    @classmethod
    def _build_training_mask(
        cls,
        semantic_result: dict,
    ) -> np.ndarray:
        semantic_ids = np.asarray(
            semantic_result["data"]
        ).squeeze()

        if semantic_ids.ndim != 2:
            raise ValueError(
                "Expected semantic ID image with shape [H, W], "
                f"got {semantic_ids.shape}"
            )

        id_to_labels = (
            semantic_result
            .get("info", {})
            .get("idToLabels", {})
        )

        if not id_to_labels:
            raise RuntimeError(
                "Semantic annotator did not return idToLabels. "
                f"Available info keys: "
                f"{list(semantic_result.get('info', {}).keys())}"
            )

        # Unknown labels remain background.
        mask = np.zeros(
            semantic_ids.shape,
            dtype=np.uint8,
        )

        for semantic_id, label_info in id_to_labels.items():
            label_text = cls._label_text(label_info)
            semantic_id = int(semantic_id)

            if "drivable" in label_text:
                target_class = CLASS_IDS["drivable"]

            elif "obstacle" in label_text:
                target_class = CLASS_IDS["obstacle"]

            else:
                target_class = CLASS_IDS["background"]

            mask[semantic_ids == semantic_id] = target_class

        return mask

    @staticmethod
    def _create_preview(mask: np.ndarray) -> np.ndarray:
        palette = np.array(
            [
                [0, 0, 0],        # background
                [255, 255, 0],    # drivable
                [0, 255, 0],      # obstacle
            ],
            dtype=np.uint8,
        )

        return palette[mask]

    def save_frame(
        self,
        frame_id: int,
        rgb_data,
        semantic_result: dict,
    ) -> None:
        frame_name = f"{frame_id:06d}"

        # RGB
        rgb = np.asarray(rgb_data)

        if rgb.ndim != 3:
            raise ValueError(
                f"Unexpected RGB shape: {rgb.shape}"
            )

        # RGB annotator normally returns RGBA.
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        rgb = rgb.astype(np.uint8, copy=False)

        Image.fromarray(rgb).save(
            self.rgb_dir / f"{frame_name}.png"
        )

        # Training mask: 0, 1, 2
        mask = self._build_training_mask(
            semantic_result
        )

        Image.fromarray(mask).save(
            self.mask_dir / f"{frame_name}.png"
        )

        # Human-readable preview
        if self.save_preview:
            preview = self._create_preview(mask)

            Image.fromarray(preview).save(
                self.preview_dir / f"{frame_name}.png"
            )

        values, counts = np.unique(
            mask,
            return_counts=True,
        )

        class_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
        }

        print(
            f"[SDG] Saved frame {frame_name}, "
            f"class pixels={class_counts}"
        )