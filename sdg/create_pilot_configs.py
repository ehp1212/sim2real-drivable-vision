#!/usr/bin/env python3

from copy import deepcopy
from pathlib import Path

import yaml


BASE_CONFIG = Path("sdg/configs/test.yaml")
CONFIG_DIR = Path("sdg/configs")


def save_config(name: str, config: dict) -> None:
    path = CONFIG_DIR / name

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    print(f"Created: {path}")


def set_capture_region(
    config: dict,
    camera_min,
    camera_max,
    look_at_min,
    look_at_max,
    obstacle_min,
    obstacle_max,
) -> None:
    config["camera"]["position"] = {
        "min": camera_min,
        "max": camera_max,
    }

    config["camera"]["look_at"] = {
        "min": look_at_min,
        "max": look_at_max,
    }

    config["randomization"]["obstacles"]["position"] = {
        "min": obstacle_min,
        "max": obstacle_max,
    }


def main() -> None:
    with BASE_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        base = yaml.safe_load(file)

    # -----------------------------------------------
    # Warehouse
    # 기존에 검증한 semantic path와 영역을 그대로 유지
    # -----------------------------------------------

    warehouse = deepcopy(base)

    warehouse["launch"]["headless"] = True
    warehouse["generation"]["num_frames"] = 100
    warehouse["generation"]["seed"] = 42
    warehouse["generation"]["output_dir"] = (
        "sdg/outputs/warehouse/run_001"
    )

    save_config(
        "warehouse_100.yaml",
        warehouse,
    )

    # -----------------------------------------------
    # Simple Room
    # -----------------------------------------------

    simple_room = deepcopy(base)

    simple_room["launch"]["headless"] = True

    simple_room["generation"]["num_frames"] = 100
    simple_room["generation"]["seed"] = 142
    simple_room["generation"]["output_dir"] = (
        "sdg/outputs/simple_room/run_001"
    )

    simple_room["environment"] = {
        "url": (
            "/Isaac/Environments/"
            "Simple_Room/simple_room.usd"
        ),
        "semantic_labels": {
            "drivable": {
                "paths": [],
                "keywords": [
                    "floor",
                    "ground",
                    "carpet",
                ],
            },
            "obstacle": {
                "paths": [],
                "keywords": [
                    "table",
                    "chair",
                    "desk",
                    "cabinet",
                    "shelf",
                ],
            },
        },
    }

    set_capture_region(
        simple_room,
        camera_min=[-1.5, -1.5, 0.18],
        camera_max=[1.5, 1.5, 0.35],
        look_at_min=[-1.0, -1.0, 0.0],
        look_at_max=[1.0, 1.0, 0.25],
        obstacle_min=[-1.2, -1.2, 0.0],
        obstacle_max=[1.2, 1.2, 0.0],
    )

    save_config(
        "simple_room_100.yaml",
        simple_room,
    )

    # -----------------------------------------------
    # Office
    # -----------------------------------------------

    office = deepcopy(base)

    office["launch"]["headless"] = True

    office["generation"]["num_frames"] = 100
    office["generation"]["seed"] = 242
    office["generation"]["output_dir"] = (
        "sdg/outputs/office/run_001"
    )

    office["environment"] = {
        "url": (
            "/Isaac/Environments/"
            "Office/office.usd"
        ),
        "semantic_labels": {
            "drivable": {
                "paths": [],
                "keywords": [
                    "floor",
                    "ground",
                    "carpet",
                ],
            },
            "obstacle": {
                "paths": [],
                "keywords": [
                    "table",
                    "desk",
                    "chair",
                    "cabinet",
                    "shelf",
                    "sofa",
                    "couch",
                    "plant",
                    "monitor",
                    "computer",
                    "printer",
                    "trash",
                    "bin",
                    "column",
                    "pillar",
                    "partition",
                ],
            },
        },
    }

    # Office는 크므로 우선 원점 근처 구역만 사용
    set_capture_region(
        office,
        camera_min=[-4.0, -4.0, 0.18],
        camera_max=[4.0, 4.0, 0.35],
        look_at_min=[-3.0, -3.0, 0.0],
        look_at_max=[3.0, 3.0, 0.25],
        obstacle_min=[-3.0, -3.0, 0.0],
        obstacle_max=[3.0, 3.0, 0.0],
    )

    save_config(
        "office_100.yaml",
        office,
    )


if __name__ == "__main__":
    main()