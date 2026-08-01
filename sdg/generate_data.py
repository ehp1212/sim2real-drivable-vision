#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate semantic-segmentation data with Isaac Sim 4.5"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an SDG YAML configuration file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------
# 1. 일반 Python 코드로 인자와 설정을 먼저 읽는다.
# ---------------------------------------------------------

args = parse_args()
config = load_config(args.config)

launch_config = {
    "headless": config["launch"]["headless"],
    "renderer": config["launch"]["renderer"],
}


# ---------------------------------------------------------
# 2. Isaac Sim을 가장 먼저 실행한다.
# ---------------------------------------------------------

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config=launch_config)


# ---------------------------------------------------------
# 3. SimulationApp 이후에 Isaac/Omniverse 모듈을 import한다.
# ---------------------------------------------------------

import json
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

import omni.replicator.core as rep
from isaacsim.core.utils.stage import open_stage, get_current_stage
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.storage.native import get_assets_root_path
from semantic_mask_writer import SemanticMaskWriter

def load_environment(environment_url: str) -> None:
    """Load one Isaac Sim environment USD as the current Stage."""

    assets_root = get_assets_root_path()

    if assets_root is None:
        raise RuntimeError("Isaac Sim asset root could not be found")

    full_url = assets_root + environment_url

    print(f"[SDG] Loading environment: {full_url}")

    if not open_stage(full_url):
        raise RuntimeError(f"Failed to load environment: {full_url}")

    # Stage 로딩이 끝나도록 몇 프레임 업데이트
    for _ in range(10):
        simulation_app.update()

def print_mesh_prim_paths() -> None:
    """Print mesh paths to identify floors and fixed obstacles."""

    stage = get_current_stage()

    print("\n[SDG] Mesh prims in current environment:")

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            print(prim.GetPath())

def apply_semantic_labels(config: dict) -> None:
    """
    Apply project semantic classes to existing environment prims.

    Final mask:
        0 = background / unlabeled
        1 = drivable
        2 = obstacle
    """

    stage = get_current_stage()

    drivable_paths = (
        config
        .get("environment", {})
        .get("semantic_labels", {})
        .get("drivable", [])
    )

    if not drivable_paths:
        raise RuntimeError(
            "No drivable Prim paths configured in YAML"
        )

    total_labeled = 0

    for root_path in drivable_paths:
        root_prim = stage.GetPrimAtPath(root_path)

        if not root_prim.IsValid():
            raise RuntimeError(
                f"Drivable Prim does not exist: {root_path}"
            )

        labeled_paths = []

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            add_update_semantics(
                prim=prim,
                semantic_label="drivable",
                type_label="class",
            )

            labeled_paths.append(
                str(prim.GetPath())
            )

        if not labeled_paths:
            raise RuntimeError(
                f"No Mesh Prim found under: {root_path}"
            )

        print(
            f"[SDG] Drivable meshes under {root_path}:"
        )

        for path in labeled_paths:
            print(f"  - {path}")

        total_labeled += len(labeled_paths)

    # Allow Stage/renderer semantic changes to update.
    for _ in range(3):
        simulation_app.update()

    print(
        f"[SDG] Total drivable meshes labeled: "
        f"{total_labeled}"
    )


def create_camera(camera_config: dict):
    """Create one RC-car-like camera."""

    camera = rep.create.camera(
        focal_length=camera_config["focal_length"],
        clipping_range=tuple(camera_config["clipping_range"]),
        name="RcCarCamera",
    )

    resolution = tuple(camera_config["resolution"])

    render_product = rep.create.render_product(
        camera,
        resolution,
        name="RcCarRenderProduct",
    )

    return camera, render_product

def register_randomizers(
    camera,
    obstacle_group,
    config: dict,
) -> None:
    """Register per-frame camera, light and obstacle randomization."""

    camera_config = config["camera"]
    randomization_config = config["randomization"]

    # -----------------------------------------------------
    # Camera bounds
    # -----------------------------------------------------

    camera_position_min = tuple(
        camera_config["position"]["min"]
    )
    camera_position_max = tuple(
        camera_config["position"]["max"]
    )

    camera_look_at_min = tuple(
        camera_config["look_at"]["min"]
    )
    camera_look_at_max = tuple(
        camera_config["look_at"]["max"]
    )

    # -----------------------------------------------------
    # Obstacle bounds
    # -----------------------------------------------------

    obstacle_config = randomization_config["obstacles"]

    obstacle_position_min = tuple(
        obstacle_config["position"]["min"]
    )
    obstacle_position_max = tuple(
        obstacle_config["position"]["max"]
    )

    obstacle_rotation_min = tuple(
        obstacle_config["rotation"]["min"]
    )
    obstacle_rotation_max = tuple(
        obstacle_config["rotation"]["max"]
    )

    obstacle_scale_min = tuple(
        obstacle_config["scale"]["min"]
    )
    obstacle_scale_max = tuple(
        obstacle_config["scale"]["max"]
    )

    obstacle_color_min = tuple(
        obstacle_config["color"]["min"]
    )
    obstacle_color_max = tuple(
        obstacle_config["color"]["max"]
    )

    # -----------------------------------------------------
    # Register per-frame randomization graph
    # -----------------------------------------------------

    with rep.trigger.on_frame():

        # Camera pose
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    camera_position_min,
                    camera_position_max,
                ),
                look_at=rep.distribution.uniform(
                    camera_look_at_min,
                    camera_look_at_max,
                ),
            )

        # Spawned obstacle poses and appearance
        if obstacle_group is not None:
            with obstacle_group:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        obstacle_position_min,
                        obstacle_position_max,
                    ),
                    rotation=rep.distribution.uniform(
                        obstacle_rotation_min,
                        obstacle_rotation_max,
                    ),
                    scale=rep.distribution.uniform(
                        obstacle_scale_min,
                        obstacle_scale_max,
                    ),
                )

                rep.randomizer.color(
                    colors=rep.distribution.uniform(
                        obstacle_color_min,
                        obstacle_color_max,
                    )
                )

        # Lighting
        lighting_config = randomization_config["lighting"]

        if lighting_config.get("enabled", True):
            rep.create.light(
                light_type="Sphere",
                count=lighting_config["count"],
                position=rep.distribution.uniform(
                    tuple(lighting_config["position"]["min"]),
                    tuple(lighting_config["position"]["max"]),
                ),
                intensity=rep.distribution.uniform(
                    lighting_config["intensity"]["min"],
                    lighting_config["intensity"]["max"],
                ),
                color=rep.distribution.uniform(
                    tuple(lighting_config["color"]["min"]),
                    tuple(lighting_config["color"]["max"]),
                ),
                temperature=rep.distribution.normal(
                    lighting_config["temperature"]["mean"],
                    lighting_config["temperature"]["std"],
                ),
                scale=rep.distribution.uniform(
                    lighting_config["scale"]["min"],
                    lighting_config["scale"]["max"],
                ),
            )

def generate_frames(
    config: dict,
    rgb_annotator,
    semantic_annotator,
    mask_writer: SemanticMaskWriter,
) -> None:
    """Capture annotator data and save it synchronously."""

    num_frames = config["generation"]["num_frames"]
    rt_subframes = config["generation"]["rt_subframes"]

    for frame_index in range(num_frames):
        print(
            f"[SDG] Capturing frame "
            f"{frame_index + 1}/{num_frames}"
        )

        # Randomize scene and render one captured frame.
        rep.orchestrator.step(
            delta_time=0.0,
            rt_subframes=rt_subframes,
        )

        # Read data immediately after capture.
        rgb_data = rgb_annotator.get_data()
        semantic_result = semantic_annotator.get_data()

        print(
            "[SDG] Annotator shapes:",
            np.asarray(rgb_data).shape,
            np.asarray(semantic_result["data"]).shape,
        )

        mask_writer.save_frame(
            frame_id=frame_index,
            rgb_data=rgb_data,
            semantic_result=semantic_result,
        )


def create_annotators(render_product):
    """Attach RGB and semantic annotators to one render product."""

    rgb_annotator = rep.AnnotatorRegistry.get_annotator(
        "rgb"
    )

    semantic_annotator = (
        rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={
                "colorize": False,
            },
        )
    )

    rgb_annotator.attach(render_product)
    semantic_annotator.attach(render_product)

    return rgb_annotator, semantic_annotator

def create_obstacles(config: dict):
    """
    Create obstacle assets defined in the YAML file.

    All generated assets receive:
        class = obstacle

    Returns:
        ReplicatorItem containing every created obstacle,
        or None when obstacle generation is disabled.
    """

    obstacle_config = (
        config
        .get("randomization", {})
        .get("obstacles", {})
    )

    if not obstacle_config.get("enabled", True):
        print("[SDG] Obstacle generation disabled")
        return None

    asset_configs = obstacle_config.get("assets", [])

    if not asset_configs:
        print("[SDG] No obstacle assets configured")
        return None

    assets_root = get_assets_root_path()

    if assets_root is None:
        raise RuntimeError(
            "Could not find the Isaac Sim asset root"
        )

    created_obstacles = []

    for asset_config in asset_configs:
        name = asset_config["name"]
        relative_url = asset_config["url"]
        count = int(asset_config.get("count", 1))

        if count <= 0:
            print(
                f"[SDG] Skipping {name}: "
                f"count must be greater than zero"
            )
            continue

        asset_url = assets_root + relative_url

        print(
            f"[SDG] Creating obstacle: "
            f"name={name}, count={count}, url={asset_url}"
        )

        obstacle = rep.create.from_usd(
            usd=asset_url,
            semantics=[("class", "obstacle")],
            count=count,
        )

        created_obstacles.append(obstacle)

    if not created_obstacles:
        print("[SDG] No obstacles were created")
        return None

    obstacle_group = rep.create.group(
        created_obstacles,
        name="ObstacleGroup",
    )

    print(
        f"[SDG] Created {len(created_obstacles)} "
        "obstacle asset groups"
    )

    return obstacle_group

def main() -> None:
    seed = config["generation"]["seed"]

    random.seed(seed)
    rep.set_global_seed(seed)
    rep.orchestrator.set_capture_on_play(False)

    # 1. Environment
    load_environment(
        config["environment"]["url"]
    )

    print_mesh_prim_paths()

    # 2. Existing environment semantics
    apply_semantic_labels(config)

    # 3. Additional obstacle assets
    obstacle_group = create_obstacles(config)

    # 4. Camera and render product
    camera, render_product = create_camera(
        config["camera"]
    )

    # 5. Scene randomization
    register_randomizers(
        camera,
        obstacle_group,
        config,
    )

    # 6. Direct annotators
    rgb_annotator, semantic_annotator = (
        create_annotators(render_product)
    )

    # 7. Normal synchronous file exporter
    mask_writer = SemanticMaskWriter(
        output_dir=config["generation"]["output_dir"],
        save_preview=config["writer"].get(
            "save_preview",
            True,
        ),
    )

    try:
        generate_frames(
            config=config,
            rgb_annotator=rgb_annotator,
            semantic_annotator=semantic_annotator,
            mask_writer=mask_writer,
        )

    finally:
        rgb_annotator.detach()
        semantic_annotator.detach()
        render_product.destroy()

    print("[SDG] Dataset generation completed")

import traceback


try:
    main()

except Exception:
    print("[SDG] Fatal error:")
    traceback.print_exc()
    raise

finally:
    simulation_app.close()