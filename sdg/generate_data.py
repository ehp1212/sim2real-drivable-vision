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

from pathlib import Path

def load_environment(environment_url: str) -> None:
    """Load one Isaac Sim environment USD as the current Stage."""

    assets_root = get_assets_root_path()

    if assets_root is None:
        raise RuntimeError("Isaac Sim asset root could not be found")

    full_url = resolve_environment_url(
        environment_url,
        assets_root,
    )

    print(f"[SDG] Loading environment: {full_url}")

    # open_stage()는 성공하면 True를 반환
    success = open_stage(full_url)

    if not success:
        raise RuntimeError(
            f"Failed to load environment: {full_url}"
        )

    # Stage 로딩 완료를 위해 몇 프레임 진행
    for _ in range(10):
        simulation_app.update()

    stage = get_current_stage()

    if stage is None:
        raise RuntimeError(
            f"Stage was not available after loading: {full_url}"
        )

    print(f"[SDG] Environment loaded: {full_url}")

def resolve_environment_url(
    environment_url: str,
    assets_root: str,
) -> str:
    """Resolve remote Isaac paths and local filesystem paths."""

    # 이미 완성된 URL
    if environment_url.startswith(
        ("http://", "https://", "omniverse://", "file://")
    ):
        return environment_url

    # Isaac Sim 기본 asset 경로
    # 예: /Isaac/Environments/Office/office.usd
    if environment_url.startswith("/Isaac/"):
        return (
            assets_root.rstrip("/")
            + environment_url
        )

    local_path = Path(environment_url).expanduser()

    # 로컬 절대경로
    # 예: /home/eun/.../room.usd
    if local_path.is_absolute():
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Local environment USD does not exist: {local_path}"
            )

        return str(local_path.resolve())

    # 로컬 상대경로가 실제로 존재하는 경우
    if local_path.is_file():
        return str(local_path.resolve())

    raise FileNotFoundError(
        f"Environment USD could not be resolved: {environment_url}"
    )

def print_floor_mesh_candidates() -> None:
    stage = get_current_stage()

    print("\n[SDG] Floor Mesh candidates:")

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path = str(prim.GetPath())
        lower_path = path.lower()

        if any(
            keyword in lower_path
            for keyword in (
                "floor",
                "ground",
                "concrete",
                "carpet",
            )
        ):
            print(f"  - {path}")

def apply_semantic_labels(config: dict) -> None:
    """
    Apply semantic labels to meshes in the loaded environment.

    YAML format:

    semantic_labels:
      drivable:
        paths: [...]
        keywords: [...]

      obstacle:
        paths: [...]
        keywords: [...]

    Final mask:
        0 = background / unlabeled
        1 = drivable
        2 = obstacle
    """

    stage = get_current_stage()

    semantic_config = (
        config
        .get("environment", {})
        .get("semantic_labels", {})
    )

    if not semantic_config:
        raise RuntimeError(
            "environment.semantic_labels is missing"
        )

    def read_rule(rule_name: str) -> tuple[list[str], list[str]]:
        rule = semantic_config.get(rule_name, {})

        if not isinstance(rule, dict):
            raise TypeError(
                f"semantic_labels.{rule_name} "
                "must contain paths and keywords"
            )

        paths = rule.get("paths", [])
        keywords = rule.get("keywords", [])

        if not isinstance(paths, list):
            raise TypeError(
                f"semantic_labels.{rule_name}.paths "
                "must be a list"
            )

        if not isinstance(keywords, list):
            raise TypeError(
                f"semantic_labels.{rule_name}.keywords "
                "must be a list"
            )

        normalized_keywords = [
            str(keyword).strip().lower()
            for keyword in keywords
            if str(keyword).strip()
        ]

        return paths, normalized_keywords

    def collect_meshes(
        paths: list[str],
        keywords: list[str],
    ) -> dict[str, object]:
        """
        Return unique Mesh prims selected by exact paths
        or by keywords contained in their name/path.
        """

        selected_meshes = {}

        # -------------------------------------------------
        # Exact Prim paths
        # -------------------------------------------------

        for prim_path in paths:
            root_prim = stage.GetPrimAtPath(prim_path)

            if not root_prim.IsValid():
                print(
                    f"[SDG][Warning] Prim not found: "
                    f"{prim_path}"
                )
                continue

            # Exact path points directly to a Mesh.
            if root_prim.IsA(UsdGeom.Mesh):
                selected_meshes[
                    str(root_prim.GetPath())
                ] = root_prim

                continue

            # Exact path points to an Xform/group.
            # Label Mesh descendants only.
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdGeom.Mesh):
                    continue

                selected_meshes[
                    str(prim.GetPath())
                ] = prim

        # -------------------------------------------------
        # Keyword matching
        # -------------------------------------------------

        if keywords:
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Mesh):
                    continue

                searchable_text = (
                    f"{prim.GetName()} {prim.GetPath()}"
                    .lower()
                )

                if any(
                    keyword in searchable_text
                    for keyword in keywords
                ):
                    selected_meshes[
                        str(prim.GetPath())
                    ] = prim

        return selected_meshes

    def apply_rule(
        rule_name: str,
        semantic_label: str,
    ) -> set[str]:
        paths, keywords = read_rule(rule_name)

        meshes = collect_meshes(
            paths=paths,
            keywords=keywords,
        )

        for prim in meshes.values():
            add_update_semantics(
                prim=prim,
                semantic_label=semantic_label,
                type_label="class",
            )

        print(
            f"[SDG] Applied '{semantic_label}' "
            f"to {len(meshes)} meshes"
        )

        for path in list(meshes.keys())[:20]:
            print(f"  - {path}")

        if len(meshes) > 20:
            print(
                f"  ... and {len(meshes) - 20} more"
            )

        return set(meshes.keys())

    # Drivable first.
    drivable_meshes = apply_rule(
        rule_name="drivable",
        semantic_label="drivable",
    )

    # Obstacle second so it wins if one Mesh matches
    # both drivable and obstacle keywords.
    obstacle_meshes = apply_rule(
        rule_name="obstacle",
        semantic_label="obstacle",
    )

    overlapping_meshes = (
        drivable_meshes & obstacle_meshes
    )

    if overlapping_meshes:
        print(
            "[SDG][Warning] Meshes matched both rules. "
            "They were finalized as obstacle:"
        )

        for path in sorted(overlapping_meshes):
            print(f"  - {path}")

    if not drivable_meshes:
        raise RuntimeError(
            "No drivable meshes matched the YAML rules"
        )

    for _ in range(3):
        simulation_app.update()

    print(
        "[SDG] Environment semantic labeling complete: "
        f"drivable={len(drivable_meshes)}, "
        f"obstacle={len(obstacle_meshes)}"
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

    print_floor_mesh_candidates()

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