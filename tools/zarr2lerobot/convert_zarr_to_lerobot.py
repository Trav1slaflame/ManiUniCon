"""Convert a ManiUniCon aggregated zarr to LeRobot Dataset v3.0.

Input:  a `{name}.{joint|cartesian}.zarr` directory produced by
        ``tools/process_demo_data.py``.
Output: a local LeRobot v3.0 dataset directory. Push-to-hub is a manual
        follow-up (``huggingface-cli upload``).

Follow-ups not handled in v1:
- Depth frames (``data/obs/depths/*``) -- no consensus in the v3 ecosystem.
- Camera intrinsics/extrinsics -- not part of the LeRobot v3 core schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np

ControlMode = Literal["joint", "cartesian"]


def detect_control_mode(zarr_path: Path | str) -> ControlMode:
    """Infer joint vs cartesian from the filename written by
    ``tools/process_demo_data.py``: the name ends with
    ``.joint.zarr`` or ``.cartesian.zarr``.
    """
    name = Path(zarr_path).name
    parts = name.split(".")
    if len(parts) >= 3 and parts[-1] == "zarr" and parts[-2] in ("joint", "cartesian"):
        return parts[-2]  # type: ignore[return-value]
    raise ValueError(
        f"cannot infer control mode from {name!r}; expected '*.joint.zarr' or '*.cartesian.zarr'"
    )


def discover_camera_keys(root) -> list[str]:
    """Return the sorted list of RGB camera array names under
    ``data/obs/images``. Raises ValueError if the group is missing or empty.
    """
    try:
        images = root["data/obs/images"]
    except KeyError:
        raise ValueError("no RGB camera group at 'data/obs/images'")
    keys = sorted(images.array_keys())
    if not keys:
        raise ValueError("no RGB camera arrays found under 'data/obs/images'")
    return keys


def build_features(
    control_mode: ControlMode,
    camera_keys: list[str],
    state_dim: int,
    action_dim: int,
    image_hw: tuple[int, int],
) -> dict:
    """Build the LeRobot v3 ``features`` dict for ``LeRobotDataset.create``.

    The LeRobot SDK auto-adds ``timestamp``, ``frame_index``, ``episode_index``,
    ``index`` and ``task_index`` -- do not add them here. The dataset-level
    fps is passed to ``LeRobotDataset.create`` directly and does not belong
    in the per-feature spec.
    """
    h, w = image_hw
    if control_mode == "cartesian":
        if state_dim != 8:
            raise ValueError(
                f"cartesian control mode requires state_dim=8 "
                f"(tcp_position[3] + tcp_orientation[4] + gripper[1]); got {state_dim}"
            )
    if control_mode == "joint":
        state_names = [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"]
    else:
        state_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

    feats: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": state_names if action_dim == state_dim else None,
        },
    }
    for cam in camera_keys:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
        }
    return feats


def resolve_task_text(cli_task: str, episode_desc: str) -> str:
    """Per-episode description wins if non-empty; else CLI fallback.

    Raises ValueError if neither produces a non-empty string.
    """
    if episode_desc and episode_desc.strip():
        return episode_desc.strip()
    if cli_task and cli_task.strip():
        return cli_task.strip()
    raise ValueError(
        "no task text available: neither per-episode description nor --task was provided"
    )


def build_episode_arrays(
    root,
    start: int,
    end: int,
    control_mode: ControlMode,
    camera_keys: list[str],
) -> dict[str, np.ndarray]:
    """Bulk-read one episode's worth of state/action/image arrays.

    Returns a dict keyed by LeRobot feature name with first dimension
    ``(end - start)``. Callers index into these arrays by local frame
    offset to assemble ``add_frame`` payloads.

    Per-episode slicing matters because ``tools/process_demo_data.py`` writes
    state and action arrays as single chunks spanning the full dataset --
    the zarr ``DirectoryStore`` has no chunk cache, so per-frame indexing
    would decompress the whole chunk once per frame (thousands of times).
    """
    obs = root["data/obs"]
    gripper = np.asarray(obs["gripper_state"][start:end], dtype=np.float32)
    if control_mode == "joint":
        joint_positions = np.asarray(
            obs["joint_positions"][start:end], dtype=np.float32
        )
        state = np.concatenate([joint_positions, gripper], axis=1)
    else:
        tcp_position = np.asarray(obs["tcp_position"][start:end], dtype=np.float32)
        tcp_orientation = np.asarray(
            obs["tcp_orientation"][start:end], dtype=np.float32
        )
        state = np.concatenate([tcp_position, tcp_orientation, gripper], axis=1)

    arrays: dict[str, np.ndarray] = {
        "observation.state": state,
        "action": np.asarray(root["data/action"][start:end], dtype=np.float32),
    }
    for cam in camera_keys:
        arrays[f"observation.images.{cam}"] = np.asarray(
            obs[f"images/{cam}"][start:end], dtype=np.uint8
        )
    return arrays


def iter_episode_bounds(root) -> Iterator[tuple[int, int, int]]:
    """Yield ``(episode_index, start, end)`` half-open ranges into the global
    time axis, derived from ``meta/episode_ends`` (cumulative convention used
    by :class:`ReplayBuffer`).
    """
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    prev = 0
    for i, end in enumerate(episode_ends):
        yield i, prev, int(end)
        prev = int(end)


def convert_zarr_to_lerobot(
    zarr_path: Path | str,
    output_dir: Path | str,
    repo_id: str,
    cli_task: str,
    fps: int,
    robot_type: str,
) -> None:
    """Convert a single aggregated zarr to a LeRobot v3.0 dataset on disk.

    Uses the official lerobot SDK's incremental writer. Always calls
    ``finalize()`` -- skipping it would leave parquet files corrupt per the
    LeRobot v3 docs "Common Issues" section.
    """
    import zarr
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)

    control_mode = detect_control_mode(zarr_path)
    root = zarr.open(str(zarr_path), mode="r")
    camera_keys = discover_camera_keys(root)

    gripper = root["data/obs/gripper_state"]
    action_arr = root["data/action"]
    if control_mode == "joint":
        state_dim = int(root["data/obs/joint_positions"].shape[1]) + int(
            gripper.shape[1]
        )
    else:
        state_dim = (
            int(root["data/obs/tcp_position"].shape[1])
            + int(root["data/obs/tcp_orientation"].shape[1])
            + int(gripper.shape[1])
        )
    action_dim = int(action_arr.shape[1])

    first_cam = root[f"data/obs/images/{camera_keys[0]}"]
    image_hw = (int(first_cam.shape[1]), int(first_cam.shape[2]))

    features = build_features(
        control_mode=control_mode,
        camera_keys=camera_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        image_hw=image_hw,
    )
    # lerobot's validate_feature_numpy_array compares ``value.shape !=
    # feature["shape"]`` directly, but numpy shape tuples never equal JSON
    # lists. Coerce shapes to tuples so the comparison works.
    for spec in features.values():
        spec["shape"] = tuple(spec["shape"])

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )

    episode_descriptions = root["meta/episode_descriptions"][:]
    try:
        for ep_idx, start, end in iter_episode_bounds(root):
            task_text = resolve_task_text(cli_task, str(episode_descriptions[ep_idx]))
            ep_arrays = build_episode_arrays(
                root, start, end, control_mode, camera_keys
            )
            for local_t in range(end - start):
                frame = {key: arr[local_t] for key, arr in ep_arrays.items()}
                frame["task"] = task_text
                dataset.add_frame(frame)
            dataset.save_episode()
    finally:
        dataset.finalize()


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a ManiUniCon aggregated zarr to LeRobot Dataset v3.0.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to '*.joint.zarr' or '*.cartesian.zarr' from tools/process_demo_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory to write the LeRobot dataset into (must not exist).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repo id, e.g. 'username/dataset-name'. Only used as metadata; push is manual.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Fallback natural-language task description. Per-episode descriptions in "
        "the zarr 'meta/episode_descriptions' take precedence when non-empty.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (matches configs/default.yaml policy_frequency=30).",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="unknown",
        help="Robot type string stored in info.json (e.g. 'ur5', 'xarm6', 'fr3').",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    convert_zarr_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        cli_task=args.task,
        fps=args.fps,
        robot_type=args.robot_type,
    )
    print(f"[zarr2lerobot] wrote LeRobot v3.0 dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
