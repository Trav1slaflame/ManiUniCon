from pathlib import Path

import numpy as np
import pytest
import zarr


def _make_synthetic_zarr(tmp_path: Path, mode: str) -> Path:
    """Build a 2-episode, 5-frame-each zarr matching process_demo_data output.

    Structure:
        data/
            action                        float64 (10, A)
            obs/
                joint_positions           float64 (10, 6)
                tcp_position              float64 (10, 3)
                tcp_orientation           float64 (10, 4)
                gripper_state             float64 (10, 1)
                images/
                    camera_0              uint8   (10, 64, 64, 3)
                    camera_1              uint8   (10, 64, 64, 3)
        meta/
            episode_ends                  int64 (2,) = [5, 10]
            episode_descriptions          <U100 (2,)
    """
    path = tmp_path / f"fake.{mode}.zarr"
    root = zarr.open(str(path), mode="w")
    T = 10
    data = root.create_group("data")
    obs = data.create_group("obs")

    rng = np.random.default_rng(0)
    obs.create_dataset("joint_positions", data=rng.random((T, 6), dtype=np.float64))
    obs.create_dataset("tcp_position", data=rng.random((T, 3), dtype=np.float64))
    obs.create_dataset("tcp_orientation", data=rng.random((T, 4), dtype=np.float64))
    obs.create_dataset("gripper_state", data=rng.random((T, 1), dtype=np.float64))

    images = obs.create_group("images")
    # 64x64 is the minimum that AV1 encoders (SVT-AV1) can reliably handle;
    # smaller frames cause the encoder to hang.
    images.create_dataset(
        "camera_0", data=rng.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8)
    )
    images.create_dataset(
        "camera_1", data=rng.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8)
    )

    if mode == "joint":
        action = np.concatenate(
            [obs["joint_positions"][:], obs["gripper_state"][:]], axis=-1
        )
    else:
        action = np.concatenate(
            [
                obs["tcp_position"][:],
                obs["tcp_orientation"][:],
                obs["gripper_state"][:],
            ],
            axis=-1,
        )
    data.create_dataset("action", data=action.astype(np.float64))

    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([5, 10], dtype=np.int64))
    meta.create_dataset(
        "episode_descriptions",
        data=np.array(["pick up the cube", ""], dtype="<U100"),
    )

    return path


@pytest.fixture
def joint_zarr(tmp_path: Path) -> Path:
    return _make_synthetic_zarr(tmp_path, "joint")


@pytest.fixture
def cartesian_zarr(tmp_path: Path) -> Path:
    return _make_synthetic_zarr(tmp_path, "cartesian")
