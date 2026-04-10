import numpy as np
import pytest
import zarr

from tools.zarr2lerobot.convert_zarr_to_lerobot import (
    build_episode_arrays,
    build_features,
    detect_control_mode,
    discover_camera_keys,
    iter_episode_bounds,
    resolve_task_text,
)


def test_joint_fixture_layout(joint_zarr):
    root = zarr.open(str(joint_zarr), mode="r")
    assert root["data/action"].shape == (10, 7)
    assert root["data/obs/images/camera_0"].shape == (10, 64, 64, 3)
    assert root["meta/episode_ends"][:].tolist() == [5, 10]
    assert root["meta/episode_descriptions"][0] == "pick up the cube"


def test_cartesian_fixture_layout(cartesian_zarr):
    root = zarr.open(str(cartesian_zarr), mode="r")
    assert root["data/action"].shape == (10, 8)


def test_detect_control_mode_joint(joint_zarr):
    assert detect_control_mode(joint_zarr) == "joint"


def test_detect_control_mode_cartesian(cartesian_zarr):
    assert detect_control_mode(cartesian_zarr) == "cartesian"


def test_detect_control_mode_invalid(tmp_path):
    bad = tmp_path / "bad.zarr"
    bad.mkdir()
    with pytest.raises(ValueError, match="cannot infer control mode"):
        detect_control_mode(bad)


def test_discover_camera_keys_ordered(joint_zarr):
    root = zarr.open(str(joint_zarr), mode="r")
    assert discover_camera_keys(root) == ["camera_0", "camera_1"]


def test_discover_camera_keys_missing_group(tmp_path):
    path = tmp_path / "empty.joint.zarr"
    root = zarr.open(str(path), mode="w")
    root.create_group("data").create_group("obs")
    with pytest.raises(ValueError, match="no RGB camera"):
        discover_camera_keys(root)


def test_discover_camera_keys_empty_group(tmp_path):
    path = tmp_path / "empty2.joint.zarr"
    root = zarr.open(str(path), mode="w")
    root.create_group("data").create_group("obs").create_group("images")
    with pytest.raises(ValueError, match="no RGB camera"):
        discover_camera_keys(root)


def test_build_features_joint():
    feats = build_features(
        control_mode="joint",
        camera_keys=["camera_0"],
        state_dim=7,
        action_dim=7,
        image_hw=(96, 96),
    )
    assert feats["observation.state"]["shape"] == [7]
    assert feats["observation.state"]["dtype"] == "float32"
    assert feats["action"]["shape"] == [7]
    assert feats["action"]["dtype"] == "float32"
    assert feats["observation.images.camera_0"]["dtype"] == "video"
    assert feats["observation.images.camera_0"]["shape"] == [96, 96, 3]


def test_build_features_cartesian_two_cams():
    feats = build_features(
        control_mode="cartesian",
        camera_keys=["front", "wrist"],
        state_dim=8,
        action_dim=8,
        image_hw=(240, 320),
    )
    assert feats["observation.state"]["shape"] == [8]
    assert feats["action"]["shape"] == [8]
    assert "observation.images.front" in feats
    assert "observation.images.wrist" in feats
    assert feats["observation.images.front"]["shape"] == [240, 320, 3]


def test_build_features_cartesian_rejects_wrong_state_dim():
    with pytest.raises(ValueError, match="cartesian control mode requires"):
        build_features(
            control_mode="cartesian",
            camera_keys=["cam"],
            state_dim=6,
            action_dim=6,
            image_hw=(64, 64),
        )


@pytest.mark.parametrize(
    ("cli_task", "episode_desc", "expected"),
    [
        ("fallback", "pick cube", "pick cube"),
        ("fallback", "", "fallback"),
        ("fallback", "   ", "fallback"),
        ("fallback", "  hi  ", "hi"),
    ],
)
def test_resolve_task_text(cli_task, episode_desc, expected):
    assert resolve_task_text(cli_task, episode_desc) == expected


def test_resolve_task_raises_if_both_empty():
    with pytest.raises(ValueError, match="no task text"):
        resolve_task_text("", "")


def test_build_episode_arrays_joint(joint_zarr):
    root = zarr.open(str(joint_zarr), mode="r")
    arrays = build_episode_arrays(
        root, start=0, end=5, control_mode="joint", camera_keys=["camera_0", "camera_1"]
    )
    assert arrays["observation.state"].shape == (5, 7)
    assert arrays["observation.state"].dtype == np.float32
    assert arrays["action"].shape == (5, 7)
    assert arrays["action"].dtype == np.float32
    assert arrays["observation.images.camera_0"].shape == (5, 64, 64, 3)
    assert arrays["observation.images.camera_0"].dtype == np.uint8
    assert arrays["observation.images.camera_1"].shape == (5, 64, 64, 3)


def test_build_episode_arrays_cartesian(cartesian_zarr):
    root = zarr.open(str(cartesian_zarr), mode="r")
    arrays = build_episode_arrays(
        root, start=5, end=10, control_mode="cartesian", camera_keys=["camera_0"]
    )
    assert arrays["observation.state"].shape == (5, 8)
    assert arrays["observation.state"].dtype == np.float32
    assert arrays["action"].shape == (5, 8)
    assert "observation.images.camera_0" in arrays
    assert "observation.images.camera_1" not in arrays


def test_iter_episode_bounds_yields_half_open_ranges(joint_zarr):
    root = zarr.open(str(joint_zarr), mode="r")
    bounds = list(iter_episode_bounds(root))
    assert bounds == [(0, 0, 5), (1, 5, 10)]


def test_iter_episode_bounds_types(joint_zarr):
    root = zarr.open(str(joint_zarr), mode="r")
    for ep_idx, start, end in iter_episode_bounds(root):
        assert isinstance(ep_idx, int)
        assert isinstance(start, int)
        assert isinstance(end, int)


def test_convert_joint_end_to_end(joint_zarr, tmp_path):
    pytest.importorskip("lerobot")
    from tools.zarr2lerobot.convert_zarr_to_lerobot import convert_zarr_to_lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = tmp_path / "out_joint"
    convert_zarr_to_lerobot(
        zarr_path=joint_zarr,
        output_dir=output_dir,
        repo_id="test/fake-joint",
        cli_task="pick up the red cube",
        fps=10,
        robot_type="ur5",
    )

    ds = LeRobotDataset("test/fake-joint", root=output_dir)
    assert ds.meta.info["codebase_version"] == "v3.0"
    assert ds.meta.info["total_episodes"] == 2
    assert ds.meta.info["total_frames"] == 10
    assert ds.meta.info["fps"] == 10

    sample = ds[0]
    assert "observation.state" in sample
    assert "action" in sample
    assert "observation.images.camera_0" in sample
    assert "observation.images.camera_1" in sample

    # Different lerobot pins expose tasks differently (dict vs DataFrame vs
    # ndarray), so stringify the whole metadata blob and check substring
    # membership -- tolerant across API versions.
    info_str = str(ds.meta.info) + str(getattr(ds.meta, "tasks", ""))
    assert "pick up the cube" in info_str
    assert "pick up the red cube" in info_str


def test_convert_cartesian_end_to_end(cartesian_zarr, tmp_path):
    pytest.importorskip("lerobot")
    from tools.zarr2lerobot.convert_zarr_to_lerobot import convert_zarr_to_lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = tmp_path / "out_cart"
    convert_zarr_to_lerobot(
        zarr_path=cartesian_zarr,
        output_dir=output_dir,
        repo_id="test/fake-cartesian",
        cli_task="default task",
        fps=30,
        robot_type="xarm6",
    )
    ds = LeRobotDataset("test/fake-cartesian", root=output_dir)
    sample = ds[0]
    assert "observation.state" in sample
    assert "action" in sample
    assert ds.meta.info["total_episodes"] == 2
    assert ds.meta.info["fps"] == 30
