from types import SimpleNamespace

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.robots.manipulator import ManipulatorRobot


def _features():
    return {
        "observation.state": {"dtype": "float32", "shape": [6], "names": None},
        "observation.images.DAVIS": {
            "dtype": "video",
            "shape": [260, 346, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.events.DAVIS": {
            "dtype": "event",
            "shape": [None, 4],
            "names": ["timestamp", "x", "y", "polarity"],
        },
        "observation.timestamps.DAVIS": {
            "dtype": "int64",
            "shape": [2],
            "names": ["start", "end"],
        },
        "action": {"dtype": "float32", "shape": [6], "names": None},
    }


def test_event_metadata_is_not_exposed_to_image_policy():
    policy_features = dataset_to_policy_features(_features())

    assert "observation.events.DAVIS" not in policy_features
    assert "observation.timestamps.DAVIS" not in policy_features
    assert policy_features["observation.images.DAVIS"].type is FeatureType.VISUAL
    assert policy_features["observation.images.DAVIS"].shape == (3, 260, 346)


def test_event_metadata_is_not_queried_from_parquet():
    cfg = SimpleNamespace(
        observation_delta_indices=[0],
        action_delta_indices=[0, 1],
        reward_delta_indices=None,
    )
    metadata = SimpleNamespace(features=_features(), fps=30)

    delta_timestamps = resolve_delta_timestamps(cfg, metadata)

    assert "observation.events.DAVIS" not in delta_timestamps
    assert "observation.timestamps.DAVIS" not in delta_timestamps
    assert delta_timestamps["observation.images.DAVIS"] == [0.0]
    assert delta_timestamps["action"] == [0.0, 1 / 30]


def test_robot_camera_features_are_image_only():
    robot = object.__new__(ManipulatorRobot)
    robot.cameras = {
        "DAVIS": SimpleNamespace(height=260, width=346, channels=3, work_mode="davis")
    }

    features = robot.camera_features

    assert set(features) == {"observation.images.DAVIS"}
