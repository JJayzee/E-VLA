import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("switch_video_version.py")
_SPEC = importlib.util.spec_from_file_location("evla_switch_video_version", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_MODULE_PATH}")
profiles = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiles)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_shared_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
        "observation.images.DAVIS": {"dtype": "video", "shape": [260, 346, 3]},
        "observation.images.Enhanced": {"dtype": "video", "shape": [260, 346, 3]},
        "observation.events.DAVIS": {"dtype": "event", "shape": [None, 4]},
        "observation.timestamps.DAVIS": {"dtype": "int64", "shape": [2]},
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_videos": 4,
        "features": features,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "event_path": "events/chunk-{episode_chunk:03d}/{event_key}/episode_{episode_index:06d}.npz",
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    episodes = [{"episode_index": index, "length": 1} for index in range(2)]
    _write_jsonl(meta_dir / "episodes.jsonl", episodes)
    _write_jsonl(
        meta_dir / "episodes_stats.jsonl",
        [
            {
                "episode_index": index,
                "stats": {
                    "observation.images.DAVIS": {"mean": [index]},
                    "observation.images.Enhanced": {"mean": [index + 10]},
                    "observation.events.DAVIS": {"count": [index]},
                    "observation.timestamps.DAVIS": {"min": [index]},
                },
            }
            for index in range(2)
        ],
    )
    for video_key in ("observation.images.DAVIS", "observation.images.Enhanced"):
        video_dir = dataset_dir / "videos" / "chunk-000" / video_key
        video_dir.mkdir(parents=True)
        for index in range(2):
            (video_dir / f"episode_{index:06d}.mp4").touch()
    event_dir = dataset_dir / "events" / "chunk-000" / "observation.events.DAVIS"
    timestamp_dir = dataset_dir / "timestamps" / "chunk-000" / "observation.timestamps.DAVIS"
    event_dir.mkdir(parents=True)
    timestamp_dir.mkdir(parents=True)
    for index in range(2):
        (event_dir / f"episode_{index:06d}.npz").touch()
        (timestamp_dir / f"episode_{index:06d}.jsonl").touch()
    return dataset_dir


def test_event_and_image_profiles_share_physical_dataset(tmp_path, monkeypatch):
    dataset_dir = _make_shared_dataset(tmp_path)
    monkeypatch.setattr(profiles, "_validate_video_headers", lambda *_: None)

    profiles.prepare_video_profile(dataset_dir, "event_davis", "observation.images.DAVIS", True)
    profiles.prepare_video_profile(dataset_dir, "image_enhanced", "observation.images.Enhanced", False)

    profiles.switch_video_profile(dataset_dir, "image_enhanced")
    image_info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert set(image_info["features"]) == {
        "observation.state",
        "action",
        "observation.images.DAVIS",
    }
    assert "observation.images.Enhanced" in image_info["video_path"]
    assert "event_path" not in image_info
    image_stats = profiles._read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
    assert "observation.events.DAVIS" not in image_stats[0]["stats"]
    assert "observation.timestamps.DAVIS" not in image_stats[0]["stats"]

    profiles.switch_video_profile(dataset_dir, "event_davis")
    event_info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert "observation.events.DAVIS" in event_info["features"]
    assert "observation.images.DAVIS" in event_info["video_path"]
    assert "event_path" in event_info

    profiles.restore_active_backup(dataset_dir)
    restored = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert restored["features"] == {
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [6]},
        "observation.images.DAVIS": {"dtype": "video", "shape": [260, 346, 3]},
        "observation.images.Enhanced": {"dtype": "video", "shape": [260, 346, 3]},
        "observation.events.DAVIS": {"dtype": "event", "shape": [None, 4]},
        "observation.timestamps.DAVIS": {"dtype": "int64", "shape": [2]},
    }


def test_profile_names_reject_paths():
    for name in ("../outside", "nested/profile", ""):
        try:
            profiles._validate_profile_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe profile name accepted: {name!r}")
