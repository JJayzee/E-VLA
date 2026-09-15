import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_KEY = "observation.events.DAVIS"
TIMESTAMP_KEY = "observation.timestamps.DAVIS"


def _read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, items):
    with Path(path).open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _load_meta(dataset_dir):
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "meta" / "info.json").open() as f:
        info = json.load(f)
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    stats = _read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")

    expected = list(range(len(episodes)))
    if [item["episode_index"] for item in episodes] != expected:
        raise ValueError(f"{dataset_dir}: episodes.jsonl episode_index must be continuous from 0.")
    if [item["episode_index"] for item in stats] != expected:
        raise ValueError(f"{dataset_dir}: episodes_stats.jsonl does not match episodes.jsonl.")
    if info["total_episodes"] != len(episodes):
        raise ValueError(
            f"{dataset_dir}: info.total_episodes={info['total_episodes']}, actual={len(episodes)}."
        )
    if info["total_frames"] != sum(item["length"] for item in episodes):
        raise ValueError(f"{dataset_dir}: info.total_frames does not equal episode lengths.")
    if info.get("total_tasks") != len(tasks):
        raise ValueError(f"{dataset_dir}: info.total_tasks does not match tasks.jsonl.")
    return info, episodes, stats, tasks


def _feature_keys(info, dtype):
    return [key for key, feature in info["features"].items() if feature["dtype"] == dtype]


def _prepare_output(dst_dir):
    dst = Path(dst_dir)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Output dataset directory is not empty: {dst}")
    for name in ["data", "events", "timestamps", "videos"]:
        (dst / name / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    return dst


def _episode_paths(dataset_dir, episode_index, video_keys, event_keys):
    dataset_dir = Path(dataset_dir)
    stem = f"episode_{episode_index:06d}"
    paths = {
        "parquet": dataset_dir / "data" / "chunk-000" / f"{stem}.parquet",
        "videos": {
            key: dataset_dir / "videos" / "chunk-000" / key / f"{stem}.mp4" for key in video_keys
        },
        "events": {
            key: dataset_dir / "events" / "chunk-000" / key / f"{stem}.npz" for key in event_keys
        },
    }
    if EVENT_KEY in event_keys:
        paths["timestamp"] = (
            dataset_dir / "timestamps" / "chunk-000" / TIMESTAMP_KEY / f"{stem}.jsonl"
        )
    return paths


def _validate_episode(dataset_dir, episode, video_keys, event_keys, features):
    episode_index = episode["episode_index"]
    episode_length = episode["length"]
    paths = _episode_paths(dataset_dir, episode_index, video_keys, event_keys)
    required = [paths["parquet"], *paths["videos"].values(), *paths["events"].values()]
    if "timestamp" in paths:
        required.append(paths["timestamp"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Episode {episode_index} is missing files: {missing}")

    df = pd.read_parquet(paths["parquet"])
    required_columns = {
        "action", "observation.state", "timestamp", "frame_index",
        "episode_index", "index", "task_index",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{paths['parquet']}: missing columns {sorted(missing_columns)}.")
    expected_dtypes = {
        "timestamp": "float32", "frame_index": "int64", "episode_index": "int64",
        "index": "int64", "task_index": "int64",
    }
    wrong_dtypes = {
        key: str(df[key].dtype) for key, dtype in expected_dtypes.items() if str(df[key].dtype) != dtype
    }
    if wrong_dtypes:
        raise ValueError(f"{paths['parquet']}: invalid index/timestamp dtypes {wrong_dtypes}.")
    if len(df) != episode_length:
        raise ValueError(
            f"{paths['parquet']}: rows={len(df)}, episodes.jsonl length={episode_length}."
        )
    if df["frame_index"].tolist() != list(range(episode_length)):
        raise ValueError(f"{paths['parquet']}: frame_index is not continuous from 0.")
    if set(df["episode_index"].tolist()) != {episode_index}:
        raise ValueError(f"{paths['parquet']}: episode_index does not equal {episode_index}.")
    if len(df) and df["index"].tolist() != list(range(int(df["index"].iloc[0]), int(df["index"].iloc[0]) + len(df))):
        raise ValueError(f"{paths['parquet']}: index is not continuous.")
    if not set(df["task_index"].tolist()).issubset(set(range(len(_read_jsonl(Path(dataset_dir) / 'meta' / 'tasks.jsonl'))))):
        raise ValueError(f"{paths['parquet']}: task_index is not defined in tasks.jsonl.")
    for key in ["action", "observation.state"]:
        values = np.stack(df[key].to_numpy())
        feature = features[key]
        if values.dtype != np.dtype(feature["dtype"]) or list(values.shape[1:]) != feature["shape"]:
            raise ValueError(
                f"{paths['parquet']}:{key}: expected {feature['dtype']} {feature['shape']}, "
                f"got {values.dtype} {list(values.shape[1:])}."
            )

    expected_keys = [f"frame_{index:06d}" for index in range(episode_length)]
    for event_path in paths["events"].values():
        with np.load(event_path) as event_data:
            if sorted(event_data.files) != expected_keys:
                raise ValueError(f"{event_path}: event frame keys do not match episode length.")
            for frame_key in event_data.files:
                events = event_data[frame_key]
                fields = {"timestamp", "x", "y", "polarity"}
                if events.dtype.names is None or set(events.dtype.names) != fields:
                    raise ValueError(f"{event_path}:{frame_key}: invalid structured event dtype {events.dtype}.")
                if events.dtype["timestamp"] != np.dtype("int64"):
                    raise ValueError(f"{event_path}:{frame_key}: event timestamp must be int64.")
                if events.dtype["x"] != np.dtype("int16") or events.dtype["y"] != np.dtype("int16"):
                    raise ValueError(f"{event_path}:{frame_key}: event x/y must be int16.")
                if events.dtype["polarity"] != np.dtype("int8"):
                    raise ValueError(f"{event_path}:{frame_key}: event polarity must be int8.")
                if len(events):
                    height, width = features["observation.images.DAVIS"]["shape"][:2]
                    if (
                        np.any(events["x"] < 0) or np.any(events["x"] >= width)
                        or np.any(events["y"] < 0) or np.any(events["y"] >= height)
                    ):
                        raise ValueError(f"{event_path}:{frame_key}: event coordinates are out of bounds.")
                    if not np.isin(events["polarity"], [0, 1]).all():
                        raise ValueError(f"{event_path}:{frame_key}: polarity must contain only 0 or 1.")
                    if np.any(np.diff(events["timestamp"]) < 0):
                        raise ValueError(f"{event_path}:{frame_key}: event timestamps are not sorted.")

    if "timestamp" in paths:
        timestamps = _read_jsonl(paths["timestamp"])
        if len(timestamps) != episode_length:
            raise ValueError(f"{paths['timestamp']}: rows={len(timestamps)}, expected={episode_length}.")
        for frame_index, timestamp in enumerate(timestamps):
            if timestamp.get("frame_index") != frame_index:
                raise ValueError(f"{paths['timestamp']}: invalid frame_index at row {frame_index}.")
            if type(timestamp.get("start")) is not int or type(timestamp.get("end")) is not int:
                raise ValueError(f"{paths['timestamp']}: start/end must be integer microseconds.")
            if timestamp["start"] > timestamp["end"]:
                raise ValueError(f"{paths['timestamp']}: start > end at frame {frame_index}.")
            event_fields = ["event_start", "event_end", "num_events"]
            if any(key in timestamp for key in event_fields) and not all(
                type(timestamp.get(key)) is int for key in event_fields
            ):
                raise ValueError(f"{paths['timestamp']}: invalid event summary at frame {frame_index}.")
    return df, paths


def _copy_sidecars(paths, dst, new_episode_index):
    stem = f"episode_{new_episode_index:06d}"
    for key, source in paths["videos"].items():
        output = dst / "videos" / "chunk-000" / key
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output / f"{stem}.mp4")
    for key, source in paths["events"].items():
        output = dst / "events" / "chunk-000" / key
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output / f"{stem}.npz")
    if "timestamp" in paths:
        output = dst / "timestamps" / "chunk-000" / TIMESTAMP_KEY
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["timestamp"], output / f"{stem}.jsonl")


def _scalar_stats(values):
    values = np.asarray(values)
    return {
        "min": [values.min().item()], "max": [values.max().item()],
        "mean": [float(values.mean())], "std": [float(values.std())], "count": [len(values)],
    }


def _update_stats(stats_item, df, new_episode_index, features):
    stats_item = json.loads(json.dumps(stats_item))
    stats_item["episode_index"] = new_episode_index
    for key in list(stats_item["stats"]):
        if key not in features:
            del stats_item["stats"][key]
    for key in ["frame_index", "episode_index", "index", "task_index"]:
        if key in stats_item["stats"] and key in df:
            stats_item["stats"][key] = _scalar_stats(df[key].to_numpy())
    return stats_item


def _write_meta(dst, info, episodes, stats, tasks, features, total_frames, split_name):
    _write_jsonl(dst / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(dst / "meta" / "episodes_stats.jsonl", stats)
    _write_jsonl(dst / "meta" / "tasks.jsonl", tasks)
    info = json.loads(json.dumps(info))
    info["features"] = features
    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info["total_videos"] = len(_feature_keys(info, "video")) * len(episodes)
    info["total_chunks"] = 1 if episodes else 0
    info["splits"] = {split_name: f"0:{len(episodes)}"}
    if not _feature_keys(info, "event"):
        info.pop("event_path", None)
    with (dst / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)


def merge_datasets(src_dirs, dst_dir):
    """Merge compatible single-task event datasets and rebuild continuous indices."""
    if not src_dirs:
        raise ValueError("src_dirs must contain at least one dataset.")
    loaded = [_load_meta(src_dir) for src_dir in src_dirs]
    first_info, _, _, first_tasks = loaded[0]
    for src_dir, (info, _, _, tasks) in zip(src_dirs[1:], loaded[1:], strict=True):
        if info["features"] != first_info["features"]:
            raise ValueError(f"{src_dir}: feature schema differs from the first dataset.")
        if info["fps"] != first_info["fps"]:
            raise ValueError(f"{src_dir}: fps differs from the first dataset.")
        if tasks != first_tasks:
            raise ValueError(
                f"{src_dir}: task metadata differs. Build multi-task datasets in the build stage."
            )

    dst = _prepare_output(dst_dir)
    video_keys = _feature_keys(first_info, "video")
    event_keys = _feature_keys(first_info, "event")
    output_episodes, output_stats = [], []
    total_frames = 0
    for src_dir, (_, episodes, stats, _) in zip(src_dirs, loaded, strict=True):
        stats_by_episode = {item["episode_index"]: item for item in stats}
        for episode in episodes:
            df, paths = _validate_episode(src_dir, episode, video_keys, event_keys, first_info["features"])
            new_index = len(output_episodes)
            df = df.copy()
            df["episode_index"] = np.full(len(df), new_index, dtype=np.int64)
            df["index"] = np.arange(total_frames, total_frames + len(df), dtype=np.int64)
            df.to_parquet(dst / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")
            _copy_sidecars(paths, dst, new_index)
            new_episode = dict(episode)
            new_episode["episode_index"] = new_index
            output_episodes.append(new_episode)
            output_stats.append(
                _update_stats(stats_by_episode[episode["episode_index"]], df, new_index, first_info["features"])
            )
            total_frames += len(df)
    _write_meta(
        dst, first_info, output_episodes, output_stats, first_tasks,
        json.loads(json.dumps(first_info["features"])), total_frames, "train",
    )
    print(f"total episodes: {len(output_episodes)}")
    print(f"total frames: {total_frames}")


def split_dataset(
    src_dir, dst_dir, split_name, split_index,
    video_del_pattern=None, state_del_pattern=None, feature_del_pattern=None,
):
    """Select episodes and retain every sidecar required by the selected feature schema."""
    video_del_pattern = video_del_pattern or []
    state_del_pattern = state_del_pattern or []
    feature_del_pattern = feature_del_pattern or []
    info, episodes, stats, tasks = _load_meta(src_dir)
    if len(split_index) != len(set(split_index)):
        raise ValueError(f"split_index contains duplicate indices: {split_index}")
    invalid = sorted(set(split_index) - set(range(len(episodes))))
    if invalid:
        raise IndexError(f"split_index contains invalid indices: {invalid}")

    features = json.loads(json.dumps(info["features"]))
    for key in list(features):
        if any(pattern in key for pattern in [*state_del_pattern, *feature_del_pattern]) or (
            features[key]["dtype"] == "video" and any(pattern in key for pattern in video_del_pattern)
        ):
            del features[key]
    video_keys = _feature_keys({"features": features}, "video")
    event_keys = _feature_keys({"features": features}, "event")
    source_video_keys = _feature_keys(info, "video")
    source_event_keys = _feature_keys(info, "event")
    stats_by_episode = {item["episode_index"]: item for item in stats}
    dst = _prepare_output(dst_dir)
    output_episodes, output_stats = [], []
    total_frames = 0

    for old_index in split_index:
        episode = episodes[old_index]
        df, all_paths = _validate_episode(
            src_dir, episode, source_video_keys, source_event_keys, info["features"]
        )
        paths = {
            "parquet": all_paths["parquet"],
            "videos": {key: all_paths["videos"][key] for key in video_keys},
            "events": {key: all_paths["events"][key] for key in event_keys},
        }
        if event_keys and "timestamp" in all_paths:
            paths["timestamp"] = all_paths["timestamp"]
        df = df.drop(columns=[key for key in df if key not in features], errors="ignore").copy()
        new_index = len(output_episodes)
        df["episode_index"] = np.full(len(df), new_index, dtype=np.int64)
        df["index"] = np.arange(total_frames, total_frames + len(df), dtype=np.int64)
        df.to_parquet(dst / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet")
        _copy_sidecars(paths, dst, new_index)
        new_episode = dict(episode)
        new_episode["episode_index"] = new_index
        output_episodes.append(new_episode)
        output_stats.append(_update_stats(stats_by_episode[old_index], df, new_index, features))
        total_frames += len(df)
    _write_meta(dst, info, output_episodes, output_stats, tasks, features, total_frames, split_name)
    print(f"total episodes: {len(output_episodes)}")
    print(f"total frames: {total_frames}")


def reindex_datasets(dataset_root, repo_id, episode_indices):
    """Remove episodes in place by rebuilding every dataset field and sidecar."""
    src_dir = Path(dataset_root, repo_id)
    info, episodes, _, _ = _load_meta(src_dir)
    invalid = sorted(set(episode_indices) - set(range(len(episodes))))
    if invalid:
        raise IndexError(f"episode_indices contains invalid indices: {invalid}")
    keep = [index for index in range(len(episodes)) if index not in set(episode_indices)]
    temporary = src_dir.with_name(f"{src_dir.name}_reindexed_temp")
    if temporary.exists():
        raise FileExistsError(f"Temporary directory already exists: {temporary}")
    split_name = next(iter(info.get("splits", {"train": ""})))
    split_dataset(src_dir, temporary, split_name, keep)
    shutil.rmtree(src_dir)
    temporary.rename(src_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Combine, split, or reindex LeRobot event datasets while preserving all sidecars."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    merge_parser = subparsers.add_parser("merge", help="Combine datasets into a new dataset directory.")
    merge_parser.add_argument("output_dir", type=Path)
    merge_parser.add_argument("input_dirs", type=Path, nargs="+")

    split_parser = subparsers.add_parser("split", help="Copy selected episodes into a new dataset directory.")
    split_parser.add_argument("input_dir", type=Path)
    split_parser.add_argument("output_dir", type=Path)
    split_parser.add_argument("split_name")
    split_parser.add_argument("episode_indices", type=int, nargs="+")

    reindex_parser = subparsers.add_parser("reindex", help="Remove episodes and reindex a dataset in place.")
    reindex_parser.add_argument("dataset_root", type=Path)
    reindex_parser.add_argument("repo_id")
    reindex_parser.add_argument("episode_indices", type=int, nargs="+")

    args = parser.parse_args()
    if args.operation == "merge":
        merge_datasets(args.input_dirs, args.output_dir)
    elif args.operation == "split":
        split_dataset(
            args.input_dir,
            args.output_dir,
            args.split_name,
            args.episode_indices,
        )
    else:
        reindex_datasets(args.dataset_root, args.repo_id, args.episode_indices)


if __name__ == "__main__":
    main()
