"""Prepare and activate metadata profiles for shared evla datasets.

Physical video, event, and timestamp directories are never renamed or removed.  A
profile only changes the active ``meta/info.json`` and
``meta/episodes_stats.jsonl`` so both the event and image-only branches can read the
same dataset directory with their own feature contract.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import av

CANONICAL_VIDEO_KEY = "observation.images.DAVIS"
PROFILE_DIRNAME = "video_profiles"
BACKUP_DIRNAME = "_active_backup"


def _read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, items):
    with Path(path).open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _activate_metadata_pair(source_info, source_stats, active_info, active_stats):
    """Stage complete files, then atomically replace each destination separately.

    The two ``os.replace`` calls are not a transactional pair: interruption between
    them can temporarily leave active info and statistics from different profiles.
    """
    temporary_info = active_info.with_suffix(".json.switching")
    temporary_stats = active_stats.with_suffix(".jsonl.switching")
    if temporary_info.exists() or temporary_stats.exists():
        raise FileExistsError(f"Temporary metadata exists: {temporary_info}, {temporary_stats}")
    shutil.copy2(source_info, temporary_info)
    shutil.copy2(source_stats, temporary_stats)
    os.replace(temporary_info, active_info)
    os.replace(temporary_stats, active_stats)


def _backup_active_metadata(dataset_dir):
    """Save the original active metadata once before the first switch."""
    dataset_dir = Path(dataset_dir)
    backup_dir = dataset_dir / "meta" / PROFILE_DIRNAME / BACKUP_DIRNAME
    backup_info = backup_dir / "info.json"
    backup_stats = backup_dir / "episodes_stats.jsonl"
    if backup_info.is_file() and backup_stats.is_file():
        return backup_dir
    if backup_dir.exists():
        raise FileExistsError(f"Incomplete active metadata backup: {backup_dir}")
    backup_dir.mkdir(parents=True)
    shutil.copy2(dataset_dir / "meta" / "info.json", backup_info)
    shutil.copy2(dataset_dir / "meta" / "episodes_stats.jsonl", backup_stats)
    return backup_dir


def _validate_profile_name(profile_name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile_name):
        raise ValueError(f"Invalid profile name: {profile_name!r}")


def _load_active_metadata(dataset_dir):
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    with info_path.open() as f:
        info = json.load(f)
    episodes = _read_jsonl(episodes_path)
    episode_stats = _read_jsonl(stats_path)
    expected_indices = list(range(len(episodes)))
    if [episode["episode_index"] for episode in episodes] != expected_indices:
        raise ValueError(f"{episodes_path}: episode_index must be continuous from 0.")
    if [item["episode_index"] for item in episode_stats] != expected_indices:
        raise ValueError(f"{stats_path}: episode_index does not match episodes.jsonl.")
    if info["total_episodes"] != len(episodes):
        raise ValueError(
            f"{info_path}: total_episodes={info['total_episodes']}, actual={len(episodes)}."
        )
    return info, episodes, episode_stats


def _video_paths(dataset_dir, source_video_key, episodes, chunks_size):
    paths = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        episode_chunk = episode_index // chunks_size
        path = (
            Path(dataset_dir)
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / source_video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        paths.append(path)
    return paths


def _validate_video_headers(video_paths, video_feature, fps):
    expected_height, expected_width, expected_channels = video_feature["shape"]
    if expected_channels != 3:
        raise ValueError(f"Expected a 3-channel video feature, got {video_feature['shape']}.")
    for video_path in video_paths:
        try:
            container = av.open(video_path)
        except Exception as error:
            raise ValueError(f"Failed to open video: {video_path}") from error
        try:
            if not container.streams.video:
                raise ValueError(f"No video stream found: {video_path}")
            stream = container.streams.video[0]
            if (stream.height, stream.width) != (expected_height, expected_width):
                raise ValueError(
                    f"{video_path}: expected {expected_width}x{expected_height}, "
                    f"got {stream.width}x{stream.height}."
                )
            if stream.average_rate is not None and abs(float(stream.average_rate) - fps) > 1e-6:
                raise ValueError(f"{video_path}: expected {fps} FPS, got {stream.average_rate}.")
        finally:
            container.close()


def _validate_event_sidecars(dataset_dir, info, episodes):
    event_keys = [
        key
        for key, feature in info["features"].items()
        if feature["dtype"] == "event" or key.startswith("observation.events.")
    ]
    if not event_keys:
        raise ValueError("An event profile requires at least one event feature in active info.json.")
    event_path_template = info.get("event_path")
    if not event_path_template:
        raise ValueError("An event profile requires event_path in active info.json.")

    for event_key in event_keys:
        camera = event_key.removeprefix("observation.events.")
        timestamp_key = f"observation.timestamps.{camera}"
        for episode in episodes:
            episode_index = episode["episode_index"]
            episode_chunk = episode_index // info.get("chunks_size", 1000)
            event_path = Path(dataset_dir) / event_path_template.format(
                episode_chunk=episode_chunk,
                event_key=event_key,
                episode_index=episode_index,
            )
            timestamp_path = (
                Path(dataset_dir)
                / "timestamps"
                / "chunk-000"
                / timestamp_key
                / f"episode_{episode_index:06d}.jsonl"
            )
            for path in (event_path, timestamp_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Event profile sidecar not found: {path}")


def _select_video_stats(episode_stats, source_video_key, source_stats_key, include_events):
    stats_key = source_stats_key or source_video_key
    selected_stats = []
    has_selected_stats = [stats_key in item["stats"] for item in episode_stats]
    if any(has_selected_stats) and not all(has_selected_stats):
        raise ValueError(f"Video stats key {stats_key!r} exists for only part of the episodes.")

    for item in episode_stats:
        output = json.loads(json.dumps(item))
        for key in list(output["stats"]):
            if key.startswith("observation.images.") or (
                not include_events
                and (key.startswith("observation.events.") or key.startswith("observation.timestamps."))
            ):
                del output["stats"][key]
        if all(has_selected_stats):
            output["stats"][CANONICAL_VIDEO_KEY] = json.loads(json.dumps(item["stats"][stats_key]))
        selected_stats.append(output)
    return selected_stats, all(has_selected_stats)


def prepare_video_profile(
    dataset_dir,
    profile_name,
    source_video_key,
    include_events,
    source_stats_key=None,
):
    """Create an info/stats profile that maps one physical video version to the canonical training key."""
    _validate_profile_name(profile_name)
    if "/" in source_video_key:
        raise ValueError(f"Video feature key must not contain '/': {source_video_key}")

    dataset_dir = Path(dataset_dir)
    info, episodes, episode_stats = _load_active_metadata(dataset_dir)
    canonical_feature = info["features"].get(CANONICAL_VIDEO_KEY)
    source_feature = info["features"].get(source_video_key)
    if source_feature is None:
        if canonical_feature is None:
            raise ValueError(
                f"Neither {source_video_key!r} nor canonical feature {CANONICAL_VIDEO_KEY!r} "
                "exists in info.json."
            )
        source_feature = json.loads(json.dumps(canonical_feature))
    if source_feature["dtype"] != "video":
        raise ValueError(f"{source_video_key}: expected dtype='video', got {source_feature['dtype']!r}.")

    video_paths = _video_paths(
        dataset_dir,
        source_video_key,
        episodes,
        info.get("chunks_size", 1000),
    )
    _validate_video_headers(video_paths, source_feature, info["fps"])
    if include_events:
        _validate_event_sidecars(dataset_dir, info, episodes)

    profile_info = json.loads(json.dumps(info))
    for key in list(profile_info["features"]):
        dtype = profile_info["features"][key]["dtype"]
        is_event_sidecar = key.startswith("observation.events.") or key.startswith(
            "observation.timestamps."
        )
        if dtype in {"video", "image"} or (not include_events and (dtype == "event" or is_event_sidecar)):
            del profile_info["features"][key]
    profile_info["features"][CANONICAL_VIDEO_KEY] = source_feature
    profile_info["video_path"] = (
        f"videos/chunk-{{episode_chunk:03d}}/{source_video_key}/"
        "episode_{episode_index:06d}.mp4"
    )
    profile_info["total_videos"] = len(video_paths)
    if not include_events:
        profile_info.pop("event_path", None)

    profile_stats, has_video_stats = _select_video_stats(
        episode_stats,
        source_video_key,
        source_stats_key,
        include_events,
    )
    profile_dir = dataset_dir / "meta" / PROFILE_DIRNAME / profile_name
    if profile_dir.exists():
        raise FileExistsError(f"Video profile already exists: {profile_dir}")
    profile_dir.mkdir(parents=True)
    with (profile_dir / "info.json").open("w") as f:
        json.dump(profile_info, f, indent=2)
    _write_jsonl(profile_dir / "episodes_stats.jsonl", profile_stats)
    with (profile_dir / "profile.json").open("w") as f:
        json.dump(
            {
                "profile_name": profile_name,
                "source_video_key": source_video_key,
                "training_video_key": CANONICAL_VIDEO_KEY,
                "include_events": include_events,
                "source_stats_key": source_stats_key or source_video_key,
                "has_video_stats": has_video_stats,
            },
            f,
            indent=2,
        )
    print(f"Prepared video profile: {profile_dir}")
    if not has_video_stats:
        print(
            f"No episode image stats found for {source_stats_key or source_video_key}; "
            "the profile omits visual stats. SmolVLA uses identity normalization for visual inputs."
        )


def switch_video_profile(dataset_dir, profile_name):
    """Activate a profile by staging and individually replacing its metadata files."""
    _validate_profile_name(profile_name)
    dataset_dir = Path(dataset_dir)
    profile_dir = dataset_dir / "meta" / PROFILE_DIRNAME / profile_name
    info_path = profile_dir / "info.json"
    stats_path = profile_dir / "episodes_stats.jsonl"
    profile_path = profile_dir / "profile.json"
    for path in [info_path, stats_path, profile_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Video profile file not found: {path}")

    with profile_path.open() as f:
        profile = json.load(f)
    with info_path.open() as f:
        info = json.load(f)
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episode_stats = _read_jsonl(stats_path)
    if len(episode_stats) != len(episodes):
        raise ValueError(
            f"{stats_path}: stats episodes={len(episode_stats)}, dataset episodes={len(episodes)}."
        )
    video_paths = _video_paths(
        dataset_dir,
        profile["source_video_key"],
        episodes,
        info.get("chunks_size", 1000),
    )
    if info["total_videos"] != len(video_paths):
        raise ValueError(
            f"{info_path}: total_videos={info['total_videos']}, actual={len(video_paths)}."
        )

    _backup_active_metadata(dataset_dir)
    active_info = dataset_dir / "meta" / "info.json"
    active_stats = dataset_dir / "meta" / "episodes_stats.jsonl"
    _activate_metadata_pair(info_path, stats_path, active_info, active_stats)
    print(
        f"Activated profile {profile_name!r}: {CANONICAL_VIDEO_KEY} -> "
        f"{profile['source_video_key']}, include_events={profile['include_events']}"
    )


def restore_active_backup(dataset_dir):
    """Restore the metadata that was active before the first profile switch."""
    dataset_dir = Path(dataset_dir)
    backup_dir = dataset_dir / "meta" / PROFILE_DIRNAME / BACKUP_DIRNAME
    backup_info = backup_dir / "info.json"
    backup_stats = backup_dir / "episodes_stats.jsonl"
    for path in (backup_info, backup_stats):
        if not path.is_file():
            raise FileNotFoundError(f"Active metadata backup not found: {path}")
    _activate_metadata_pair(
        backup_info,
        backup_stats,
        dataset_dir / "meta" / "info.json",
        dataset_dir / "meta" / "episodes_stats.jsonl",
    )
    print(f"Restored original active metadata from {backup_dir}")


def describe_active_profile(dataset_dir):
    """Return the active logical/physical video mapping and event visibility."""
    dataset_dir = Path(dataset_dir)
    info, episodes, _ = _load_active_metadata(dataset_dir)
    video_features = {
        key: value for key, value in info["features"].items() if value["dtype"] in {"video", "image"}
    }
    event_features = [key for key, value in info["features"].items() if value["dtype"] == "event"]
    return {
        "dataset_dir": str(dataset_dir),
        "episodes": len(episodes),
        "video_path": info.get("video_path"),
        "video_features": video_features,
        "event_features": event_features,
        "event_path": info.get("event_path"),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate and create one metadata profile.")
    prepare.add_argument("dataset_dir", type=Path)
    prepare.add_argument("profile_name")
    prepare.add_argument("source_video_key")
    prepare.add_argument("--include-events", action="store_true")
    prepare.add_argument("--source-stats-key")

    switch = subparsers.add_parser("switch", help="Activate a prepared metadata profile.")
    switch.add_argument("dataset_dir", type=Path)
    switch.add_argument("profile_name")

    restore = subparsers.add_parser("restore", help="Restore metadata saved before the first switch.")
    restore.add_argument("dataset_dir", type=Path)

    status = subparsers.add_parser("status", help="Print the active metadata mapping.")
    status.add_argument("dataset_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.command == "prepare":
        prepare_video_profile(
            args.dataset_dir,
            args.profile_name,
            args.source_video_key,
            args.include_events,
            args.source_stats_key,
        )
    elif args.command == "switch":
        switch_video_profile(args.dataset_dir, args.profile_name)
    elif args.command == "restore":
        restore_active_backup(args.dataset_dir)
    elif args.command == "status":
        print(json.dumps(describe_active_profile(args.dataset_dir), indent=2))
