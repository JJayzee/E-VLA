# Check continuity of dataset timestamp files and report missing frame indices.

"""
    1. check_timestamp_continuity(): report missing frames.
    2. reslice_frame_events(): reslice events at image exposure end times.
    3. write_event_timestamp_to_json(): write event times and counts to JSONL.
    4. check_num_frames_without_events(): count frames with no events.
    5. remove_and_rename(): replace original event and timestamp files with resliced files.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

#############################################################################
# utils functions

def _files_by_episode(directory, suffix):
    files = {}
    for path in sorted(Path(directory).glob(f"episode_*.{suffix}")):
        try:
            episode_index = int(path.stem.split("_")[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid episode filename: {path}") from error
        if episode_index in files:
            raise ValueError(f"Duplicate episode {episode_index} in {directory}.")
        files[episode_index] = path
    return files


def _matched_episode_files(timestamp_dir, event_dir):
    timestamp_files = _files_by_episode(timestamp_dir, "jsonl")
    event_files = _files_by_episode(event_dir, "npz")
    if timestamp_files.keys() != event_files.keys():
        missing_events = sorted(timestamp_files.keys() - event_files.keys())
        missing_timestamps = sorted(event_files.keys() - timestamp_files.keys())
        raise ValueError(
            f"Timestamp/event episodes do not match. Missing events={missing_events}, "
            f"missing timestamps={missing_timestamps}."
        )
    return [(index, timestamp_files[index], event_files[index]) for index in timestamp_files]


def _read_timestamps(timestamp_file):
    with Path(timestamp_file).open() as f:
        timestamps = [json.loads(line) for line in f if line.strip()]
    for frame_index, timestamp in enumerate(timestamps):
        if timestamp.get("frame_index") != frame_index:
            raise ValueError(f"{timestamp_file}: invalid frame_index at row {frame_index}.")
        if type(timestamp.get("start")) is not int or type(timestamp.get("end")) is not int:
            raise ValueError(f"{timestamp_file}: start/end must be integer microseconds.")
    return timestamps


def check_timestamp_continuity(dataset_root, repo_id):
    # Check timestamp continuity and report missing frame indices.
    timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000", "observation.timestamps.DAVIS")
    timestamp_files = _files_by_episode(timestamp_dir, "jsonl")
    total_episodes = len(timestamp_files)
    print(f"Find {total_episodes} episodes in dataset {repo_id}")

    repo_dir = Path(dataset_root, repo_id)
    with open(repo_dir / "meta" / "info.json") as f:
        info = json.load(f)
    expected_episodes = set(range(info["total_episodes"]))
    if set(timestamp_files) != expected_episodes:
        raise ValueError(
            f"Timestamp episodes do not match info.total_episodes: "
            f"expected={sorted(expected_episodes)}, actual={sorted(timestamp_files)}."
        )
    frame_period_us = round(1_000_000 / info["fps"])
    gap_threshold_us = round(frame_period_us * 1.2)
    missing_index = {episode_index: [] for episode_index in timestamp_files}
    episode_total_frames = 0
    for episode_index, timestamp_file in timestamp_files.items():
        timestamps = _read_timestamps(timestamp_file)
        episode_total_frames += len(timestamps)
        for index in range(1, len(timestamps)):
            gap = timestamps[index]["start"] - timestamps[index - 1]["start"]
            if gap > gap_threshold_us:
                print(
                    f"Episode {episode_index}: timestamp gap {gap}us between "
                    f"frames {index - 1} and {index}"
                )
                missing_index[episode_index].append(index - 1)
    missing_total = sum(len(indices) for indices in missing_index.values())
    print(f"missing frames / total frames: {missing_total} / {episode_total_frames}")
    return missing_index

def reslice_frame_events(dataset_root, repo_id):
    # Reslice events at image exposure end times.
    # Merge episode events first, excluding repeated frame event arrays.
    # The merged array contains unique captured events. Resliced frames may still
    # repeat events when the source images are duplicates.
    timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000", "observation.timestamps.DAVIS")
    event_dir = Path(dataset_root, repo_id, "events", "chunk-000", "observation.events.DAVIS")
    matched_files = _matched_episode_files(timestamp_dir, event_dir)
    total_episodes = len(matched_files)
    print(f"Find {total_episodes} episodes in dataset {repo_id}")
    new_event_dir = Path(dataset_root, repo_id, "events", "chunk-000-resliced", "observation.events.DAVIS")
    new_event_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(dataset_root, repo_id, "meta", "info.json")) as f:
        frame_period_us = round(1_000_000 / json.load(f)["fps"])
    for episode_index, timestamp_file, event_file in matched_files:
        print(f"Processing episode {episode_index}...")
        temporary_dir = Path(dataset_root, repo_id, "temp_events", f"episode_{episode_index:06d}")
        temporary_dir.mkdir(parents=True, exist_ok=False)
        timestamp_data = _read_timestamps(timestamp_file)
        total_frames = len(timestamp_data)

        with np.load(event_file) as event_data:
            expected_keys = [f"frame_{index:06d}" for index in range(total_frames)]
            if sorted(event_data.files) != expected_keys:
                raise ValueError(f"{event_file}: event frame keys do not match {timestamp_file}.")
            total_events = None
            for frame_key in expected_keys:
                frame_events = event_data[frame_key]
                if total_events is None:
                    total_events = frame_events
                elif len(total_events):
                    frame_events = frame_events[frame_events["timestamp"] > total_events[-1]["timestamp"]]
                    total_events = np.concatenate([total_events, frame_events])
                else:
                    total_events = np.concatenate([total_events, frame_events])

        for frame_index in range(total_frames):
            frame_end = timestamp_data[frame_index]["end"]
            frame_start = frame_end - frame_period_us
            slice_mask = (total_events['timestamp'] > frame_start) & (total_events['timestamp'] <= frame_end)
            sliced_events = total_events[slice_mask]
            np.save(temporary_dir / f"frame_{frame_index:06d}.npy", sliced_events)
        new_event_npz = new_event_dir / f"episode_{episode_index:06d}.npz"
        frame_npys = sorted(temporary_dir.glob("frame_*.npy"))
        np.savez_compressed(new_event_npz, **{path.stem: np.load(path) for path in frame_npys})
        shutil.rmtree(temporary_dir)
        temporary_dir.parent.rmdir()

def write_event_timestamp_to_json(dataset_root, repo_id):
    # Write event timestamps and counts to timestamp JSONL files.
    timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000", "observation.timestamps.DAVIS")
    new_timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000-resliced", "observation.timestamps.DAVIS")
    new_timestamp_dir.mkdir(parents=True, exist_ok=True)
    event_dir = Path(dataset_root, repo_id, "events", "chunk-000-resliced", "observation.events.DAVIS")
    matched_files = _matched_episode_files(timestamp_dir, event_dir)
    print(f"Find {len(matched_files)} episodes in dataset {repo_id}")

    for episode_index, timestamp_file, event_file in matched_files:
        timestamp_data = _read_timestamps(timestamp_file)
        with np.load(event_file) as event_data:
            expected_keys = [f"frame_{index:06d}" for index in range(len(timestamp_data))]
            if sorted(event_data.files) != expected_keys:
                raise ValueError(f"{event_file}: event frame keys do not match {timestamp_file}.")
            for timestamp in timestamp_data:
                frame_events = event_data[f"frame_{timestamp['frame_index']:06d}"]
                if len(frame_events) == 0:
                    timestamp["event_start"] = -1
                    timestamp["event_end"] = -1
                    timestamp["num_events"] = 0
                else:
                    timestamp["event_start"] = int(frame_events[0]["timestamp"])
                    timestamp["event_end"] = int(frame_events[-1]["timestamp"])
                    timestamp["num_events"] = len(frame_events)
        with open(new_timestamp_dir / f"episode_{episode_index:06d}.jsonl", "w") as f:
            for timestamp in timestamp_data:
                f.write(json.dumps(timestamp) + "\n")

def check_num_frames_without_events(dataset_root, repo_id):
    # Count frames without events in each episode's timestamp file.
    timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000-resliced", "observation.timestamps.DAVIS")
    timestamp_files = list(_files_by_episode(timestamp_dir, "jsonl").values())
    total_episodes = len(timestamp_files)
    print(f"Find {total_episodes} episodes in dataset {repo_id}")

    for timestamp_file in timestamp_files:
        episode_index = int(timestamp_file.stem.split("_")[1])
        num_frames_without_events = 0
        with open(timestamp_file, "r") as f:
            for line in f:
                if json.loads(line)["num_events"] == 0:
                    num_frames_without_events += 1
        print(f"Episode {episode_index}: {num_frames_without_events} frames without events")

def remove_and_rename(dataset_root, repo_id):
    repo_dir = Path(dataset_root, repo_id)
    old_events_dir = repo_dir / "events" / "chunk-000"
    new_events_dir = repo_dir / "events" / "chunk-000-resliced"
    old_timestamps_dir = repo_dir / "timestamps" / "chunk-000"
    new_timestamps_dir = repo_dir / "timestamps" / "chunk-000-resliced"
    if not new_events_dir.is_dir() or not new_timestamps_dir.is_dir():
        raise FileNotFoundError(
            f"Resliced event/timestamp directories must both exist: {new_events_dir}, {new_timestamps_dir}"
        )
    shutil.rmtree(old_events_dir)
    os.rename(new_events_dir, old_events_dir)
    shutil.rmtree(old_timestamps_dir)
    os.rename(new_timestamps_dir, old_timestamps_dir)

# end of utils functions
#############################################################################





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=["check", "reslice", "write-event-metadata", "check-empty", "activate-resliced"],
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    args = parser.parse_args()
    operations = {
        "check": check_timestamp_continuity,
        "reslice": reslice_frame_events,
        "write-event-metadata": write_event_timestamp_to_json,
        "check-empty": check_num_frames_without_events,
        "activate-resliced": remove_and_rename,
    }
    result = operations[args.operation](args.dataset_root, args.repo_id)
    if args.operation == "check":
        print(f"Missing index: {json.dumps(result, indent=4)}")


if __name__ == "__main__":
    main()
