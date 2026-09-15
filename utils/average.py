# Compute mean grayscale intensity and event rate over a range of dataset episodes.
# Inputs: dataset root, repository ID, first episode, and last episode.
# Frame path: root_dir/repo_id/video_frames/chunk-000/observation.images.DAVIS/episode_{episode_index:06d}
# Event counts: root_dir/repo_id/timestamps/chunk-000/observation.timestamps.DAVIS/episode_{episode_index:06d}.jsonl
# Each JSONL row can contain {"frame_index": 11, "num_events": 2}.

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def compute_averages(root_dir: str, repo_id: str, start_episode: int, end_episode: int):
    """
    Compute mean grayscale intensity and event rate for an inclusive episode range.

    Args:
        root_dir: Dataset root directory.
        repo_id: Dataset repository ID.
        start_episode: First episode, inclusive.
        end_episode: Last episode, inclusive.

    Returns:
        tuple: (frame_gray_mean, event_rate_mean)
            frame_gray_mean: Mean grayscale intensity across all selected frames.
            event_rate_mean: Mean number of events per selected frame.
    """
    root = Path(root_dir) / repo_id

    all_gray_values = []
    all_num_events = []

    for ep_idx in range(start_episode, end_episode + 1):
        # Frame grayscale intensity.
        frame_dir = (
            root
            / "video_frames"
            / "chunk-000"
            / "observation.images.DAVIS"
            / f"episode_{ep_idx:06d}"
        )
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        for frame_path in frame_paths:
            print(f"processing episode {ep_idx} frame {frame_path.stem}...")
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Failed to read image: {frame_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            all_gray_values.append(float(np.mean(gray)))

        # Event rate.
        event_file = (
            root
            / "timestamps"
            / "chunk-000"
            / "observation.timestamps.DAVIS"
            / f"episode_{ep_idx:06d}.jsonl"
        )
        if not event_file.is_file():
            raise FileNotFoundError(f"Timestamp file not found: {event_file}")
        episode_num_events = []
        with open(event_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if type(data.get("num_events")) is not int:
                        raise ValueError(f"Missing or invalid num_events in {event_file}: {data}")
                    episode_num_events.append(data["num_events"])
        if len(frame_paths) != len(episode_num_events):
            raise ValueError(
                f"Episode {ep_idx}: image frames={len(frame_paths)}, timestamp rows={len(episode_num_events)}."
            )
        all_num_events.extend(episode_num_events)

    frame_gray_mean = float(np.mean(all_gray_values)) if all_gray_values else 0.0
    event_rate_mean = float(np.mean(all_num_events)) if all_num_events else 0.0
    total_image_frames = len(all_gray_values)
    total_event_frames = len(all_num_events)

    return frame_gray_mean, event_rate_mean, total_image_frames, total_event_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("start_episode", type=int)
    parser.add_argument("end_episode", type=int)
    args = parser.parse_args()
    frame_gray_mean, event_rate_mean, total_image_frames, total_event_frames = compute_averages(
        args.dataset_root, args.repo_id, args.start_episode, args.end_episode
    )
    print(f"Frame gray mean: {frame_gray_mean}")
    print(f"Event rate mean: {event_rate_mean}")
    print(f"Total image frames: {total_image_frames}")
    print(f"Total event frames: {total_event_frames}")
