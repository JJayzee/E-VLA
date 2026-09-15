#!/usr/bin/env python3
"""
Check frame counts for a selected episode-video feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import av

###################################################################
# utils functions

def get_frame_counts(video_path: Path) -> Tuple[int, int, str]:
    try:
        container = av.open(str(video_path))
    except Exception:
        return 0, 0, "open_failed"

    expected = 0
    actual = 0
    status = "ok"

    try:
        if not container.streams.video:
            return 0, 0, "no_video_stream"

        stream = container.streams.video[0]
        expected = int(stream.frames or 0)
        for _ in container.decode(video=0):
            actual += 1

        if actual == 0:
            status = "no_frames"
        elif expected > 0 and actual != expected:
            status = "count_mismatch"
    finally:
        container.close()

    return expected, actual, status


def check_repo_videos(repo_dir: Path, key: str) -> List[Dict[str, object]]:
    video_dir = (
        repo_dir
        / "videos"
        / "chunk-000"
        / key
    )
    video_files = sorted(video_dir.glob("episode_*.mp4"))
    with (repo_dir / "meta" / "episodes.jsonl").open() as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    expected_stems = [f"episode_{item['episode_index']:06d}" for item in episodes]
    actual_stems = [path.stem for path in video_files]
    if actual_stems != expected_stems:
        raise ValueError(
            f"{video_dir}: video episodes do not match episodes.jsonl: "
            f"expected={expected_stems}, actual={actual_stems}."
        )
    results: List[Dict[str, object]] = []

    for video_file, episode in zip(video_files, episodes, strict=True):
        stream_frames, actual, status = get_frame_counts(video_file)
        expected = episode["length"]
        if status == "ok" and actual != expected:
            status = "episode_length_mismatch"
        results.append(
            {
                "file": str(video_file),
                "episode": video_file.stem,
                "expected_frames": expected,
                "actual_frames": actual,
                "stream_frames": stream_frames,
                "status": status,
            }
        )
    return results

def remove_and_rename(repo_dir):
    repo_dir = Path(repo_dir)
    videos_dir = repo_dir / "videos" / "chunk-000"
    davis_dir = videos_dir / "observation.images.DAVIS"
    davis_wb_dir = videos_dir / "observation.images.DAVISWB"

    if davis_dir.exists():
        if davis_dir.is_dir():
            for item in davis_dir.iterdir():
                if item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                else:
                    item.unlink()
            davis_dir.rmdir()
        else:
            davis_dir.unlink()

    if davis_wb_dir.exists():
        davis_wb_dir.rename(davis_dir)

# end of utils functions
###################################################################



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--video-key", default="observation.images.DAVIS")
    args = parser.parse_args()
    results = check_repo_videos(args.dataset_dir, args.video_key)
    bad = [result for result in results if result["status"] != "ok"]
    print(f"Checked {len(results)} videos in {args.dataset_dir}.")
    print(f"Corrupted videos: {len(bad)}")
    for item in bad:
        print(
            f"{item['episode']} expected={item['expected_frames']} "
            f"actual={item['actual_frames']} status={item['status']}"
        )
    if bad:
        raise RuntimeError(f"Found {len(bad)} corrupted or inconsistent videos.")

if __name__ == "__main__":
    main()
