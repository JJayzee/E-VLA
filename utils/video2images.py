# Decode videos into images for white balance, enhancement, and other processing.

import argparse
import json
from pathlib import Path

import av
import cv2


def video2images(src_dir):
    src_dir = Path(src_dir)
    video_dir = src_dir/"videos"/"chunk-000"/"observation.images.DAVIS"
    output_dir = src_dir/"video_frames"/"chunk-000"/"observation.images.DAVIS"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (src_dir / "meta" / "episodes.jsonl").open() as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    video_files = sorted(video_dir.glob("episode_*.mp4"))
    expected_stems = [f"episode_{episode['episode_index']:06d}" for episode in episodes]
    if [path.stem for path in video_files] != expected_stems:
        raise ValueError(f"{video_dir}: video episodes do not match episodes.jsonl.")

    for video_path, episode in zip(video_files, episodes, strict=True):
        video_file = video_path.name
        print(f"processing {video_file}...")
        episode_index = int(video_file.split(".")[0].split("_")[1])
        image_file_dir = output_dir / f"episode_{episode_index:06d}"
        image_file_dir.mkdir(parents=True, exist_ok=True)
        # read every frame from video, by pyav to support av1 encode
        container = av.open(video_path)
        frame_count = 0
        for frame_index, frame in enumerate(container.decode(video=0)):
            frame_image = frame.to_ndarray(format="bgr24")  # (h,w,c), uint8, bgr
            image_file = image_file_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(image_file, frame_image):
                raise OSError(f"Failed to write image: {image_file}")
            frame_count += 1
        container.close()
        if frame_count != episode["length"]:
            raise ValueError(
                f"{video_path}: decoded frames={frame_count}, episode length={episode['length']}."
            )
        output_count = len(list(image_file_dir.glob("frame_*.png")))
        if output_count != frame_count:
            raise ValueError(f"{image_file_dir}: found {output_count} PNGs after decoding {frame_count} frames.")

def main_video2image():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    video2images(args.dataset_dir)


if __name__ == "__main__":
    main_video2image()
