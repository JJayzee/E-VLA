# check LeRobotEventDataset

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def check_parquet(src_dir, episode_idx):
    # check base information of parquet data
    src_dir = Path(src_dir)
    pq_dir = src_dir / "data" / "chunk-000"
    pq_files = sorted(pq_dir.glob("*.parquet"))
    print(f"Find {len(pq_files)} episodes in dataset.")

    episode_pq = pq_files[episode_idx]
    print(f"Checking episode {episode_idx} ...")
    df = pd.read_parquet(episode_pq)
    print(f"Find {len(df)} frames in this episode.")
    print("Column names and dtpes:\n", df.dtypes)
    
    # print("Data head and tail:")
    # print(df.head())
    # print(df.tail())

    obs_state = df["observation.state"].values
    print("begin")
    print(f"Observation state: {obs_state[:]}")
    
    # Convert nested dataset values to NumPy arrays if needed.
    # if isinstance(action_data[0], (list, np.ndarray)):
    #     action_array = np.array([np.array(action) for action in action_data])
    # else:
    #     action_array = action_data.reshape(-1, 1)

    # if isinstance(obs_state_data[0], (list, np.ndarray)):
    #     obs_state_array = np.array([np.array(obs) for obs in obs_state_data])
    # else:
    #     obs_state_array = obs_state_data.reshape(-1, 1)

    # print(f"Action array shape: {action_array.shape}")
    # print(f"Observation state array shape: {obs_state_array.shape}")


def edit_parquet(src_dir, episode_idx):
    # edit some items in parquet data
    src_dir = Path(src_dir)
    pq_dir = src_dir / "data" / "chunk-000"
    pq_file = pq_dir / f"episode_{episode_idx:06d}.parquet"
    if not pq_file.is_file():
        raise FileNotFoundError(f"Parquet file not found: {pq_file}")
    df = pd.read_parquet(pq_file)
    df["index"] += 1
    output = pq_dir / f"episode_{episode_idx:06d}_edited.parquet"
    df.to_parquet(output)
    print(f"Saved edited parquet to {output}")


def check_none_zero_video(video_dir):
    # check if the videos are all dark dead in a dir.
    video_files = sorted(Path(video_dir).glob("*.mp4"))

    for video_file in video_files:
        print(f"checking video: {video_file.name}")
        video = cv2.VideoCapture(str(video_file))
        if not video.isOpened():
            raise ValueError(f"Failed to open video: {video_file}")
        # check every frame
        i = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            none_zero_pixel = np.sum(frame != 0)
            if none_zero_pixel == 0:
                print(f"all-zero frame: {i}")
            i += 1
        video.release()
        if i == 0:
            raise ValueError(f"Video contains no frames: {video_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    check_parquet(args.dataset_dir, args.episode)

if __name__ == "__main__":
    main()
