'''
Generate videos with event overlays from the raw event dataset.
Overlay modes include polarity pseudocolor and RGB events.
'''

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def _event_on_image_polarity_pseudo_color(image, events, img_color='rgb'):
    '''
    Overlay event polarity as pseudocolor: green for negative and red for positive events.
    :parameter events: event structure array.
    :parameter image: frame image, (h,w,c) array, dtype=uint8.
    :return image: (h,w,c) array, dtype=uint8.
    '''
    if events is None or len(events) == 0:
        return image
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['polarity']
        if p == 0:
            image[y, x] = [0, 255, 0]  # Green for negative events
        elif p == 1:
            if img_color == 'rgb':
                image[y, x] = [255, 0, 0]  # Red for positive events
            elif img_color == 'bgr':
                image[y, x] = [0, 0, 255]
    return image

def _event_on_image_rgb_events(image, events, img_color='rgb'):
    '''
    :parameter events: event structure array.
    :parameter image: frame image, (h,w,c) array, dtype=uint8.
    :return image: (h,w,c) array, dtype=uint8.
    '''
    # Reserved for the RGB-event representation.
    return image


def _event_on_image(image, events, img_color='rgb', type='polarity'):
    if type == 'polarity':
        return _event_on_image_polarity_pseudo_color(image, events, img_color)
    elif type == 'rgb':
        return _event_on_image_rgb_events(image, events, img_color)
    else:
        raise ValueError(f"Unknown event type: {type}")


def make_overlay_image(dataset_root, repo_id, type='polarity', num_window=2000):
    dataset_root = Path(dataset_root)
    record_dir = dataset_root / repo_id
    event_dir = record_dir/"events"/"chunk-000/observation.events.DAVIS"
    images_dir = record_dir/"video_frames"/"chunk-000"/"observation.images.DAVIS"
    output_dir = record_dir/"event_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    events = sorted(os.listdir(event_dir))
    episodes = sorted(os.listdir(images_dir))
    episodes_path = sorted([images_dir/episode for episode in episodes])
    event_stems = [Path(path).stem for path in events]
    episode_stems = [path.stem for path in episodes_path]
    if event_stems != episode_stems:
        raise ValueError(f"Event/image episodes do not match: {event_stems}, {episode_stems}.")

    for event_file, episode_path in zip(events, episodes_path, strict=True):
        print(f"processing {event_file}...")
        episode_images = sorted(os.listdir(episode_path))
        episode_images_path = [episode_path/image for image in episode_images]
        episode_events = np.load(event_dir/event_file)
        frame_keys = sorted(episode_events.files)
        expected_keys = [f"frame_{index:06d}" for index in range(len(episode_images_path))]
        if frame_keys != expected_keys:
            raise ValueError(
                f"{event_file}: event frames={len(frame_keys)}, images={len(episode_images_path)}."
            )
        for i, _ in enumerate(episode_images_path):
            image = cv2.imread(episode_images_path[i])
            if image is None:
                raise ValueError(f"Failed to read image: {episode_images_path[i]}")
            events = episode_events[f"frame_{i:06d}"]
            events = events[-num_window:]
            image = _event_on_image(np.zeros_like(image), events, "bgr", type)
            output = output_dir/episode_path.stem
            output.mkdir(parents=True, exist_ok=True)
            output_path = output/episode_images_path[i].name
            cv2.imwrite(output_path, image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--type", default="polarity")
    parser.add_argument("--num-window", type=int, default=4000)
    args = parser.parse_args()
    make_overlay_image(args.dataset_root, args.repo_id, args.type, args.num_window)



if __name__ == "__main__":
    main()
