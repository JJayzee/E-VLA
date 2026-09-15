'''
Generate videos with event overlays from the raw event dataset.
Overlay modes include polarity pseudocolor and RGB events.
'''

import argparse
import json
import os
from pathlib import Path

import av
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


def make_overlay_video_num_window(dataset_root, repo_id, type='polarity', num_window=2000):
    dataset_root = Path(dataset_root)
    record_dir = dataset_root / repo_id
    event_dir = record_dir/"events"/"chunk-000/observation.events.DAVIS"
    video_dir = record_dir/"videos"/"chunk-000"/"observation.images.DAVIS"
    timestamp_dir = record_dir/"timestamps"/"chunk-000"/"observation.timestamps.DAVIS"
    output_dir = record_dir/"videos"/"chunk-000"/f"observation.images.DAVIS{num_window}"
    output_dir.mkdir(parents=True, exist_ok=True)
    events = sorted(os.listdir(event_dir))
    videos = sorted(os.listdir(video_dir))
    timestamps = sorted(os.listdir(timestamp_dir))
    event_dtype = np.dtype([
                ('timestamp', '<i8'),
                ('x', '<i2'),
                ('y', '<i2'),
                ('polarity', 'i1')
            ])

    event_stems = [Path(path).stem for path in events]
    video_stems = [Path(path).stem for path in videos]
    timestamp_stems = [Path(path).stem for path in timestamps]
    if event_stems != video_stems or event_stems != timestamp_stems:
        raise ValueError(
            f"Episode files do not match: events={event_stems}, videos={video_stems}, "
            f"timestamps={timestamp_stems}."
        )

    for event_file, video_file, timestamp_file in zip(events, videos, timestamps, strict=True):
        print(f"processing {event_file}...")
        overlay_images = []
        event_buffer = np.zeros(0, dtype=event_dtype)
        event_file = event_dir/event_file
        video_file = video_dir/video_file
        timestamp_file = timestamp_dir/timestamp_file
        episode_events = np.load(event_file)
        frame_keys = sorted(episode_events.files)
        print(f"num of event frame: {len(frame_keys)}")

        timestamp_data = []
        with open(timestamp_file, "r") as f:
            for line in f:
                timestamp_data.append(json.loads(line))
        if len(frame_keys) != len(timestamp_data):
            raise ValueError(
                f"{event_file}: event frames={len(frame_keys)}, timestamps={len(timestamp_data)}."
            )

        # read every frame from video, by pyav to support av1 encode
        container = av.open(video_file)
        frame_index = 0
        latest_timestamp = 0
        for frame in container.decode(video=0):
            frame_image = frame.to_ndarray(format="bgr24")
            frame_key = frame_keys[frame_index]
            frame_events = episode_events[frame_key]

            new_events_mask = frame_events['timestamp'] > latest_timestamp
            new_frame_events = frame_events[new_events_mask]
            if len(new_frame_events) > 0:
                event_buffer = np.concatenate([event_buffer, new_frame_events])
                latest_timestamp = event_buffer[-1]['timestamp']

            frame_end_timestamp = timestamp_data[frame_index]['end']

            # slice events before frame end timestamp
            lt_mask = event_buffer['timestamp'] <= frame_end_timestamp
            sliced_mask = np.zeros_like(lt_mask)
            if sum(lt_mask) > 0:
                true_indices = np.nonzero(lt_mask)[0]
                sliced_indices = true_indices[-num_window:]
                sliced_mask[sliced_indices] = True
            sliced_events = event_buffer[sliced_mask]
            overlay_images.append(_event_on_image(frame_image, sliced_events, 'bgr', type))

            # cut event buffer
            gt_mask = event_buffer['timestamp'] > frame_end_timestamp
            update_mask = sliced_mask | gt_mask
            event_buffer = event_buffer[update_mask]
            frame_index += 1

        container.close()
        if frame_index != len(frame_keys):
            raise ValueError(f"{video_file}: video frames={frame_index}, event frames={len(frame_keys)}.")
        
        # write video to av1 format .mp4 file
        # LeRobot records with CRF 30; re-encoding at that setting loses quality.
        # Use CRF 22 to retain more detail.
        # libaom-av1 can produce higher quality but encodes much more slowly.
        output_path = output_dir/f"{event_file.stem}.mp4"
        container_out = av.open(output_path, mode="w")
        stream = container_out.add_stream("libsvtav1", rate=30)
        stream.width = 346
        stream.height = 260
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": "22",
            "g": "2",
        }
        
        # encode frame, notice bgr24 to match channel order
        for img in overlay_images:
            video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            packet = stream.encode(video_frame)
            if packet:
                container_out.mux(packet)
        
        # flush encoder
        for packet in stream.encode():
            container_out.mux(packet)
        
        container_out.close()

        print(f"video saved to {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--type", default="polarity")
    parser.add_argument("--num-window", type=int, default=4000)
    args = parser.parse_args()
    make_overlay_video_num_window(args.dataset_root, args.repo_id, args.type, args.num_window)



if __name__ == "__main__":
    main()
