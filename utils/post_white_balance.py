'''
Apply white balance to raw DAVIS images, then write the result as video.
'''

import argparse
import os
from pathlib import Path

import av
import cv2
import numpy as np


def apply_white_balance(image, light_type):
    # input: uint8, (H,W,C), bgr

    image = image.astype(np.float32) / 255.0
    image = srgb_to_linear(image)

    if light_type == "room":
        gains = np.array([1.2366, 0.6378, 1.6042])  # bgr
    elif light_type == "lamp":
        gains = np.array([1.2891957, 0.6607546, 1.4066648]) # bgr
    else:
        raise ValueError(f"Undefined light type {light_type}, must in [room, lamp].")

    if light_type == "room":
        # lum estimation
        lum = 0.2126 * image[..., 2] + 0.7152 * image[..., 1] + 0.0722 * image[..., 0]
        # green calibration, by experiment
        lum_key = np.array([ 0.0, 0.4, 0.6,  0.7,  0.8,  0.84, 0.86, 0.9, 1.0])
        adj_gain = np.array([1.0, 0.9, 0.72, 0.68, 0.72, 0.85, 1.0,  1.2, 1.3])
        green_boost = np.interp(lum, lum_key, adj_gain)

    image *= gains.reshape(1, 1, 3)

    if light_type == "room":
        image[..., 1] *= green_boost

    image = linear_to_srgb(image) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)
    
    
def srgb_to_linear(image):
    # input: float32, (H,W,C), [0,1]
    # output: float32, (H,W,C), [0,1]
    a = 0.055
    mask = image <= 0.04045
    out = np.empty_like(image)
    out[mask] = image[mask] / 12.92
    out[~mask] = ((image[~mask] + a) / (1 + a)) ** 2.4
    return out

def linear_to_srgb(image):
    # input: float32, (H,W,C), [0,1]
    # output: float32, (H,W,C), [0,1]
    a = 0.055
    mask = image <= 0.0031308
    out = np.empty_like(image)
    out[mask] = image[mask] * 12.92
    out[~mask] = (1 + a) * (image[~mask] ** (1 / 2.4)) - a
    return out

def post_white_balance(src_dir, light_type="room"):

    src_dir = Path(src_dir)
    image_dir = src_dir/"video_frames"/"chunk-000"/"observation.images.DAVIS"
    output_dir = src_dir/"videos"/"chunk-000"/"observation.images.DAVISWB"
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_dir_name in sorted(os.listdir(image_dir)):
        print(f"processing {episode_dir_name}...")
        wb_images = []

        # for lowlight dataset
        # episode_index = int(episode_dir_name.split("_")[1])
        # if episode_index < 60:
        #     light_type = "lamp"
        # else:
        #     light_type = "room"
        
        episode_dir = image_dir / episode_dir_name
        for image_file in sorted(os.listdir(episode_dir)):
            image_path = episode_dir / image_file
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if image.shape != (260, 346, 3) or image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 image (260,346,3), got {image.dtype} {image.shape}: {image_path}")
            wb_image = apply_white_balance(image, light_type)
            cv2.imwrite(image_path, wb_image)
            wb_images.append(wb_image)
        
        # write video to av1 format .mp4 file
        # LeRobot records with CRF 30; re-encoding at that setting loses quality.
        # Use CRF 22 to retain more detail.
        # libaom-av1 can produce higher quality but encodes much more slowly.
        output_path = output_dir / f"{episode_dir_name}.mp4"
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
        for img in wb_images:
            video_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            packet = stream.encode(video_frame)
            if packet:
                container_out.mux(packet)
        
        # flush encoder
        for packet in stream.encode():
            container_out.mux(packet)
        
        container_out.close()

        print(f"video saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--light-type", choices=["room", "lamp"], required=True)
    args = parser.parse_args()
    post_white_balance(args.dataset_dir, args.light_type)
