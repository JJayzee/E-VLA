#!/usr/bin/env python
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def _events_to_image_by_accu_and_bayer(events: torch.Tensor, accu_mode: str) -> torch.Tensor:
    """
    Accumulate events in a single-channel Bayer image, then demosaic to RGB with OpenCV.

    Args:
        events: (N, 4) int32 tensor [x, y, polarity, timestamp], without padding.
        accu_mode: 'sum' or 'count'.

    Returns:
        (3, 260, 346) float32 RGB event image in [0, 1].
    """
    device = events.device
    height, width = 260, 346

    bayer_image = np.zeros((height, width), dtype=np.float32)

    if len(events) > 0:
        x = events[:, 0].cpu().numpy().astype(np.int16)
        y = events[:, 1].cpu().numpy().astype(np.int16)
        if accu_mode == "sum":
            pol = events[:, 2].cpu().numpy().astype(np.float32)
            # Match online processing by treating polarity 0 as -1.
            pol = np.where(pol == 0, -1, pol)

        for i in range(len(x)):
            if 0 <= x[i] < width and 0 <= y[i] < height:
                if accu_mode == "sum":
                    bayer_image[y[i], x[i]] += pol[i]
                elif accu_mode == "count":
                    bayer_image[y[i], x[i]] += 1.0
                else:
                    raise ValueError(f"Unknown accumulation mode: {accu_mode}")

    log_scale = 100.0  # Contrast enhancement factor.

    if accu_mode == "sum":
        max_abs_val = np.abs(bayer_image).max()
        if max_abs_val > 0:
            bayer_normalized = bayer_image / max_abs_val  # [-1, 1]
            sign = np.sign(bayer_normalized)
            abs_bayer_normalized = np.abs(bayer_normalized)
            bayer_normalized = sign * np.log1p(abs_bayer_normalized * log_scale) / np.log1p(log_scale)
        else:
            bayer_normalized = bayer_image
        bayer_normalized = (bayer_normalized + 1.0) / 2.0  # [0, 1]

    elif accu_mode == "count":
        max_val = bayer_image.max()
        if max_val > 0:
            bayer_normalized = bayer_image / max_val  # [0, 1]
            bayer_normalized = np.log1p(bayer_normalized * log_scale) / np.log1p(log_scale)
        else:
            bayer_normalized = bayer_image
    else:
        raise ValueError(f"Unknown accumulation mode: {accu_mode}")

    # OpenCV demosaicing requires uint8.
    bayer_uint8 = np.clip(bayer_normalized * 255, 0, 255).astype(np.uint8)
    event_image = cv2.cvtColor(bayer_uint8, cv2.COLOR_BAYER_RG2RGB)

    event_image = event_image.astype(np.float32) / 255.0  # [0, 1]
    event_image = torch.from_numpy(event_image).permute(2, 0, 1).to(device)  # (3,H,W)

    return event_image


def _events_to_image(events: torch.Tensor, event_repr: str = "accu_count") -> torch.Tensor:
    """
    Offline wrapper for SmolVLA's `_events_to_image`.

    Args:
        events: (N, 4) int32 tensor [x, y, polarity, timestamp].
        event_repr: "accu_count" or "accu_sum".

    Returns:
        (3, 260, 346) float32 tensor in [0, 1].
    """
    if event_repr == "accu_count":
        return _events_to_image_by_accu_and_bayer(events, accu_mode="count")
    elif event_repr == "accu_sum":
        return _events_to_image_by_accu_and_bayer(events, accu_mode="sum")
    else:
        raise ValueError(f"Unknown event representation: {event_repr}")


def prepare_events_offline_single(events: torch.Tensor, event_repr: str = "accu_count") -> torch.Tensor:
    """
    Offline `prepare_events` processes one (N, 4) sample.

    - Filter x<0 or y<0 as padding.
    - Return a (3,H,W) image in [0, 1].
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Expected events with shape (N,4), got {events.shape}")

    img = _events_to_image(events, event_repr=event_repr)  # (3,260,346), [0,1]
    return img


def save_image(tensor_img: torch.Tensor, out_path: Path):
    """
    Save a (3,H,W) tensor in [0, 1] as a PNG image.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tensor_img.ndim != 3 or tensor_img.shape[0] != 3:
        raise ValueError(f"Expected image tensor with shape (3,H,W), got {tensor_img.shape}")

    img_01 = torch.clamp(tensor_img, 0.0, 1.0)

    img_np = img_01.mul(255.0).byte().cpu().permute(1, 2, 0).numpy()  # (H,W,3)
    # OpenCV writes BGR, so reverse RGB channel order.
    if not cv2.imwrite(str(out_path), img_np[:, :, ::-1]):
        raise OSError(f"Failed to write image: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--event-repr", choices=["accu_count", "accu_sum"], default="accu_count")
    args = parser.parse_args()
    event_dir = args.dataset_dir / "events" / "chunk-000" / "observation.events.DAVIS"
    output_dir = args.output_dir or args.dataset_dir / "rgb_event_images"
    event_files = sorted(event_dir.glob("*.npz"))
    for event_file in event_files:
        print(f"processing {event_file}...")
        episode_index = int(event_file.name.split(".")[0].split("_")[1])
        episode_output_dir = output_dir / f"episode_{episode_index:06d}"
        episode_output_dir.mkdir(parents=True, exist_ok=True)
        arr = np.load(event_file)
        frame_keys = sorted(arr.keys())
        for frame_key in frame_keys:
            events = arr[frame_key]
            data = np.stack(
                [events["x"], events["y"], events["polarity"], events["timestamp"]],
                axis=-1,
            ).astype(np.int32)
            events = torch.from_numpy(data)
            img = prepare_events_offline_single(events, event_repr=args.event_repr)
            save_image(img, episode_output_dir / f"{frame_key}.png")

if __name__ == "__main__":
    main()
