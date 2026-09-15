# This file implements many event representations for SmolVLA.
# They share common input and output format:
# input: (N, 4) event tensor, torch.int32, [x, y, polarity, timestamp], arbitrary length.
# output: (3, 260, 346) tensor, torch.float32, [0, 1], RGB event image
# Public representations are accu_count and accu_sum. The E2VID-style stub raises.

import cv2
import numpy as np
import torch


def _events_to_image(events: torch.Tensor, event_repr: str) -> torch.Tensor:
    """Convert event tensor to 3-channel event image.
    This is a wrapper function for different event representations.

    Args:
        events: (N, 4) tensor with [x, y, polarity, timestamp], torch.int32 on device,
        already unpadded, so can have arbitrary length.
        event_repr: implemented event representation, ``accu_count`` or ``accu_sum``

    Returns:
        (3, 260, 346) tensor with RGB event image in [0, 1] range
    """
    if event_repr == "accu_count":
        return _events_to_image_by_accu_and_bayer(events, accu_mode="count")
    elif event_repr == "accu_sum":
        return _events_to_image_by_accu_and_bayer(events, accu_mode="sum")
    else:
        raise ValueError(f"Unknown event representation: {event_repr}")


def _events_to_image_by_accu_and_bayer(events: torch.Tensor, accu_mode: str) -> torch.Tensor:
    """Convert event tensor to 3-channel event image using Bayer pattern.

    Events are accumulated into a single-channel Bayer image, then demosaiced using OpenCV.

    Args:
        events: (N, 4) tensor with [x, y, polarity, timestamp], torch.int32 on device,
        already unpadded, so can have arbitrary length.
        accu_mode: how to accumulate events, 'sum' or 'count'
        
    Returns:
        (3, 260, 346) tensor with RGB event image in [0, 1] range
    """
    if accu_mode not in {"count", "sum"}:
        raise ValueError(f"Unknown accumulation mode: {accu_mode}")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Events must have shape (N, 4), got {tuple(events.shape)}.")

    device = events.device
    height, width = 260, 346
    
    bayer_image = np.zeros((height, width), dtype=np.float32)
    
    if len(events) > 0:
        # Extract coordinates and polarity
        x = events[:, 0].cpu().numpy().astype(np.int16)  # column 0
        y = events[:, 1].cpu().numpy().astype(np.int16)  # column 1
        invalid = (x < 0) | (x >= width) | (y < 0) | (y >= height)
        if invalid.any():
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Event coordinates must satisfy 0 <= x < {width} and 0 <= y < {height}; "
                f"event {first} has (x={int(x[first])}, y={int(y[first])})."
            )
        if accu_mode == "sum":
            pol = events[:, 2].cpu().numpy().astype(np.float32)  # column 2, polarity values
            pol = np.where(pol == 0, -1, pol)
        
        # Accumulate events into single-channel Bayer image
        for i in range(len(x)):
            if accu_mode == "sum":
                bayer_image[y[i], x[i]] += pol[i]
            elif accu_mode == "count":
                bayer_image[y[i], x[i]] += 1.0  
    
    log_scale = 100.0  # Adjust this to control contrast enhancement (higher = more contrast)

    if accu_mode == "sum":
        """ accumulate events by summing polarity values
        most no events pixels are mapped to 0.5 """
        max_abs_val = np.abs(bayer_image).max()
        if max_abs_val > 0:
            bayer_normalized = bayer_image / max_abs_val    # norm to [-1, 1]
            sign = np.sign(bayer_normalized)
            abs_bayer_normalized = np.abs(bayer_normalized)
            bayer_normalized = sign * np.log1p(abs_bayer_normalized * log_scale)/np.log1p(log_scale)  # range [-1, 1]
        else:
            bayer_normalized = bayer_image   # all zeros
        bayer_normalized = (bayer_normalized + 1.0) / 2.0  # range [0, 1]

    elif accu_mode == "count":
        """ accumulate events by counting the number of events
        most no events pixels are mapped to 0 """
        max_val = bayer_image.max()
        if max_val > 0:
            bayer_normalized = bayer_image / max_val    # norm to [0, 1]
            bayer_normalized = np.log1p(bayer_normalized * log_scale)/np.log1p(log_scale)  # range [0, 1]
        else:
            bayer_normalized = bayer_image   # all zeros

    else:
        raise ValueError(f"Unknown accumulation mode: {accu_mode}")
    
    # Use OpenCV for demosaicing (RGGB pattern -> RGB)
    # OpenCV requires uint8 input, so we need to convert
    bayer_uint8 = np.clip((bayer_normalized * 255), 0, 255).astype(np.uint8)
    event_image = cv2.cvtColor(bayer_uint8, cv2.COLOR_BAYER_RG2RGB)
    
    # Convert back to float32 and restore original scale
    event_image = event_image.astype(np.float32)
    event_image = event_image / 255.0  # range [0, 1]

    event_image = torch.from_numpy(event_image).permute(2, 0, 1).to(device)
    
    return event_image


def _events_to_image_by_E2VID_style(events: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("E2VID style event representation is not implemented yet")
