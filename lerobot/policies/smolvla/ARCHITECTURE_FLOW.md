# Event Adapter data flow

## Observation contract

| Key | Capture/storage | Policy input |
| --- | --- | --- |
| `observation.images.*` | NumPy RGB HWC `uint8` | Torch BCHW/BTCHW float |
| `observation.state` | Torch or NumPy `(D,)` | Batched float tensor |
| `observation.events.DAVIS` | Structured NumPy events | Torch `int32 (N,4)` or `(B,N,4)` |
| `observation.timestamps.DAVIS` | Exposure interval in microseconds | Online window boundary |
| `task` | String | String or list of strings |

Dense event columns are `[x, y, polarity, timestamp]`.

## Capture

```text
DAVIS.async_read()
  -> image + exposure interval + events
  -> robot observation
  -> policy inference and/or dataset writer
  -> video + event NPZ + timestamp JSONL
```

## Training

```text
LeRobotEventDataset
  -> event window and padding
  -> DataLoader batch
  -> image/event preprocessing
  -> EventEncoder + CrossModalFusion + VLAFlowMatching
  -> loss
```

## Online inference

```text
observation with exposure timestamps
  -> append new events to buffer
  -> apply checkpoint window
  -> event representation
  -> action chunk and action queue
```

## Offline inference

```text
dataset sample without exposure timestamps
  -> use the supplied event window
  -> predict_action_chunk()
```

Call `policy.reset()` at each episode boundary.
