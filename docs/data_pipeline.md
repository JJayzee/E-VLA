# E-VLA event-adapter data contract

This document defines the event-adapter dataset, policy, and deployment interface.
Third-party enhancement models and runtimes are not included. User-provided image
methods must preserve the same image and policy input contract.

Overlay dataset synthesis is implemented by `utils/make_overlay_image.py` and
`utils/make_overlay_video.py`. The external-image keys
`observation.images.E2VID`, `observation.images.Evlight`,
`observation.images.RetinexNet`, and `observation.images.Retinexformer` are
profile-switch options only; their method implementations are not included.

## Canonical features

| Feature | Representation |
| --- | --- |
| `observation.images.<camera>` | RGB `uint8` HWC at capture/storage; float TCHW or BTCHW in dataset batches |
| `observation.events.<camera>` | Structured NumPy events at capture/storage; dense Torch `int32` for the policy |
| `observation.timestamps.<camera>` | Exposure interval `[start_us,end_us]` |
| `observation.state` | Floating-point robot state |
| `action` | Floating-point robot action |
| `task` | Natural-language instruction |

The dense event column order is `[x, y, polarity, timestamp]`. Event and exposure
timestamps use microseconds; LeRobot frame timestamps use seconds.

Public conversion boundaries copy their inputs rather than modifying observation
dictionaries in place.

## Capture and storage

`InivationDavisCamera.async_read()` returns an RGB image, its exposure interval,
and structured events in `davis` mode. The event dtype contains
`timestamp:int64`, `x:int16`, `y:int16`, and `polarity:int8`.

```text
data/chunk-xxx/episode_xxxxxx.parquet
videos/chunk-xxx/observation.images.<camera>/episode_xxxxxx.mp4
events/chunk-xxx/observation.events.<camera>/episode_xxxxxx.npz
timestamps/chunk-xxx/observation.timestamps.<camera>/episode_xxxxxx.jsonl
```

Videos use the LeRobot image/video path. Event NPZ files contain one structured
array per `frame_xxxxxx`. Exposure timestamps are stored in JSONL sidecars with
`frame_index`, `start`, and `end` fields.

## Dataset samples

`LeRobotEventDataset` aligns the video, event NPZ, and exposure sidecar. It returns
events already sliced to the configured count or duration window.

| Value | Sample | Batch |
| --- | --- | --- |
| image | `(T,C,H,W)` | `(B,T,C,H,W)` |
| state | `(T,D_state)` | `(B,T,D_state)` |
| count-window events | `(event_window_size,4)` | `(B,event_window_size,4)` |
| duration-window events | `(20000,4)` | `(B,20000,4)` |
| action | `(chunk_size,D_action)` | `(B,chunk_size,D_action)` |

Unused event rows are filled with `-1`. Dataset samples omit exposure timestamps
because their events are already windowed.

All datasets in a multitask run must have compatible FPS and feature schemas.

The published dataset is [JJayzee/E-VLA-MultiLight](https://huggingface.co/datasets/JJayzee/E-VLA-MultiLight)
on Hugging Face. After download, its local root is `dataset_task_light_raw`.
Its dataset identifiers are:

```text
pap/200lux
pap/lowlight
sorting/200lux
sorting/lowlight
stack/200lux
stack/lowlight
```

Every leaf directory is a complete LeRobotDataset v2.1 dataset. Its canonical
modalities are `observation.images.DAVIS`, `observation.events.DAVIS`, and
`observation.timestamps.DAVIS`, with the paths shown in the capture-and-storage
section above.

## Policy boundary

Training performs image preparation, event representation, state/action padding,
and normalization inside `SmolVLAPolicy.forward()`. Event slicing remains a dataset
operation and is not repeated by the policy.

Online observations include exposure timestamps. `select_action()` appends new
events to the policy buffer, slices at the exposure end time, constructs the event
representation, and fills the action queue. `policy.reset()` clears both queues.

Offline observations omit exposure timestamps. Their event arrays are treated as
complete dataset windows.

## Event-window configuration

Exactly one of `event_window_size` and `event_window_duration` must be positive.
Training synchronizes the selected value between dataset and policy configuration.
Inference uses the checkpoint value.

Duration-window dataset samples are padded to 20,000 rows. Online raw-event
inference supports batch size 1.

## Metadata profiles

`utils/switch_video_version.py` stores profiles under
`meta/video_profiles/<profile>/`. Each profile contains `info.json`,
`episodes_stats.jsonl`, and `profile.json`.

The logical key `observation.images.DAVIS` may point to another physical video
directory through `info.video_path`. Event profiles include the event feature and
`event_path`; image profiles omit them from metadata. Physical modality files are
not changed.

Profile activation replaces each metadata file atomically, but the pair is not a
cross-file transaction. The active metadata is backed up before the first switch.

## Processing utilities

Utilities under `utils/` cover video decoding, white balance, frame validation,
event reslicing, overlay synthesis, dataset composition, and metadata profiles.
Dataset paths are provided on the command line. Compatible locally recorded
datasets can be combined into a new directory with:

```bash
python utils/merge_split_event_dataset.py merge OUTPUT_DIR INPUT_DIR [INPUT_DIR ...]
```

The utility validates metadata and sidecars, assigns continuous episode and frame
indices, and writes a complete LeRobotDataset v2.1 output. It also provides
`split` and `reindex` subcommands; run each subcommand with `--help` for its
arguments.

Before replacing a canonical video or sidecar directory:

1. run the corresponding validation function;
2. verify episode sets, frame counts, FPS, and feature schema;
3. keep a recoverable copy of the source dataset.
