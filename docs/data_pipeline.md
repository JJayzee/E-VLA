# Image-only data contract

This document defines the dataset and policy interface for
`image_only`. Event inputs are handled on `event_adapter`.

## Features

| Feature | Representation |
| --- | --- |
| `observation.images.<camera>` | RGB `uint8` HWC at camera/storage boundaries; float BCHW or BTCHW at the policy boundary |
| `observation.state` | Floating-point robot state, normally 6-D for SO-100 |
| `action` | Floating-point robot action, normally 6-D for SO-100 |
| `task` | Natural-language instruction |

Image-only `input_features` must not contain `observation.events.*` or
`observation.timestamps.*`.

`dataset_to_policy_features()` excludes event data by dtype and canonical key.
Delta-timestamp resolution does not query event sidecars. Robot observations
expose only images and state to the image policy. These boundaries do not mutate
stored observations.

## Multiple datasets

`dataset.repo_id` accepts a whitespace-separated list. Component datasets must
use the same FPS and compatible feature schemas. Their common image keys determine
the policy inputs.

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

Every leaf directory is a complete LeRobotDataset v2.1 dataset with `data/`,
`videos/`, `events/`, `timestamps/`, and `meta/`. The canonical image key is
`observation.images.DAVIS`; event and exposure-timestamp sidecars use
`observation.events.DAVIS` and `observation.timestamps.DAVIS`. Image-only loading
selects the image-policy features without changing the stored dataset.

Verify `meta/info.json`, episode files, video paths, frame counts, FPS, image keys,
and state/action dimensions before training.

Compatible locally recorded datasets can be combined into a new directory with:

```bash
python utils/merge_split_event_dataset.py merge OUTPUT_DIR INPUT_DIR [INPUT_DIR ...]
```

The utility validates metadata and sidecars, assigns continuous episode and frame
indices, and writes a complete LeRobotDataset v2.1 output. It also provides
`split` and `reindex` subcommands; run each subcommand with `--help` for its
arguments.

## Offline enhancement

Enhancement models and runtimes are not part of evla. User-provided enhanced
videos must match the training and deployment contract for:

- metadata key and video path;
- RGB/BGR order, dtype, and numeric range;
- spatial size, resize, and padding;
- frame order, FPS, and any temporal state;
- checkpoint input features.

Enhancement does not run inside the image-only policy. A deployment runtime, when
needed, must apply the same transformation used to create the training videos.

The declared external-image keys are `observation.images.E2VID`,
`observation.images.Evlight`, `observation.images.RetinexNet`, and
`observation.images.Retinexformer`. They are profile-switch options only; evla does
not implement these methods.

## Metadata profiles

A dataset may contain image, event, and timestamp files for both branches.
Prepared profiles are stored under `meta/video_profiles/<profile>/` and contain
`info.json`, `episodes_stats.jsonl`, and `profile.json`.

The logical image key remains `observation.images.DAVIS` while
`info.video_path` selects the physical video directory. An event profile includes
the event feature and `event_path`; an image profile omits them from metadata.
Physical files are not changed.

Profile activation stages complete files and replaces each destination
atomically. The two-file metadata update is not transactional. The first switch
saves the active metadata under `meta/video_profiles/_active_backup/`.
