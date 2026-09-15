# E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

**Accepted to ECCV 2026.** [Paper](https://arxiv.org/abs/2604.04834) |
[Project repository](https://github.com/JJayzee/E-VLA)

This `image_only` branch of E-VLA is a research fork of
[Hugging Face LeRobot](https://github.com/huggingface/lerobot) for low-light
SO-100 manipulation. This branch provides the image-only SmolVLA workflow:

```text
RGB or offline-enhanced RGB
  -> multitask training
  -> dataset validation
  -> PyTorch deployment
```

Offline dataset validation only checks the inference pipeline and training
convergence. Action MSE (and the RMSE reported by the script) is not a
meaningful performance metric. The script's RMSE-threshold "success rate"
is not robot task success.

Event capture and event-adapter training are provided on `event_adapter`.
Third-party enhancement models and runtimes are not included. Compatible
enhancement methods may be implemented outside the policy.

The standard external-image options are `observation.images.E2VID`,
`observation.images.Evlight`, `observation.images.RetinexNet`, and
`observation.images.Retinexformer`. Their outputs can be selected with metadata
profiles; their implementations are not included. Overlay synthesis is provided on
`event_adapter`.

## Installation

evla targets Python 3.10.18. Install Python and FFmpeg first, then install the
tested PyTorch pair separately before installing the project dependencies.

```bash
conda env create -f environment.yml
conda activate evla
```

Install FFmpeg in the activated environment:

```bash
conda install -y ffmpeg=7.1.1 -c conda-forge
```

The tested CUDA 12.6 PyTorch pair is:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Use the same PyTorch and torchvision versions with the appropriate wheel index
when targeting CPU or another supported accelerator. Vendor builds such as
Jetson may require adjusting the version constraints in `pyproject.toml`. See the
[PyTorch installer](https://pytorch.org/get-started/locally/) and [LeRobot
installation guide](https://huggingface.co/docs/lerobot/en/installation).

```bash
python -m pip install -e ".[smolvla,feetech,davis]"
```

The `davis` extra installs the tested Python binding
`dv-processing==1.7.9`. Add the `intelrealsense` extra if needed. The
distribution name is `evla` and the Python package namespace is `lerobot`.

## Data contract

Policy inputs are limited to:

- `observation.images.<camera>`;
- `observation.state`;
- `task`.

The output is `action`. Images use RGB `uint8` HWC at camera and storage
boundaries, and floating-point BCHW or BTCHW at the policy boundary.

Image-only feature conversion excludes `observation.events.*` and
`observation.timestamps.*`, including when a shared dataset contains event
sidecars. Events are not treated as state or hidden policy inputs.

The DAVIS configuration uses `work_mode="davis"` to preserve synchronized camera
acquisition. Event payloads and exposure timestamps are removed at the robot
observation boundary before image-only recording or inference.

See [Data pipeline](data_pipeline.md) for the complete data contract.

## Dataset

The published dataset is available at [JJayzee/E-VLA-MultiLight](https://huggingface.co/datasets/JJayzee/E-VLA-MultiLight)
on Hugging Face. Download it separately from the code repository:

```bash
hf download JJayzee/E-VLA-MultiLight --type dataset --local-dir /path/to/data
```

The download retains `dataset_task_light_raw/` beneath the chosen local directory.
The dataset is divided only by task and illumination:

| Subset | Episodes | Frames |
| --- | ---: | ---: |
| `pap/200lux` | 130 | 44,500 |
| `pap/lowlight` | 180 | 63,232 |
| `sorting/200lux` | 95 | 58,743 |
| `sorting/lowlight` | 149 | 98,142 |
| `stack/200lux` | 80 | 34,592 |
| `stack/lowlight` | 90 | 40,101 |

Each subset is an independent LeRobotDataset v2.1 dataset:

```text
dataset_task_light_raw/<task>/<illumination>/
├── data/chunk-000/episode_*.parquet
├── videos/chunk-000/observation.images.DAVIS/episode_*.mp4
├── events/chunk-000/observation.events.DAVIS/episode_*.npz
├── timestamps/chunk-000/observation.timestamps.DAVIS/episode_*.jsonl
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

The published files contain the original DAVIS RGB videos, events, exposure
timestamps, robot state, actions, and task annotations. Generated enhancement
and overlay videos are not part of the dataset. The image-only branch reads the
RGB, state, action, and task fields and ignores the event sidecars.

## Dataset video profiles

`utils/switch_video_version.py` maps the logical key
`observation.images.DAVIS` to a selected physical video directory by activating a
metadata profile. It does not modify video, event, or timestamp files.

Prepare profiles while event metadata is available:

```bash
DATASET_DIR=dataset_task_light_raw/pap/200lux

python utils/switch_video_version.py prepare "$DATASET_DIR" event_davis \
  observation.images.DAVIS --include-events
python utils/switch_video_version.py prepare "$DATASET_DIR" image_enhanced \
  observation.images.Enhanced
```

Activate or inspect a profile:

```bash
python utils/switch_video_version.py switch "$DATASET_DIR" image_enhanced
python utils/switch_video_version.py status "$DATASET_DIR"
python utils/switch_video_version.py switch "$DATASET_DIR" event_davis
python utils/switch_video_version.py restore "$DATASET_DIR"
```

Each metadata file is replaced atomically. Activating the `info.json` and
`episodes_stats.jsonl` pair is not a cross-file transaction.

## Commands

Edit dataset, device, and output paths at the top of each script.

| Command | Purpose |
| --- | --- |
| `lerobot/cmds/train_smvla_image.sh` | Multitask image training |
| `lerobot/cmds/valid_dataset.sh` | Offline dataset diagnostic |
| `lerobot/cmds/valid_robot.sh` | Online policy validation; moves the robot |
| `lerobot/cmds/record.sh` | Image-only demonstration recording |
| `lerobot/cmds/replay.sh` | Episode replay; moves the robot |
| `lerobot/cmds/teleoperate.sh` | SO-100 teleoperation; moves the robot |

Camera, motor, and dataset diagnostics are also available in `lerobot/cmds/`.

`utils/merge_split_event_dataset.py` combines compatible locally recorded
datasets into a new LeRobotDataset v2.1 directory while preserving RGB, event,
timestamp, and metadata sidecars. It also exposes explicit `split` and `reindex`
subcommands.

## Citation and license

If you use this project in research, please cite the E-VLA paper:

```bibtex
@inproceedings{zhai2026evla,
  author = {Zhai, Jiajun and Shi, Hao and Guo, Shangwei and Yang, Kailun and Wang, Kaiwei},
  title = {E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2026}
}
```

The source code is released at
[JJayzee/E-VLA](https://github.com/JJayzee/E-VLA). E-VLA includes code derived
from LeRobot. Cite the upstream project where appropriate. The source is
distributed under the Apache License 2.0; see
[LICENSE](../LICENSE) and [NOTICE](../NOTICE).
