#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import torch
import torch.utils
from PIL import Image

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    check_delta_timestamps,
    check_timestamps_sync,
    create_empty_dataset_info,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    validate_episode_buffer,
    validate_frame,
    write_json,
)
from lerobot.datasets.video_utils import (
    get_safe_default_codec,
)
from lerobot.policies.smolvla.event_repr import _events_to_image_by_accu_and_bayer

DEFAULT_EVENT_NPY_PATH = "event_npys/{event_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.npy"
DEFAULT_EVENT_PATH = "events/chunk-{episode_chunk:03d}/{event_key}/episode_{episode_index:06d}.npz"
EVENT_DTYPE = np.dtype(
    [("timestamp", "<i8"), ("x", "<i2"), ("y", "<i2"), ("polarity", "i1")]
)


def _canonicalize_stored_events(events: np.ndarray) -> np.ndarray:
    """Return the canonical structured on-disk event representation.

    Input may use named DAVIS fields or dense ``[x,y,polarity,timestamp]`` rows.
    """
    events = np.asarray(events)
    if events.dtype.names is not None:
        required_fields = {"timestamp", "x", "y", "polarity"}
        if not required_fields.issubset(events.dtype.names):
            raise ValueError(f"Structured events are missing fields: {required_fields - set(events.dtype.names)}")
        canonical = np.empty(events.shape, dtype=EVENT_DTYPE)
        for field in ("timestamp", "x", "y", "polarity"):
            canonical[field] = events[field]
        return canonical
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Dense events must have shape (N, 4), got {events.shape}.")
    canonical = np.empty(events.shape[0], dtype=EVENT_DTYPE)
    canonical["x"] = events[:, 0]
    canonical["y"] = events[:, 1]
    canonical["polarity"] = events[:, 2]
    canonical["timestamp"] = events[:, 3]
    return canonical


CODEBASE_VERSION = "v2.1"

class LeRobotEventDatasetMetadata(LeRobotDatasetMetadata):

    def __init__(self, *args, **kwargs):
        # same to parent
        super().__init__(*args, **kwargs)

    @property
    def event_path(self) -> str | None:
        """Formattable string for the event files."""
        return self.info["event_path"]

    @property
    def event_keys(self) -> list[str]:
        """Keys to access event data."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "event"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image", "event"]]

    def get_event_file_path(self, ep_index: int, event_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.event_path.format(episode_chunk=ep_chunk, event_key=event_key, episode_index=ep_index)
        return Path(fpath)

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotEventDatasetMetadata":
        """Creates metadata for a LeRobotEventDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        # TODO(aliberts, rcadene): implement sanity check for features
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        obj.info["event_path"] = DEFAULT_EVENT_PATH

        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotEventDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        event_window_size: int | None = None,
        event_window_duration: int | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        ├── events
        │   ├── chunk-000
        │   │   └── observation.events.DAVIS
        |   |       ├── episode_000000.npz
        |   |       ├── episode_000001.npz
        |   |       ├── episode_000002.npz
        |   |       └── ...
        │   └── ...
        └── videos
            ├── chunk-000
            │   ├── observation.images.DAVIS
            |   |   ├── episode_000000.mp4
            |   |   ├── episode_000001.mp4
            |   |   ├── episode_000002.mp4
            |   |   └── ...
            │   ├── observation.images.Realsense
            |   |   ├── episode_000000.mp4
            |   |   ├── episode_000001.mp4
            |   |   ├── episode_000002.mp4
            |   |   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
        """
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.event_window_size = event_window_size
        self.event_window_duration = event_window_duration
        if self.event_window_size is not None and self.event_window_duration is not None:
            raise ValueError("event_window_size and event_window_duration are mutually exclusive.")
        if self.event_window_size is not None and self.event_window_size <= 0:
            raise ValueError("event_window_size must be a positive integer.")
        if self.event_window_duration is not None and self.event_window_duration <= 0:
            raise ValueError("event_window_duration must be a positive duration in microseconds.")
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotEventDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(self, *args, **kwargs):
        super().push_to_hub(*args, **kwargs)

    def download_episodes(self, *args, **kwargs):
        super().download_episodes(*args, **kwargs)

    def get_episodes_file_paths(self) -> list[Path]:
        fpaths = super().get_episodes_file_paths()
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        if len(self.meta.event_keys) > 0:
            event_files = [
                str(self.meta.get_event_file_path(ep_idx, ev_key))
                for ev_key in self.meta.event_keys
                for ep_idx in episodes
            ]
            fpaths += event_files
        return fpaths


    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys and key not in self.meta.event_keys
        }

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        feature_type: str | None = "video",
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        if feature_type == "video":
            keys = self.meta.video_keys
        elif feature_type == "event":
            keys = self.meta.event_keys
        else:
            raise ValueError(f"Invalid feature type: {feature_type}, must be 'video' or 'event'")
        
        for key in keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        
        return query_timestamps

    def _query_timestamps(self, episode_index: int, frame_index: int) -> tuple[int, int]:
        """
        Args:
            episode_index: int, the episode index
            frame_index: int, the frame index in the episode
        Returns:
            tuple[int, int], the start and end timestamp(in us) of the queried frame
        """
        timestamp_key = "observation.timestamps.DAVIS"
        timestamp_path = self.root / "timestamps" / "chunk-000" / f"{timestamp_key}" / f"episode_{episode_index:06d}.jsonl"
        
        if not timestamp_path.is_file():
            raise FileNotFoundError(f"Timestamp file not found: {timestamp_path}")
        
        with open(timestamp_path, "r") as f:
            for line in f:
                timestamp_data = json.loads(line)
                if timestamp_data["frame_index"] == frame_index:
                    return timestamp_data["start"], timestamp_data["end"]
            raise ValueError(f"Frame index {frame_index} not found in timestamp file: {timestamp_path}")

    def _query_events(self, query_indices: dict[str, int], ep_idx: int, image_start_timestamp: int, image_end_timestamp: int) -> dict[str, torch.Tensor]:
        """  
        Args:
            query_indices: dict, key is event_key, value is the frame index of the queryed frame in the episode
            ep_idx: episode index
            image_start_timestamp: int, the exposure start time stamp, us
            image_end_timestamp: int, the exposure end time stamp, us
            
        Returns:
            A dict mapping each event key to an int32 tensor with dense columns
            ``[x, y, polarity, timestamp]``. Count windows have shape ``(event_window_size, 4)``;
            duration windows currently have shape ``(20000, 4)``. Unused rows are padded with ``-1``.
        Handles duplicate previous-frame events, empty event windows, and invalid
        timestamp ranges. Count windows may cross the first frame of an episode
        or contain more events than the requested window size before truncation.
      """

        item = {}

        if self.event_window_size is None and self.event_window_duration is None:
            raise ValueError("event_window_size or event_window_duration must be set when getitem from LeRobotEventDataset.")
        if self.event_window_size is not None and self.event_window_duration is not None:
            raise ValueError("event_window_size and event_window_duration cannot be set at the same time.")
        
        for ev_key, f_idx in query_indices.items():
            event_path = self.root / self.meta.get_event_file_path(ep_idx, ev_key)
            
            if not event_path.is_file():
                raise FileNotFoundError(f"Event file not found: {event_path}")
            
            if self.event_window_size is not None:
                # Number window event sample
                with np.load(event_path) as event_data:
                    collected_events = np.empty((0, 4), dtype=np.int32)
                    eldest_timestamp = 1e10
                    while len(collected_events) < self.event_window_size and f_idx >= 0:
                        frame_key = f"frame_{f_idx:06d}"
                        if frame_key in event_data:
                            frame_events = _canonicalize_stored_events(event_data[frame_key])
                            new_events_mask = frame_events['timestamp'] < eldest_timestamp
                            new_frame_events = frame_events[new_events_mask]
                            # select the events based on image end timestamp
                            new_frame_events = new_frame_events[new_frame_events['timestamp'] <= image_end_timestamp]
                            if len(new_frame_events) > 0:
                                new_frame_events = np.column_stack([
                                    new_frame_events['x'],
                                    new_frame_events['y'],
                                    new_frame_events['polarity'],
                                    new_frame_events['timestamp']
                                ]).astype(np.int32)
                                collected_events = np.concatenate([new_frame_events, collected_events], axis=0)
                                eldest_timestamp = collected_events[0, 3]
                            f_idx -= 1
                        else:
                            raise ValueError(f"Error in _query_events: {frame_key} is not in {event_path}.")

                    if len(collected_events) == 0:
                        all_events = np.zeros((0, 4), dtype=np.int32)
                    else:
                        all_events = collected_events
            
            elif self.event_window_duration is not None:
                # Duration window event sample
                with np.load(event_path) as event_data:
                    collected_events = np.empty((0, 4), dtype=np.int32)
                    duration = 0
                    eldest_timestamp = 1e10
                    while duration < self.event_window_duration and f_idx >= 0:
                        frame_key = f"frame_{f_idx:06d}"
                        if frame_key in event_data:
                            frame_events = _canonicalize_stored_events(event_data[frame_key])
                            if len(frame_events) > 0:
                                duration = image_end_timestamp - frame_events['timestamp'][0]
                            else:
                                duration = duration + 33333 # 33.33ms
                            new_events_mask = frame_events['timestamp'] < eldest_timestamp
                            new_frame_events = frame_events[new_events_mask]
                            # select the events based on image end timestamp
                            duration_mask = (new_frame_events['timestamp'] <= image_end_timestamp) & (new_frame_events['timestamp'] >= image_end_timestamp-self.event_window_duration)
                            new_frame_events = new_frame_events[duration_mask]
                            if len(new_frame_events) > 0:
                                new_frame_events = np.column_stack([
                                    new_frame_events['x'],
                                    new_frame_events['y'],
                                    new_frame_events['polarity'],
                                    new_frame_events['timestamp']
                                ]).astype(np.int32)
                                collected_events = np.concatenate([new_frame_events, collected_events], axis=0)
                                eldest_timestamp = collected_events[0, 3]
                            f_idx -= 1
                        else:
                            raise ValueError(f"Error in _query_events: {frame_key} is not in {event_path}.")

                    if len(collected_events) == 0:
                        all_events = np.zeros((0, 4), dtype=np.int32)
                    else:
                        all_events = collected_events
            
            # Ensure fixed shape (padding_length, 4) for consistent batching
            # Pad with -1 if events are fewer than padding_length
            # Use -1 to avoid confusion with real events at (0,0)
            padding_length = self.event_window_size if self.event_window_size is not None else 20000
            if len(all_events) >= padding_length:
                all_events = all_events[-padding_length:]
            else:
                padding = np.full((padding_length - len(all_events), 4), -1, dtype=np.int32)
                all_events = np.concatenate([all_events, padding], axis=0)
            
            events_tensor = torch.from_numpy(all_events).to(torch.int32)

            item[ev_key] = events_tensor

        return item

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices, feature_type="video")
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if len(self.meta.event_keys) > 0:
            frame_idx = item["frame_index"].item()

            image_start_timestamp, image_end_timestamp = self._query_timestamps(ep_idx, frame_idx)

            event_frames = self._query_events({"observation.events.DAVIS": frame_idx}, ep_idx, image_start_timestamp, image_end_timestamp)
            item = {**event_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.video_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]


        # Keep dataset samples side-effect free; rendering belongs outside the
        # dataset loading path.
        # events = events[valid_mask]
        # ev_img = _events_to_image_by_accu_and_bayer(events, "count")
        # ev_img = (((ev_img + 1.0) / 2.0) * 255).numpy().astype(np.uint8)
        # ev_img = Image.fromarray(np.transpose(ev_img, (1, 2, 0)))
        # ev_img.save(save_dir / "events.png")
        # after check, it seems not problem with events and images


        return item  # Events are already windowed and padded; policy.forward must not window them again.

    def _get_event_file_path(self, episode_index: int, event_key: str, frame_index: int) -> Path:
        # used in `add_frame` to save events temporarily in disk
        # not the final path of the event file.
        fpath = DEFAULT_EVENT_NPY_PATH.format(
            event_key=event_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def add_frame(self, frame: dict, task:str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images and events — which are
        written in a temporary directory — nothing is written to disk. To save those frames, the 'save_episode()'
        method then needs to be called.
        """
        # Work on a shallow copy: callers may reuse the native robot observation for display/inference.
        frame = dict(frame)

        # Exposure timestamps are sidecar metadata rather than a dataset feature.
        frame_timestamps = frame.pop("observation.timestamps.DAVIS", None)
        if isinstance(frame_timestamps, torch.Tensor):
            frame_timestamps = frame_timestamps.detach().cpu().reshape(-1).tolist()
        elif isinstance(frame_timestamps, np.ndarray):
            frame_timestamps = frame_timestamps.reshape(-1).tolist()
        if frame_timestamps is not None:
            if len(frame_timestamps) != 2:
                raise ValueError("DAVIS exposure timestamps must contain exactly [start, end].")
            frame_start, frame_end = frame_timestamps
        else:
            frame_start, frame_end = None, None

        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].detach().cpu().numpy()
            if name in self.features and self.features[name]["dtype"] == "event":
                frame[name] = _canonicalize_stored_events(frame[name])

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        
        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]   # is a counter of frames in the episode
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # temporary add frame timestamps for DAVIS camera
        if frame_start is not None and frame_end is not None:
            self.episode_buffer["frame_start_ts"].append(int(frame_start))
            self.episode_buffer["frame_end_ts"].append(int(frame_end))

        # Add frame features to episode_buffer
        for key in frame:
            # frame: dict with keys:
            # - observation.state: the state of the robot
            # - observation.images: the images from the cameras: numpy array(uint8, HWC)
            # - observation.events: structured camera array or dense NumPy (N, 4) event array
            # - action: the action to be sent to the robot
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            elif self.features[key]["dtype"] == "event":
                ev_path = self._get_event_file_path(
                    episode_index=self.episode_buffer["episode_index"], event_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    ev_path.parent.mkdir(parents=True, exist_ok=True)
                # still using the same writer to save events as .npy files
                self._save_image(frame[key], ev_path)
                self.episode_buffer[key].append(str(ev_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.
        
        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer
        
        # temporary get frame timestamps for DAVIS camera
        frame_start_ts = episode_buffer.pop("frame_start_ts", None)
        frame_end_ts = episode_buffer.pop("frame_end_ts", None)
        if frame_start_ts is not None and frame_end_ts is not None:
            timestamp_dir = self.root / "timestamps" / "chunk-000" / "observation.timestamps.DAVIS"
            timestamp_dir.mkdir(parents=True, exist_ok=True)
            file_path = timestamp_dir / f"episode_{episode_buffer['episode_index']:06d}.jsonl"
            with open(file_path, "w") as f:
                for idx in range(len(frame_start_ts)):
                    timestamp_line = {}
                    timestamp_line["frame_index"] = idx
                    timestamp_line["start"] = int(frame_start_ts[idx])
                    timestamp_line["end"] = int(frame_end_ts[idx])
                    f.write(json.dumps(timestamp_line) + "\n")

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video", "event"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()   # waiting for all images and events be written to temp disk
        self._save_episode_table(episode_buffer, episode_index) # save the episode table to parquet(not include images if used videos)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        has_video_keys = len(self.meta.video_keys) > 0
        has_event_keys = len(self.meta.event_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if not use_batched_encoding:
            if has_video_keys:
                self.encode_episode_videos(episode_index)
            if has_event_keys:
                self.encode_episode_events(episode_index)

        # `meta.save_episode` should be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # Check if we should trigger batch encoding
        if use_batched_encoding:
            if not has_video_keys and not has_event_keys:
                raise ValueError("No video or event keys to batch encode.")
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                if has_video_keys:
                    logging.info(
                        f"Batch encoding {self.batch_encoding_size} videos for episodes {start_ep} to {end_ep - 1}"
                    )
                    self.batch_encode_videos(start_ep, end_ep)
                if has_event_keys:
                    logging.info(
                        f"Batch encoding {self.batch_encoding_size} events for episodes {start_ep} to {end_ep - 1}"
                    )
                    self.batch_encode_events(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        # Episode data index and timestamp checking
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # Verify that we have one parquet file per episode and the number of video files matches the number of encoded episodes
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == (self.num_episodes - self.episodes_since_last_encoding) * len(
            self.meta.video_keys
        )
        event_files = list(self.root.rglob("*.npz"))
        assert len(event_files) == (self.num_episodes - self.episodes_since_last_encoding) * len(
            self.meta.event_keys
        )

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]

        # Clean up image and event files for the current episode buffer
        if self.image_writer is not None:
            for video_key in self.meta.video_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=video_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
            for event_key in self.meta.event_keys:
                event_dir = self._get_event_file_path(
                    episode_index=episode_index, event_key=event_key, frame_index=0
                ).parent
                if event_dir.is_dir():
                    shutil.rmtree(event_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def encode_episode_events(self, episode_index: int) -> None:
        """
        Use numpy to package all temporary event .npy files into a single .npz file.
        """
        for key in self.meta.event_keys:
            event_path = self.root / self.meta.get_event_file_path(episode_index, key)
            event_dir = event_path.parent
            event_dir.mkdir(parents=True, exist_ok=True)
            if event_path.is_file():
                # Skip if event is already encoded. Could be the case when resuming data recording.
                continue

            event_frames_to_save = {}
            event_npy_dir = self._get_event_file_path(
                episode_index=episode_index, event_key=key, frame_index=0
            ).parent
            event_npy_paths = sorted(event_npy_dir.glob("*.npy"))
            for i, npy_path in enumerate(event_npy_paths):
                event_frames_to_save[f"frame_{i:06d}"] = np.load(npy_path)
            np.savez_compressed(event_path, **event_frames_to_save)
            shutil.rmtree(event_npy_dir)
        
    def batch_encode_events(self, start_episode: int = 0, end_episode: int | None = None) -> None:
        """
        Batch encode events for multiple episodes.
        """
        raise NotImplementedError("Batch encoding events is not implemented for LeRobotEventDataset.")

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> "LeRobotEventDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotEventDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            robot_type=robot_type,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotEventDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotEventDataset`s.

    The underlying `LeRobotEventDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotEventDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        event_window_size: int | None = None,
        event_window_duration: int | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotEventDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                event_window_size=event_window_size,
                event_window_duration=event_window_duration,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
    
    class Proxy_MetaData:
        def __init__(self, features, camera_keys, stats):
            self.features = features
            self.stats = stats
            self.camera_keys = camera_keys
            self.event_keys = [key for key, ft in features.items() if ft["dtype"] == "event"]
            self.video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]

    @property
    def meta(self):
        return MultiLeRobotEventDataset.Proxy_MetaData(self.features, self.camera_keys, self.stats)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return True if self._datasets[0].meta.video_keys else False

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if feats["dtype"] in ["image", "video", "event"]:
                keys.append(key)
        return keys
    
    @property
    def video_keys(self) -> list[str]:
        """Keys to access video frames."""
        video_keys = []
        for key, feats in self.features.items():
            if feats["dtype"] == "video":
                video_keys.append(key)
        return video_keys

    @property
    def event_keys(self) -> list[str]:
        """Keys to access event data."""
        keys = []
        for key, feats in self.features.items():
            if feats["dtype"] == "event":
                keys.append(key)
        return keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    @property
    def episode_data_index(self) -> dict[str, torch.Tensor]:
        """Concatenate per-dataset episode indices into global indices for samplers."""
        from_indices = []
        to_indices = []
        offset = 0
        for ds in self._datasets:
            ep_idx = ds.episode_data_index
            from_indices.append(ep_idx["from"] + offset)
            to_indices.append(ep_idx["to"] + offset)
            offset += ds.num_frames
        return {
            "from": torch.cat(from_indices) if len(from_indices) > 0 else torch.LongTensor([]),
            "to": torch.cat(to_indices) if len(to_indices) > 0 else torch.LongTensor([]),
        }

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("Can't find the dataset index for the given index.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(self.repo_id_to_index[self._datasets[dataset_idx].repo_id])
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Features: {self.features.keys()},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  FPS: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Event Keys: {self.event_keys},\n"
            f"  Video Frame Keys: {self.video_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
