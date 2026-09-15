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

"""
This script will help you convert any local LeRobot dataset from codebase version 2.0 to
2.1. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.

Usage:

```bash
python convert_local_dataset_v20_to_v21.py \
    --local-path=/path/to/your/dataset \
    --repo-id=your_dataset_name
```

"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def convert_local_dataset(
    local_path: str,
    repo_id: str,
    num_workers: int = 4,
    skip_data_validation: bool = False,
):
    """
    Convert a local LeRobot dataset from v2.0 to v2.1 format.
    
    Args:
        local_path: Path to your local dataset directory
        repo_id: A name for your dataset (used internally by LeRobotDataset)
        num_workers: Number of workers for parallelizing stats compute
        skip_data_validation: Skip loading parquet data (useful when LFS files aren't downloaded)
    """
    dataset_root = Path(local_path)
    
    # Verify that this is a v2.0 dataset
    info_file = dataset_root / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Dataset info file not found at {info_file}")
    
    if skip_data_validation:
        # Only update metadata, don't process data files
        from lerobot.datasets.utils import load_info, write_info
        
        print("Skip data validation mode - only updating metadata")
        info = load_info(dataset_root)
        info["codebase_version"] = CODEBASE_VERSION
        write_info(info, dataset_root)
        
        # Remove old stats.json
        stats_file = dataset_root / STATS_PATH
        if stats_file.exists():
            stats_file.unlink()
            print(f"Removed deprecated {STATS_PATH}")
        
        print(f"Successfully updated dataset metadata from {V20} to {V21}")
        print("Note: Episode stats were not computed due to skip_data_validation=True")
        print("You may need to recompute stats later when LFS files are available.")
        return
    
    with SuppressWarnings():
        # Load the dataset from local path with pyav video backend
        dataset = LeRobotDataset(
            repo_id, 
            root=dataset_root, 
            revision=V20,
            video_backend="pyav"
        )

    # Remove existing episodes_stats.jsonl if it exists
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    # Convert stats from v2.0 to v2.1 format
    convert_stats(dataset, num_workers=num_workers)
    
    # Load reference stats and check consistency
    ref_stats = load_stats(dataset.root)
    if ref_stats is not None:
        print("Checking consistency with reference stats...")
        check_aggregate_stats(dataset, ref_stats)
        print("Stats consistency check passed!")
    else:
        print("No reference stats found - skipping consistency check")

    # Update codebase version in info.json
    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    # Remove old stats.json file
    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()
        print(f"Removed deprecated {STATS_PATH}")

    print(f"Successfully converted local dataset at {local_path} from {V20} to {V21}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-path",
        type=str,
        required=True,
        help="Path to your local dataset directory (should contain 'meta', 'data', and optionally 'videos' folders).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="A name for your dataset (used internally by LeRobotDataset).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="Skip loading parquet data (useful when LFS files aren't downloaded)",
    )

    args = parser.parse_args()
    convert_local_dataset(**vars(args))