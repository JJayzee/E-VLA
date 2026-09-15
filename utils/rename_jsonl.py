import argparse
import os
from pathlib import Path


def json2jsonl(dataset_root, repo_id):
    # find all json files in the directory and rename them to jsonl
    print(f"Renaming json files to jsonl files in {repo_id}")
    timestamp_dir = Path(dataset_root, repo_id, "timestamps", "chunk-000", "observation.timestamps.DAVIS")
    for file in os.listdir(timestamp_dir):
        if file.endswith(".json"):
            source = timestamp_dir / file
            destination = timestamp_dir / file.replace(".json", ".jsonl")
            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")
            source.rename(destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("repo_id")
    args = parser.parse_args()
    json2jsonl(args.dataset_root, args.repo_id)

if __name__ == "__main__":
    main()
