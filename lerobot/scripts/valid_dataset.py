#!/usr/bin/env python


"""Validate a trained SmolVLA model checkpoint on real datasets.

Offline dataset validation checks the inference pipeline and provides a diagnostic for
training convergence. Action MSE and the reported RMSE are not meaningful evaluation
metrics; the RMSE-threshold "success rate" is not robot task success. The dataset
format should be the same as used during training.

Usage examples:

Validate a model checkpoint on a real dataset:
```
python lerobot/scripts/valid_dataset.py \
    --dataset.repo_id=pap/200lux \
    --dataset.root=dataset_task_light_raw/pap/200lux \
    --dataset.video_backend=pyav \
    --policy.path=/path/to/your/pap-200lux/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --policy.use_amp=false \
    --output_dir=outputs/validation_image_only \
    --job_name=validation_image_only \
    --num_workers=4 \
    --batch_size=16 \
    --action_success_threshold=2 \
    --save_predictions=true \
```

"""
import os

# Set tokenizer parallelism before importing any transformers/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any, Dict

import numpy as np
import torch
from termcolor import colored
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.valid import ValidatePipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)


def compute_action_rmse(predicted_actions: torch.Tensor, true_actions: torch.Tensor) -> float:
    """Compute Root Mean Squared Error between predicted and true actions."""
    if predicted_actions.shape != true_actions.shape:
        raise ValueError(f"Shape don't match: predicted_actions.shape: {predicted_actions.shape}, true_actions.shape: {true_actions.shape}")
    return torch.sqrt(torch.mean((predicted_actions - true_actions) ** 2, dim=1)).cpu().numpy()

def validate_on_dataset(
    dataset: Any,
    policy: PreTrainedPolicy,
    cfg: ValidatePipelineConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a diagnostic dataset pass; RMSE and threshold pass rate are not task metrics.
    
    Args:
        dataset: The dataset to validate on
        policy: The trained policy to evaluate
        cfg: Validation configuration
        device: Device to run inference on
        
    Returns:
        Dictionary containing validation metrics and results
    """
    logging.info("Starting dataset validation")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # Don't shuffle for consistent validation
        num_workers=cfg.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    
    policy.eval()
    
    # Metrics tracking
    validation_metrics = {
        "action_rmse": AverageMeter("action_rmse", ":.2f"),                   # average_mes on all frames
        "action_success_rate": AverageMeter("action_success", ":.2f"),      # success_frames / total_frames
        "inference_time_ms": AverageMeter("inf_time_ms", ":.2f"),           # average inference time on all batches
    }
    
    all_predictions = []
    all_ground_truth = []
    all_rmse = []
    
    total_batches = len(dataloader)
    successful_predictions = 0
    total_predictions = 0
    
    logging.info(f"Running validation on {total_batches} batches")
    
    with torch.no_grad():
        progress_bar = trange(total_batches, desc="Validating on dataset", leave=False)
        for batch_idx, batch in enumerate(dataloader):
            progress_bar.update(1)
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Extract ground truth actions
            true_actions = batch["action"].clone()      # (batch_size, chunk_size, action_dim)
            
            # Remove actions from batch for inference
            inference_batch = {k: v for k, v in batch.items() if k != "action"}
            
            # Time the inference
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            
            # Run inference
            with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                predicted_chunk = policy.predict_action_chunk(inference_batch)
                predicted_actions = predicted_chunk[:, 0, :]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            
            inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Ensure predictions and ground truth have the same shape
            if true_actions.ndim == 3:
                true_actions = true_actions[:, 0, :]
            if predicted_actions.shape != true_actions.shape:
                raise ValueError(f"Shape don't match: predicted_actions.shape: {predicted_actions.shape}, true_actions.shape: {true_actions.shape}")
            
            # Compute metrics for this batch
            batch_rmse = compute_action_rmse(predicted_actions, true_actions) # rmse over action_dims, (batch,)
            # cal how many frames are success in this batch
            batch_successes = int((batch_rmse < cfg.action_success_threshold).sum())

            # Update metrics
            validation_metrics["action_rmse"].update(batch_rmse.mean(), predicted_actions.shape[0])
            validation_metrics["action_success_rate"].update(
                batch_successes / predicted_actions.shape[0] * 100, 
                predicted_actions.shape[0]
            )
            validation_metrics["inference_time_ms"].update(inference_time, 1)
            
            successful_predictions += batch_successes
            total_predictions += predicted_actions.shape[0]
            
            # Output batch statistics to console
            print(f"Batch {batch_idx}: Avg RMSE = {batch_rmse.mean():.2f}, Success Rate = {batch_successes} / {predicted_actions.shape[0]}")
            
            # Store predictions if requested
            if cfg.save_predictions:
                all_predictions.append(predicted_actions.cpu().numpy())      # predicted_actions: (batch_size, action_dim)
                all_ground_truth.append(true_actions.cpu().numpy())          # true_actions: (batch_size, action_dim)
                all_rmse.append(batch_rmse)                    # rmse over action_dims, (batch,)
            
    # Compute final metrics
    if total_predictions == 0:
        raise ValueError("Cannot validate an empty dataset.")
    overall_success_rate = (successful_predictions / total_predictions) * 100
    
    results = {
        "aggregated": {
            "total_samples": int(total_predictions),
            "successful_predictions": int(successful_predictions),
            "overall_success_rate": float(overall_success_rate),
            "avg_action_rmse": float(validation_metrics["action_rmse"].avg),
            "avg_inference_time_ms": float(validation_metrics["inference_time_ms"].avg),
            "action_success_threshold": float(cfg.action_success_threshold),
        },
        "metrics": {k: float(v.avg) for k, v in validation_metrics.items()},
    }
    
    if cfg.save_predictions:
        results["predictions"] = {
            "predicted_actions": np.concatenate(all_predictions, axis=0),
            "ground_truth_actions": np.concatenate(all_ground_truth, axis=0),
            "rmse": np.concatenate(all_rmse, axis=0),
        }
    
    logging.info(f"Validation completed. Success rate: {overall_success_rate:.2f}%")
    
    return results


@parser.wrap()
def validate(cfg: ValidatePipelineConfig):
    """Main validation function."""
    logging.info(pformat(cfg.__dict__))
    
    # Set seed for reproducibility
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Check device availability
    device = get_safe_torch_device(cfg.policy.device, log=True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logging.info("Loading dataset")
    
    # Create a minimal TrainPipelineConfig for dataset loading
    # method 'make_dataset' only use TrainPipelineConfig.dataset and TrainPipelineConfig.policy,
    # so other default settings (ie. batch_size, num_workers, etc.) in TrainPipelineConfig will be ignored.
    train_cfg = TrainPipelineConfig(
        dataset=cfg.dataset,
        policy=cfg.policy
    )
    dataset = make_dataset(train_cfg)
    
    logging.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    
    # Load policy
    logging.info("Loading policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    policy.to(device)
    policy.eval()
    
    logging.info(f"Policy loaded: {type(policy).__name__}")
    
    # Run validation
    start_time = time.time()
    
    with torch.no_grad():
        validation_results = validate_on_dataset(dataset, policy, cfg, device)
    
    validation_results["aggregated"]["total_validation_time_s"] = time.time() - start_time
    
    # Print results
    logging.info("\n" + "="*50)
    logging.info("VALIDATION RESULTS")
    logging.info("="*50)
    logging.info(f"Total samples: {validation_results['aggregated']['total_samples']}")
    logging.info(f"Successful predictions: {validation_results['aggregated']['successful_predictions']}")
    logging.info(f"Overall success rate: {validation_results['aggregated']['overall_success_rate']:.2f}%")
    logging.info(f"Average action RMSE: {validation_results['aggregated']['avg_action_rmse']:.2f}")
    logging.info(f"Action success threshold: {validation_results['aggregated']['action_success_threshold']}")
    logging.info(f"Average inference time: {validation_results['aggregated']['avg_inference_time_ms']:.2f}ms")
    logging.info(f"Total validation time: {validation_results['aggregated']['total_validation_time_s']:.2f}s")
    logging.info("="*50)
    
    # Save results
    results_file = cfg.output_dir / "validation_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = validation_results.copy()
    predictions_data = None
    if "predictions" in json_results:
        predictions_data = json_results.pop("predictions")
    
    # print all keys and its dtypes in json_results recursively
    def print_dtypes(obj, indent=0):
        if isinstance(obj, dict):
            for key, value in obj.items():
                logging.info(f"{' ' * indent}{key}: {type(value)}")
                print_dtypes(value, indent + 2)
        elif isinstance(obj, list):
            for item in obj:
                print_dtypes(item, indent + 2)
    
    # print_dtypes(json_results)
    
    json_results = convert_for_json(json_results)
    
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    
    logging.info(f"Validation results saved to: {results_file}")
    
    # Save detailed predictions if requested
    if cfg.save_predictions and predictions_data is not None:
        episode_lengthes = [dataset.meta.episodes[i]["length"] for i in range(len(dataset.meta.episodes))]
        episode_lengthes = np.insert(np.cumsum(episode_lengthes), 0, 0)
        episode_lengthes = episode_lengthes[:-1].tolist()

        predictions_file = cfg.output_dir / "predictions.txt"
        
        # Convert numpy arrays to integer lists
        predicted_actions = predictions_data["predicted_actions"].astype(int).tolist()
        ground_truth_actions = predictions_data["ground_truth_actions"].astype(int).tolist()
        rmses = predictions_data["rmse"].tolist()
        
        # Create table format with tab-separated values
        with open(predictions_file, "w") as f:
            # Write header
            f.write("frame idx\t\t\t\tpred\t\t\t\t\t\t\t\ttrue\t\t\t\t\trmse\n")
            # Write data rows
            episode_idx = 0
            episode_begin_idx = 0
            for idx, (pred, true, sample_rmse) in enumerate(
                zip(predicted_actions, ground_truth_actions, rmses, strict=True)
            ):
                pred_str = str(pred)
                true_str = str(true)
                if idx in episode_lengthes:
                    episode_begin_idx = idx
                    f.write(f"\nEpisode {episode_idx}\n")
                    episode_idx += 1
                f.write(
                    f"{idx - episode_begin_idx}   \t\t{pred_str}   \t\t{true_str}   \t\t{sample_rmse:.2f}\n"
                )
        
        logging.info(f"Predictions saved to: {predictions_file}")
    
    logging.info("Validation completed successfully")


if __name__ == "__main__":
    init_logging()
    validate()
