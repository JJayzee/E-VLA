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

import logging

from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_event_dataset import LeRobotEventDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | LeRobotEventDatasetMetadata | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    Policy dimensions and normalization statistics are derived from dataset metadata.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.

    Raises:
        ValueError: Dataset metadata is required.
    Returns:
        PreTrainedPolicy: _description_
    """
    if ds_meta is None:
        raise ValueError("Dataset metadata is required to construct an evla policy.")

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)
    # if train from huggingface/lerobot/smolvla, must use dataset_stats
    # train from ckpt on the same datasets, use or not is the same
    # validate on robot, not need to use
    norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    have_norm_keys_in_config = all(x in cfg.load_ignore_keys for x in norm_keys)
    if have_norm_keys_in_config and ds_meta.stats != {}:
        logging.info("Use dataset stats to init norm layers")
        kwargs["dataset_stats"] = ds_meta.stats
    elif have_norm_keys_in_config and ds_meta.stats == {}:
        raise ValueError("Load dataset stats, but dataset stats is empty.")
    else:
        logging.info("Not use dataset stats to init norm layers")

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        logging.info(f"Load pretrained model from {cfg.pretrained_path}")
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        logging.info("Make a fresh policy")
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
