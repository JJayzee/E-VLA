#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import logging
import math
import os
import re
import threading
import time
from collections import deque

import cv2
import numpy as np
import safetensors
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoProcessor

from lerobot.cameras.davis_utils import apply_white_balance
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import (
    Normalize,
    Unnormalize,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.utils import get_safe_dtype

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def canonicalise(k: str) -> str:
    """
    Remove dataset-variant markers like '.so100-blue_' or '.so100_' from a
    normalisation-buffer key.
    """
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: dict[str, torch.Tensor], ref_keys: set[str], *, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    • Re-keys `checkpoint ` so that every entry matches the *reference* key set.
    • If several variant keys collapse to the same canonical name we keep the
      first one and log the collision.
    • Returns the new dict + a list of entries that could not be matched.
    """
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:  # duplicate after collapsing
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    # if verbose:
    #     for canon, variants in collisions.items():
    #         logging.info(f"[standardise_state_dict] '{canon}'  ←  {variants}")
    #         for v in variants:
    #             logging.info(f"variants: {v}, value: {checkpoint[v]}")
    #     if unmatched:
    #         logging.info(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: dict, rename_str: str):
    """
    Renames keys in a checkpoint dictionary based on the given rename string.

    Args:
        checkpoint (dict): The checkpoint dictionary.
        rename_str (str): A string specifying key mappings in the format "old1//new1,old2//new2".

    Returns:
        dict: The modified checkpoint with renamed keys.
    """

    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def load_smolvla(
    model: torch.nn.Module,
    filename: str | os.PathLike,
    *,
    device: str = "cpu",
    checkpoint_keys_mapping: str = "",
) -> torch.nn.Module:
    state_dict = safetensors.torch.load_file(filename, device=device)

    # Optional user-supplied renames (e.g. "model._orig_mod.//model.")
    if checkpoint_keys_mapping and "//" in checkpoint_keys_mapping:
        state_dict = rename_checkpoint_keys(state_dict, checkpoint_keys_mapping)

    state_dict, _ = standardise_state_dict(state_dict, set(model.state_dict().keys()))
    ignore_keys = model.config.load_ignore_keys
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(ignore_keys)}
    
    logging.info("load state dict")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if not all(key.startswith(ignore_keys) for key in missing) or unexpected:
        logging.info("Missing keys: ")
        for key in missing:
            logging.info(f"    {key}")
        logging.info("Unexpected keys: ")
        for key in unexpected:
            logging.info(f"    {key}")
        raise RuntimeError(f"SmolVLA {len(missing)} missing / {len(unexpected)} unexpected keys")
    
    if torch.isinf(model.normalize_inputs.buffer_observation_state["mean"]).any():
        logging.info("WARNING: Fail to load ACTION normalization parameters, use default dataset stats.")
        raise ValueError("Fail to load ACTION normalization parameters, use default dataset stats.")
        model.normalize_inputs.buffer_observation_state["mean"].data = torch.tensor([3.5377, 151.1628, 149.7005,  37.6185, -92.1948,  27.4836])
        model.normalize_inputs.buffer_observation_state["std"].data = torch.tensor([20.3128, 38.1258, 28.4479, 26.2382,  6.7694,  9.1883])
        model.normalize_targets.buffer_action["mean"].data = torch.tensor([3.5025, 152.6187, 147.8046,  36.6929, -92.1198,  26.8268])
        model.normalize_targets.buffer_action["std"].data = torch.tensor([20.4067, 37.2996, 29.7945, 27.0999,  6.8156,  9.7244])
        model.unnormalize_outputs.buffer_action["mean"].data = torch.tensor([3.5025, 152.6187, 147.8046,  36.6929, -92.1198,  26.8268])
        model.unnormalize_outputs.buffer_action["std"].data = torch.tensor([20.4067, 37.2996, 29.7945, 27.0999,  6.8156,  9.7244])

    return model


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    # Keep inference intermediates aligned with the model's float32 compute path.
    dtype = get_safe_dtype(torch.float32, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer
        self.model = VLAFlowMatching(config)
        self.reset()

        # async inference
        self._infer_thread = None
        self._queues_lock = threading.Lock()
        self._running_infer = False

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _canonicalize_inference_batch(self, observation: dict) -> dict[str, Tensor | str]:
        """Return a copied, batched image-only policy observation.

        The caller's camera observation stays in its native form for optional
        recording. Images become float32 BCHW tensors in [0, 1], and state
        becomes a batched float32 tensor on the policy device.
        """
        batch = dict(observation)
        device = torch.device(self.config.device)

        for key, feature in self.config.image_features.items():
            if key not in batch:
                raise KeyError(f"Missing required inference feature: {key}")
            value = batch[key]
            source_is_integer = isinstance(value, np.ndarray) and np.issubdtype(
                value.dtype, np.integer
            )
            if isinstance(value, np.ndarray):
                if key == "observation.images.DAVIS":
                    value = apply_white_balance(value, np.array([1.4066648, 0.6607546, 1.2891957]))
                value = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                source_is_integer = not torch.is_floating_point(value)
            else:
                raise TypeError(f"{key} must be a numpy array or torch tensor.")

            expected_channels = feature.shape[0]
            if value.ndim == 3:
                if value.shape[0] == expected_channels:
                    pass
                elif value.shape[-1] == expected_channels:
                    value = value.permute(2, 0, 1)
                else:
                    raise ValueError(f"{key} has no {expected_channels}-channel CHW/HWC layout.")
                value = value.unsqueeze(0)
            elif value.ndim != 4:
                raise ValueError(f"{key} must have HWC, CHW, or BCHW shape, got {tuple(value.shape)}.")
            if value.shape[1] != expected_channels:
                raise ValueError(f"{key} has invalid channel dimension in shape {tuple(value.shape)}.")
            value = value.to(device=device, dtype=torch.float32).contiguous()
            if source_is_integer or (value.numel() and value.max().item() > 1.0):
                value = value / 255.0
            batch[key] = value

        if OBS_STATE not in batch:
            raise KeyError(f"Missing required inference feature: {OBS_STATE}")
        state = batch[OBS_STATE]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"{OBS_STATE} must be a numpy array or torch tensor.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch[OBS_STATE] = state.to(device=device, dtype=torch.float32)
        return batch

    # HACK(aliberts, danaaubakirova): we overwrite this classmethod here to fix smolVLA-specific issues
    @classmethod
    def _load_as_safetensor(
        cls,
        model: "SmolVLAPolicy",
        model_file: str,
        map_location: str,
        strict: bool,
    ):
        # safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return load_smolvla(
            model,
            model_file,
            device=map_location,
            checkpoint_keys_mapping="model._orig_mod.//model.",
        )

    def get_optim_params(self):
        """
        Get optimizer parameters, optionally grouped by model components with different learning rates.
        
        Returns:
            If use_grouped_lr is False: All parameters (for backward compatibility)
            If use_grouped_lr is True: List of parameter groups, each with 'params' and 'lr' keys
        """
        if not self.config.use_grouped_lr:
            # Backward compatibility: return all parameters
            return self.parameters()
        
        # Helper function to get learning rate for a component
        default_lr = self.config.optimizer_lr
        default_wd = self.config.optimizer_weight_decay
        def get_lr(component_lr: float | None) -> float:
            return component_lr if component_lr is not None else default_lr
        def get_wd(component_wd: float | None) -> float:
            return component_wd if component_wd is not None else default_wd
        
        # Track all collected parameters using parameter objects (more reliable than names)
        grouped_params = set()
        param_groups_with_names = []
        
        # 1. VLM text model parameters - should be frozen
        if hasattr(self.model, 'vlm_with_expert') and hasattr(self.model.vlm_with_expert, 'vlm'):
            vlm = self.model.vlm_with_expert.vlm
            for name, param in vlm.named_parameters():
                if param.requires_grad:
                    raise ValueError(f"VLM parameter '{name}' is unexpectedly trainable! VLM should be frozen.")
        
        # 2. Action expert parameters (lm_expert)
        action_expert_params = []
        if hasattr(self.model, 'vlm_with_expert') and hasattr(self.model.vlm_with_expert, 'lm_expert'):
            lm_expert = self.model.vlm_with_expert.lm_expert
            for name, param in lm_expert.named_parameters():
                if param.requires_grad:
                    action_expert_params.append(param)
                    grouped_params.add(param)
        
        if action_expert_params:
            param_groups_with_names.append({
                'params': action_expert_params,
                'lr': get_lr(self.config.lr_action_expert),
                'wd': get_wd(self.config.wd_action_expert),
                'name': 'action_expert'
            })
        # 3. State projection parameters
        state_proj_params = []
        if hasattr(self.model, 'state_proj'):
            for name, param in self.model.state_proj.named_parameters():
                if param.requires_grad:
                    state_proj_params.append(param)
                    grouped_params.add(param)
        
        if state_proj_params:
            param_groups_with_names.append({
                'params': state_proj_params,
                'lr': get_lr(self.config.lr_state_proj),
                'wd': get_wd(self.config.wd_state_proj),
                'name': 'state_proj'
            })
        
        # 4. Action projection parameters (action_in_proj, action_out_proj, action_time_mlp_in, action_time_mlp_out)
        action_proj_params = []
        for proj_name in ['action_in_proj', 'action_out_proj', 'action_time_mlp_in', 'action_time_mlp_out']:
            if hasattr(self.model, proj_name):
                proj_module = getattr(self.model, proj_name)
                for name, param in proj_module.named_parameters():
                    if param.requires_grad:
                        action_proj_params.append(param)
                        grouped_params.add(param)
        
        if action_proj_params:
            param_groups_with_names.append({
                'params': action_proj_params,
                'lr': get_lr(self.config.lr_action_proj),
                'wd': get_wd(self.config.wd_action_proj),
                'name': 'action_proj'
            })
        
        # 5. Collect trainable parameters not assigned to a group (normally none).
        ungrouped_params = []
        ungrouped_param_names = []
        for name, param in self.named_parameters():
            if param.requires_grad and param not in grouped_params:
                ungrouped_params.append(param)
                ungrouped_param_names.append(name)
        
        # Put any ungrouped parameters in the 'other' group.
        if ungrouped_params:
            param_groups_with_names.append({
                'params': ungrouped_params,
                'lr': default_lr,
                'wd': default_wd,
                'name': 'other'
            })
        
        # Build parameter groups for the PyTorch optimizer.
        final_param_groups = []
        for group in param_groups_with_names:
            final_param_groups.append({
                'params': group['params'],
                'lr': group['lr'],
                'weight_decay': group.get('wd', default_wd),
                'name': group['name']
            })
        
        # Log parameter-group details.
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_grouped = sum(sum(p.numel() for p in group['params']) for group in final_param_groups)
        logging.info(f"[SmolVLA] parameters statistics (trainable: {total_trainable:,}, grouped: {total_grouped:,}):")
        for i, group in enumerate(final_param_groups):
            num_params = sum(p.numel() for p in group['params'])
            group_name = group['name']
            logging.info(f"  - {group_name}: {num_params:,} parameters, lr={group['lr']:.2e}, wd={group['weight_decay']:.2e}")
        
        if ungrouped_params:
            raise ValueError(f"[SmolVLA] found {len(ungrouped_params)} parameters in 'other' group: {ungrouped_param_names}")
        
        return final_param_groups

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        input batch dict:
          images are torch.float32, (B,C,H,W), [0,1], on device, already white balanced.
          state is float32, Bx1D tensor on device, the state of the robot.
          task is string, the task name.
          robot_type is string, the robot type.
        output:
          action is float32, (B,Chunk,OriActionDim) tensor on device, the action to be sent to the robot.
        """
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        

        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        
        start_time = time.perf_counter()
        images, img_masks = self.prepare_images(batch)
        logging.info(f"Prepare images time: {(time.perf_counter() - start_time)*1000:.3f} ms")

        start_time = time.perf_counter()
        lang_tokens, lang_masks = self.prepare_language(batch)
        logging.info(f"Prepare language time: {(time.perf_counter() - start_time)*1000:.3f} ms")

        start_time = time.perf_counter()
        state = self.prepare_state(batch)
        logging.info(f"Prepare state time: {(time.perf_counter() - start_time)*1000:.3f} ms")

        start_time = time.perf_counter()
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        logging.info(f"Model sample_actions time: {(time.perf_counter() - start_time)*1000:.3f} ms")

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    def ema_smooth(self, actions: Tensor) -> Tensor:
        """Exponential moving average smoothing in an action chunk.

        actions: (chunk_size, batch_size, action_dim)
        alpha: smoothing factor, between 0 and 1. Lower alpha means more smoothing.
        """
        T, B, D = actions.shape
        device = actions.device
        alpha = 0.7

        smoothed_actions = torch.zeros_like(actions, device=device)
        smoothed_actions[0] = actions[0]
        for t in range(1, T):
            smoothed_actions[t] = alpha * actions[t] + (1 - alpha) * smoothed_actions[t - 1]
        
        # optional linear tail correction
        tail_diff = actions[-1] - smoothed_actions[-1]
        correction_factors = torch.linspace(0, 1, steps=T, device=device).view(T, 1, 1)
        smoothed_actions += correction_factors * tail_diff

        return smoothed_actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        input batch dict:
            images are uint8, HWC, RGB, image is not white balanced.
            state is float32, Bx1D tensor on device, the state of the robot.
            task is string, the task name.
            robot_type is string, the robot type.
        output:
            action is float32, BxActionDim tensor on device, the action to be sent to the robot.
        """
        self.eval()

        preprocess_start = time.perf_counter()
        batch = self._canonicalize_inference_batch(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by querying the policy.
        if len(self._queues[ACTION]) == 0:
            wb_time = time.perf_counter() - preprocess_start
            logging.info(f"White balance time: {wb_time*1000:.3f} ms") # 21, 

            batch = self._prepare_batch(batch)  # Normalize inputs, do nothing for VISUAL features
            
            # _get_action_chunk will just ignore the extra keys in batch.
            # for missing keys:
            # if n_empty_cam > 0, create -1 padded images and 0 masks at the end of visual token sequence.
            # if n_empty_cam = 0, ignore missing keys

            mid_time = time.perf_counter()
            actions = self._get_action_chunk(batch, noise)  # (B, Chunk, OriActionDim) tensor
            infer_time = time.perf_counter() - mid_time
            logging.info(f"Get action chunk time: {infer_time*1000:.3f} ms")

            # Optional smooth method.
            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            actions = actions.transpose(0, 1)
            # actions  = self.ema_smooth(actions)

            self._queues[ACTION].extend(actions[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        # Convert tensor to scalar for WandB logging
        loss_dict["losses_after_forward"] = losses.mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            # Convert tensor to scalar for WandB logging
            loss_dict["losses_after_in_ep_bound"] = losses.mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        # Convert tensor to scalar for WandB logging
        loss_dict["losses_after_rm_padding"] = losses.mean().item()

        # For backward pass
        loss = losses.mean()
        # For backward pass
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        input image: (b,c,h,w) or (b,1,c,h,w), [0,1], torch.float32
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch[OBS_STATE].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            train_expert_only=self.config.train_expert_only,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        logging.info(f"State proj grad: {self.config.train_state_proj}")

        for params in self.action_in_proj.parameters():
            params.requires_grad = self.config.train_action_proj
        for params in self.action_out_proj.parameters():
            params.requires_grad = self.config.train_action_proj
        for params in self.action_time_mlp_in.parameters():
            params.requires_grad = self.config.train_action_proj
        for params in self.action_time_mlp_out.parameters():
            params.requires_grad = self.config.train_action_proj
        logging.info(f"Action proj grad: {self.config.train_action_proj}")

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens for SmolVLM processing.
        
        Args:
            images: list of image tensors
            img_masks: list of image masks
            lang_tokens: language token IDs
            lang_masks: language attention masks
            state: robot state tensor 
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        start_time = time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        logging.info(f"Embed prefix time: {(time.perf_counter() - start_time)*1000:.3f} ms")
        start_time = time.perf_counter()
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        logging.info(f"Prefix forward time (compute cache): {(time.perf_counter() - start_time)*1000:.3f} ms")
        start_time = time.perf_counter()

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        _time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while _time >= -dt / 2:
            expanded_time = _time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            # Euler step
            x_t += dt * v_t
            _time += dt
        logging.info(f"Iterative denoising time: {(time.perf_counter() - start_time)*1000:.3f} ms")
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
