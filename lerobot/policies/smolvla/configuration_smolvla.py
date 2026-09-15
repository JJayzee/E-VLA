# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = True  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    # which keys to ignore when loading the state dict
    load_ignore_keys: tuple[str, ...] = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
    # ("normalize_inputs", "normalize_targets", "unnormalize_outputs", "model.vlm_with_expert.event_")

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images for optional camera slots in upstream checkpoints.
    # left and right wrist cameras in addition to the primary camera.
    empty_cameras: int = 0

    # Controls shared vision embeddings when loading a base checkpoint.
    adapt_to_pi_aloha: bool = False

    # Controls delta-action decoding for compatible checkpoints.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    use_amp: bool = False

    # Attention utils
    use_cache: bool = True

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Event encoder configuration
    use_event_encoder: bool = True
    event_vit_layers: int = 4
    event_vit_hidden_dim: int = 384
    event_vit_num_heads: int = 6
    share_vision_embeddings: bool = True  # whether to reuse siglip vision embeddings in event encoder  

    # Adapter configuration
    event_fusion_layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11])  # Which SigLIP layers to fuse events at
    fusion_image_dropout: float = 0.0

    # Event window sample 
    event_window_size: int | None = None
    event_window_duration: int | None = None
    # Event representation
    event_repr: str = "accu_count"

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_event_branch_only: bool = False
    train_expert_and_event_branch: bool = False
    train_state_proj: bool = True
    train_action_proj: bool = True

    # Default training presets
    optimizer_lr: float = 2e-4  # Default learning rate
    optimizer_weight_decay: float = 1e-10 # Default weight decay
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 10
    scheduler_warmup_steps: int = 625
    scheduler_decay_steps: int = 11250
    scheduler_decay_lr: float = 5e-6

    # Grouped learning rates for different model components
    use_grouped_lr: bool = False  # Enable grouped learning rate management
    lr_action_expert: float | None = None  # Action expert (lm_expert) learning rate
    lr_event_encoder: float | None = None  # Event encoder learning rate
    lr_fusion_modules: float | None = None  # Event fusion modules learning rate
    lr_state_proj: float | None = None  # State projection learning rate
    lr_action_proj: float | None = None  # Action projection layers (action_in_proj, action_out_proj, action_time_mlp) learning rate

    wd_action_expert: float | None = None
    wd_event_encoder: float | None = None
    wd_fusion_modules: float | None = None
    wd_state_proj: float | None = None
    wd_action_proj: float | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.event_window_size is not None and self.event_window_duration is not None:
            raise ValueError("event_window_size and event_window_duration are mutually exclusive.")
        if self.event_window_size is not None and self.event_window_size <= 0:
            raise ValueError("event_window_size must be a positive integer.")
        if self.event_window_duration is not None and self.event_window_duration <= 0:
            raise ValueError("event_window_duration must be a positive duration in microseconds.")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    # override image_features and event_features properties from PreTrainedConfig
    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL and len(ft.shape) == 3}

    @property
    def event_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL and len(ft.shape) != 3}

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
