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

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.cameras.configs import CameraConfig, InivationDavisCameraConfig
from lerobot.motors.configs import FeetechMotorsBusConfig, MotorsBusConfig


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass
class ManipulatorRobotConfig(RobotConfig):
    """Common configuration for the SO-100 manipulator and its cameras."""

    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    max_relative_target: list[float] | float | None = None
    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in (*self.leader_arms.values(), *self.follower_arms.values()):
                arm.mock = True
            for camera in self.cameras.values():
                camera.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name, arm in self.follower_arms.items():
                if len(arm.motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm "
                        f"'{name}' has {len(arm.motors)} motors."
                    )


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    """SO-100 arm with Feetech STS3215 motors and a DAVIS camera."""

    calibration_dir: str = ".cache/calibration/so100"
    max_relative_target: int | None = 20

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1",
                motors={
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "DAVIS": InivationDavisCameraConfig(
                fps=30,
                exposure_duration_ms=10,
                work_mode="davis",
                white_balance=0,
                first_frame_wb=0,
                events_range="all",
            ),
        }
    )

    mock: bool = False
