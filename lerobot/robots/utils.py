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

from typing import Protocol

from lerobot.robots.configs import RobotConfig, So100RobotConfig


def get_arm_id(name: str, arm_type: str) -> str:
    return f"{name}_{arm_type}"


class Robot(Protocol):
    robot_type: str
    features: dict

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type != "so100":
        raise ValueError(f"Robot type '{robot_type}' is not available in this evla branch.")
    return So100RobotConfig(**kwargs)


def make_robot_from_config(config: RobotConfig) -> Robot:
    if not isinstance(config, So100RobotConfig):
        raise TypeError(f"Expected So100RobotConfig, got {type(config).__name__}.")
    from lerobot.robots.manipulator import ManipulatorRobot

    return ManipulatorRobot(config)


def make_robot(robot_type: str, **kwargs) -> Robot:
    return make_robot_from_config(make_robot_config(robot_type, **kwargs))
