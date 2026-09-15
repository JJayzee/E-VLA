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

from lerobot.motors.configs import FeetechMotorsBusConfig, MotorsBusConfig


class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> dict[str, MotorsBus]:
    motors_buses = {}
    for key, cfg in motors_bus_configs.items():
        if cfg.type != "feetech":
            raise ValueError(f"The motor type '{cfg.type}' is not supported by the SO-100 branch.")
        from lerobot.motors.feetech import FeetechMotorsBus

        if not isinstance(cfg, FeetechMotorsBusConfig):
            raise TypeError(f"Expected FeetechMotorsBusConfig, got {type(cfg).__name__}.")
        motors_buses[key] = FeetechMotorsBus(cfg)
    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type != "feetech":
        raise ValueError(f"The motor type '{motor_type}' is not supported by the SO-100 branch.")
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(FeetechMotorsBusConfig(**kwargs))
