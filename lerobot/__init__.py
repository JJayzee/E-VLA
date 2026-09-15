#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lightweight public registries for the evla package.

evla is a real-robot project built around the SO-100, DAVIS events, and
SmolVLA. Dataset repositories are user-defined and are intentionally not
hard-coded here; dataset conversion utilities operate on an explicit
repository ID instead.
"""

from lerobot.__version__ import __version__  # noqa: F401

# Policies, robots, motors, and cameras supported by this source tree.
available_policies = ["smolvla"]
available_robots = ["so100"]
available_motors = ["feetech"]

# Keep all camera backends discoverable: DAVIS is the primary sensor, while
# OpenCV and RealSense are useful for generic collection and debugging.
available_cameras = ["opencv", "intelrealsense", "inivationdavis"]
