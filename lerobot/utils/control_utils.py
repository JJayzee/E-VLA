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

########################################################################################
# Utilities
########################################################################################

# Fixed single-camera DAVIS control path for `predict_action` (robot inference loop).
# Image is expected white-balanced upstream; uint8 HWC before this helper converts it.
DAVIS_IMAGE_KEY = "observation.images.DAVIS"
DAVIS_IMAGE_HWC_SHAPE = (260, 346, 3)

import logging
import time
import traceback
from contextlib import nullcontext
from functools import cache

import numpy as np
import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.constants import ACTION, OBS_STATE
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, get_features_from_robot
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots.utils import Robot
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, has_method


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    for name in robot.leader_arms:
        key = f"read_leader_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRlead", robot.logs[key])

    for name in robot.follower_arms:
        key = f"write_follower_{name}_goal_pos_dt_s"
        if key in robot.logs:
            log_dt("dtWfoll", robot.logs[key])

        key = f"read_follower_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRfoll", robot.logs[key])

    for name in robot.cameras:
        key = f"read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp, task, robot_type=None, return_observation=False):
    """Predict from a copied image-only observation without mutating camera data."""
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        policy_observation = policy._canonicalize_inference_batch(observation)
        policy_observation["task"] = task if task else ""
        policy_observation["robot_type"] = robot_type if robot_type else ""
        action = policy.select_action(policy_observation)

        action = action.squeeze(0)
        action = action.to("cpu")

    if return_observation:
        return action, policy_observation
    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    warmup_time_s,
    fps,
):
    # We should get all cameras' observations, start threads
    # while warming up to check their work state.
    warmup_control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        events=events,
        fps=fps,
    )

@safe_stop_image_writer
def warmup_control_loop(
    robot,
    control_time_s=None,
    events=None,
    fps: int | None = None,
):
    # A time strict control loop for warmup.
    # Enable teleoperation and cameras capture.

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    loop_start_time = time.perf_counter()
    loop_end_time = loop_start_time + 1 / fps
    while time.perf_counter() - start_episode_t < control_time_s:
        # this read will trigger all cameras reading threads
        _, _ = robot.teleop_step(record_data=True)

        dt_s = loop_end_time - time.perf_counter()
        time.sleep(dt_s) if dt_s > 0 else None

        loop_start_time = loop_end_time
        loop_end_time = loop_start_time + 1 / fps

        if events["exit_early"]:
            events["exit_early"] = False
            break

    if 'DAVIS' in robot.cameras:
            # disconnect davis camera to stop read thread and clear the buffers
            robot.cameras['DAVIS'].disconnect()


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    fps,
    single_task,
):
    record_control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        dataset=dataset,
        events=events,
        fps=fps,
        single_task=single_task,
    )


def valid_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    save_data,
    record_every_control_frame,
    policy,
    fps,
    single_task,
):
    valid_control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        save_data=save_data,
        record_every_control_frame=record_every_control_frame,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        single_task=single_task,
    )



@safe_stop_image_writer
def record_control_loop(
    robot,
    control_time_s=None,
    dataset: LeRobotDataset = None,
    events=None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    # A time strict control loop for recording.

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if dataset is None:
        raise ValueError("You need to provide a dataset to record data.")
    
    if single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if fps is not None and dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    if not robot.is_connected:
        robot.connect()     # cameras will be connected, but threads will not start.

    # sync DAVIS inner frame phase with the control loop
    if 'DAVIS' in robot.cameras:
        davis_camera = robot.cameras['DAVIS']
        # reconnect the davis camera to control period
        davis_camera.disconnect()
        davis_camera.connect()

        # clear davis camera's buffer:
        if davis_camera.work_mode in ["image", "davis"]:
            while True:
                if davis_camera.camera.getNextFrame() is None:
                    break
            while True:
                if davis_camera.camera.getNextFrame() is not None:
                    print(f"Timestamp: get first frame timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
                    break
            # at this time, davis camera has just output a frame,
            # and this frame is immediately consumed by .getNextFrame().
            # so the buffer is empty now.

            # clear the events buffer
            if davis_camera.work_mode == "davis":
                old_events = davis_camera.camera.getNextEventBatch()
                while old_events is not None:
                    old_events = davis_camera.camera.getNextEventBatch()

            # busy wait 45ms to ensure next frame is ready.
            # and match the period of davis camera and the control loop.
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 0.045:
                pass
        else:
            # in events only mode, just clear the events buffer
            old_events = davis_camera.camera.getNextEventBatch()
            while old_events is not None:
                old_events = davis_camera.camera.getNextEventBatch()
        

    start_episode_t = time.perf_counter()
    loop_start_time = time.perf_counter()
    loop_end_time = loop_start_time + 1 / fps
    # print(f"1_Timestamp: episode start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
    frame_index = 0
    while time.perf_counter() - start_episode_t < control_time_s:
        print("\n\n---------------------------------------------------------------------------------------")
        print(f"Loop: Frame {frame_index:04d} start: {((time.perf_counter() * 1000)%1000):.2f} ms, end time: {(loop_end_time*1000)%1000:.2f} ms\n")
        observation, action = robot.teleop_step(record_data=True)
        # observation is a dict with keys:
        # - observation.state: the state of the robot
        # - observation.images: the images from the cameras: array(uint8, HWC)
        # action is a dict with keys:
        # - action: the action to be sent to the robot: tensor

        print(f"\nLoop: observation finish: {((time.perf_counter() * 1000)%1000):.2f} ms")

        frame = {**observation, **action}
        dataset.add_frame(frame, task=single_task)
        # Images are saved asynchronously; the episode buffer keeps their paths.

        dt_s = loop_end_time - time.perf_counter()
        if dt_s > 0:
            print(f"Loop: Frame: {frame_index:04d}\tWaiting for next frame (ms): {dt_s * 1000:.2f}")
            time.sleep(dt_s)

        dt_s = time.perf_counter() - loop_start_time
        log_control_info(robot, dt_s, fps=fps)

        final_time = time.perf_counter() - loop_start_time
        print(f"Loop: Frame: {frame_index:04d}    Final Time (ms): {final_time * 1000:.2f}    Timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
        frame_index += 1
        loop_start_time = loop_end_time
        loop_end_time = loop_start_time + 1 / fps

        if events["exit_early"]:
            events["exit_early"] = False
            break

    if 'DAVIS' in robot.cameras:
        # disconnect davis camera to stop read thread and clear the buffers
        robot.cameras['DAVIS'].disconnect()


@safe_stop_image_writer
def teleop_control_loop(
    robot,
    control_time_s=None,
    fps: int | None = None,
):
    # A time strict control loop for teleoperation.
    if control_time_s is None:
        control_time_s = float("inf")

    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()

    if fps is None:
        # no fps limit
        while time.perf_counter() - start_episode_t < control_time_s:
            robot.teleop_step(record_data=False)
    else:
        loop_start_time = time.perf_counter()
        loop_end_time = loop_start_time + 1 / fps
        while time.perf_counter() - start_episode_t < control_time_s:
            robot.teleop_step(record_data=False)

            dt_s = loop_end_time - time.perf_counter()
            time.sleep(dt_s) if dt_s > 0 else None
            loop_start_time = loop_end_time
            loop_end_time = loop_start_time + 1 / fps


def reset_environment(robot, events, reset_time_s, fps, max_relative_target=None):
    # Reset environment while record and valid
    # Not only to reset env but also to set the position of the robot
    # robot start position may be different from the end position of last episode
    # to record correcting dataset, robot end position could not be zero-position
    # so we need to teleoperate the robot while resetting the environment
    # to avoid large action and collision, we limit the max_relative_target

    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    ori_max_relative_target = None
    if max_relative_target is not None:
        ori_max_relative_target = robot.config.max_relative_target
        robot.config.max_relative_target = max_relative_target

    reset_control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
    )

    if ori_max_relative_target is not None:
        robot.config.max_relative_target = ori_max_relative_target


@safe_stop_image_writer
def reset_control_loop(
    robot,
    control_time_s=None,
    events=None,
    fps: int | None = None,
):

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    loop_start_time = time.perf_counter()
    loop_end_time = loop_start_time + 1 / fps
    while time.perf_counter() - start_episode_t < control_time_s:
        robot.teleop_step(record_data=False)

        dt_s = loop_end_time - time.perf_counter()
        if dt_s > 0:
            time.sleep(dt_s)

        loop_start_time = loop_end_time
        loop_end_time = loop_start_time + 1 / fps

        if events["exit_early"]:
            events["exit_early"] = False
            break


@safe_stop_image_writer
def valid_control_loop(
    robot,
    control_time_s=None,
    display_data=False,
    save_data=False,
    record_every_control_frame=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    # A time strict control loop for recording.
    print(f"Begin valid_control_loop: {((time.perf_counter() * 1000)%1000):.2f} ms")

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    if not robot.is_connected:
        robot.connect()

    # sync DAVIS inner frame phase with the control loop
    if 'DAVIS' in robot.cameras:
        print(f"Begin DAVIS sync: {((time.perf_counter() * 1000)%1000):.2f} ms")
        davis_camera = robot.cameras['DAVIS']
        # reconnect the davis camera to control period
        davis_camera.disconnect()
        davis_camera.connect()

        # clear davis camera's buffer:
        if davis_camera.work_mode in ["image", "davis"]:
            while True:
                if davis_camera.camera.getNextFrame() is None:
                    break
            while True:
                if davis_camera.camera.getNextFrame() is not None:
                    print(f"Timestamp: get first frame timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
                    break
            # at this time, davis camera has just output a frame,
            # and this frame is immediately consumed by .getNextFrame().
            # so the buffer is empty now.

            # DAVIS produces events in combined mode even though this branch
            # exposes image-only observations. Drain the hardware queue so it
            # cannot grow unbounded during deployment.
            if davis_camera.work_mode == "davis":
                old_events = davis_camera.camera.getNextEventBatch()
                while old_events is not None:
                    old_events = davis_camera.camera.getNextEventBatch()

            # busy wait 45ms to ensure next frame is ready.
            # and match the period of davis camera and the control loop.
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 0.045:
                pass
        else:
            # in events only mode, just clear the events buffer
            old_events = davis_camera.camera.getNextEventBatch()
            while old_events is not None:
                old_events = davis_camera.camera.getNextEventBatch()

    start_episode_t = time.perf_counter()
    loop_start_time = time.perf_counter()
    loop_end_time = loop_start_time + 1 / fps
    # print(f"1_Timestamp: episode start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
    frame_index = 0
    while time.perf_counter() - start_episode_t < control_time_s:
        # print("\n\n---------------------------------------------------------------------------------------")
        # print(f"Loop: Frame {frame_index:04d} start: {((time.perf_counter() * 1000)%1000):.2f} ms, end time: {(loop_end_time*1000)%1000:.2f} ms\n")
        
        observation = robot.capture_observation()
        # print(f"Loop: observation finish: {((time.perf_counter() * 1000)%1000):.2f} ms")
        # observation is a dict with keys:
        # - observation.state: the state of the robot, tensor
        # - observation.images: uint8, HWC, RGB, array

        if len(policy._queues[ACTION]) == 0:
            inference = True
        else:
            inference = False

        # state will add batch dim and move to gpu predict_action.
        # predict_action needs:
        # observation.images.DAVIS: uint8, HWC tensor
        # observation.state: float32, 1D tensor

        infer_start_time = time.perf_counter()

        pred_action, processed_observation = predict_action(
            observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp, 
            task=single_task, robot_type=robot.robot_type, return_observation=True
        )
        infer_time = time.perf_counter() - infer_start_time
        print(f"Total predict action time (ms): {infer_time * 1000:.2f} ms")
        # print(f"Loop: predict action finish: {((time.perf_counter() * 1000)%1000):.2f} ms")
        # Action can eventually be clipped using `max_relative_target` to limit the speed,
        # so action actually sent is saved in the dataset.

        action = robot.send_action(pred_action)
        action = {"action": action}

        if save_data and (record_every_control_frame or inference):
            # Save the image representation consumed by the policy.
            # gpu tensor, float32, 0-1, BCHW
            observation = processed_observation
            del observation["task"]
            del observation["robot_type"]
            for key in observation:
                if key.startswith("observation"):
                    observation[key] = observation[key].cpu()
                    observation[key] = observation[key].squeeze(0)
                    if "image" in key:
                        observation[key] = (observation[key].permute(1,2,0) * 255).numpy().astype('uint8')  # HWC uint8
            frame = {**observation, **action}
            dataset.add_frame(frame, task=single_task)
            # Image files are written asynchronously; the episode buffer keeps
            # their paths rather than duplicate image arrays.

        if display_data and not is_headless():
            for k, v in action.items():
                for i, vv in enumerate(v):
                    rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                image = observation[key]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                rr.log(key, rr.Image(image), static=True)

        dt_s = loop_end_time - time.perf_counter()
        if dt_s > 0:
            # print(f"Loop: Frame: {frame_index:04d}\tWaiting for next frame (ms): {dt_s * 1000:.2f}")
            time.sleep(dt_s)

        dt_s = time.perf_counter() - loop_start_time
        log_control_info(robot, dt_s, fps=fps)

        final_time = time.perf_counter() - loop_start_time
        # print(f"Loop: Frame: {frame_index:04d}    Final Time (ms): {final_time * 1000:.2f}    Timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
        frame_index += 1
        loop_start_time = time.perf_counter()
        loop_end_time = loop_start_time + 1 / fps

        if events["exit_early"]:
            events["exit_early"] = False
            break
    
    if 'DAVIS' in robot.cameras:
        # disconnect davis camera to stop read thread and clear the buffers
        robot.cameras['DAVIS'].disconnect()

    policy.reset()



# safe stop recording
# will spend 5 seconds to reset the robot
def stop_recording(robot, listener, display_data):

    if not robot.is_connected:
        robot.connect()

    robot.config.max_relative_target = 5

    loop_start_time = time.perf_counter()
    loop_end_time = loop_start_time + 5
    while time.perf_counter() < loop_end_time:
        robot.teleop_step(record_data=False)

    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**get_features_from_robot(robot, use_videos), **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
