"""
This file contains utilities for recording frames from Intel Realsense cameras.
Add some functions to control the camera including trigger and master/slave mode.
Add frames alignment to align depth frames to color frames.
By JiajunZhai, 2025-06-04
"""

import argparse
import concurrent.futures
import logging
import math
import shutil
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
from PIL import Image

from lerobot.cameras.configs import IntelRealSenseCameraConfig
from lerobot.utils.robot_utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.utils.utils import capture_timestamp_utc

SERIAL_NUMBER_INDEX = 1

# Define constants for sync mode
SYNC_MODE_DEFAULT = 0
SYNC_MODE_MASTER = 1
SYNC_MODE_SLAVE = 2
SYNC_MODE_FULL_SLAVE = 3

# Define constants for output trigger
OUTPUT_TRIGGER_DISABLED = 0
OUTPUT_TRIGGER_ENABLED = 1


def find_cameras(raise_when_empty=True, mock=False) -> list[dict]:
    """
    Find the names and the serial numbers of the Intel RealSense cameras
    connected to the computer.
    """
    if mock:
        import tests.cameras.mock_pyrealsense2 as rs
    else:
        import pyrealsense2 as rs

    cameras = []
    for device in rs.context().query_devices():
        serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
        name = device.get_info(rs.camera_info.name)
        cameras.append(
            {
                "serial_number": serial_number,
                "name": name,
            }
        )

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
        )

    return cameras


def save_image(img_array, serial_number, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{serial_number}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save image for camera {serial_number} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    serial_numbers: list[int] | None = None,
    fps=None,
    exposure_duration_ms=None,
    width=None,
    height=None,
    color_mode="rgb",
    white_balance=None,
    gain=None,
    gamma=None,
    contrast=None,
    use_depth=False,
    rotation=0,
    sync_mode=None,
    use_output_trigger=False,
    use_align_frames=False,
    set_emitter=0,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given serial number.
    """
    if serial_numbers is None or len(serial_numbers) == 0:
        camera_infos = find_cameras(mock=mock)
        serial_numbers = [cam["serial_number"] for cam in camera_infos]

    if mock:
        import tests.cameras.mock_cv2 as cv2
    else:
        import cv2

    print("Connecting cameras")
    cameras = []
    for cam_sn in serial_numbers:
        print(f"serial_number={cam_sn}")
        config = IntelRealSenseCameraConfig(
            serial_number=cam_sn, fps=fps, exposure_duration_ms=exposure_duration_ms, width=width, height=height, color_mode=color_mode, 
            white_balance=white_balance, gain=gain, gamma=gamma, contrast=contrast,
            use_depth=use_depth, rotation=rotation, mock=mock,set_emitter=set_emitter,
            sync_mode=sync_mode, use_output_trigger=use_output_trigger, use_align_frames=use_align_frames
        )
        camera = IntelRealSenseCamera(config)
        camera.connect()
        print(
            f"IntelRealSenseCamera({camera.serial_number}, fps={camera.fps}, exposure={exposure_duration_ms} ms, \
            width={camera.capture_width}, height={camera.capture_height}, color_mode={camera.color_mode}, \
            use_depth={camera.use_depth}, rotation={camera.rotation}, align_frames={camera.use_align_frames}, sync_mode={camera.config.sync_mode}, output_trigger={camera.config.use_output_trigger})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed, and we will end up
                    # saving the same images from the cameras multiple times until the RAM/disk is full.
                    read_start_time = time.perf_counter()
                    if use_depth:
                        image, depth_map = camera.read() if fps is None else camera.async_read()
                    else:
                        image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")
                    read_end_time = time.perf_counter()
                    # print(f"Read time: {read_end_time - read_start_time:.4f} seconds")

                    if use_depth:
                        # resize depth map from (0,65535) to (0,255) for visual check.
                        dmax, dmin = np.max(depth_map), np.min(depth_map)
                        depth_map = ((1.0-(depth_map-dmin)/(dmax-dmin))*255).astype(np.uint8)
                        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
                        image = np.concatenate((image, depth_map), axis=1)

                    executor.submit(
                        save_image,
                        image,
                        camera.serial_number,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    # print(f"waiting for {1000*(1 / fps - dt_s):.2f} ms")
                    busy_wait(1 / fps - dt_s - 0.0002)  # 0.0002 is a experience delay.

                if time.perf_counter() - start_time > record_time_s:
                    break

                # print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

                frame_index += 1

            for camera in cameras:
                camera.disconnect()
    finally:
        print(f"Images have been saved to {images_dir}")
        


class IntelRealSenseCamera:
    """
    The IntelRealSenseCamera class is similar to OpenCVCamera class but adds additional features for Intel Real Sense cameras:
    - is instantiated with the serial number of the camera - won't randomly change as it can be the case of OpenCVCamera for Linux,
    - can also be instantiated with the camera's name — if it's unique — using IntelRealSenseCamera.init_from_name(),
    - depth map can be returned.

    To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
    ```bash
    python lerobot/cameras/intelrealsense.py --images-dir outputs/images_from_intelrealsense_cameras
    ```

    When an IntelRealSenseCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of instantiating with a serial number:
    ```python
    from lerobot.cameras.configs import IntelRealSenseCameraConfig

    config = IntelRealSenseCameraConfig(serial_number=128422271347)
    camera = IntelRealSenseCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of instantiating with a name if it's unique:
    ```
    config = IntelRealSenseCameraConfig(name="Intel RealSense D405")
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=30, width=1280, height=720)
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=90, width=640, height=480)
    config = IntelRealSenseCameraConfig(serial_number=128422271347, fps=90, width=640, height=480, color_mode="bgr")
    # Note: might error out upon `camera.connect()` if these settings are not compatible with the camera
    ```

    Example of returning depth:
    ```python
    config = IntelRealSenseCameraConfig(serial_number=128422271347, use_depth=True)
    camera = IntelRealSenseCamera(config)
    camera.connect()
    color_image, depth_map = camera.read()
    ```
    
    Example of enabling output trigger:
    ```python
    config = IntelRealSenseCameraConfig(serial_number=128422271347)
    camera = IntelRealSenseCamera(config)
    camera.connect()
    camera.set_output_trigger(OUTPUT_TRIGGER_ENABLED)
    ```
    """

    def __init__(
        self,
        config: IntelRealSenseCameraConfig,
    ):
        self.config = config
        if config.name is not None:
            self.serial_number = self.find_serial_number_from_name(config.name)
        else:
            self.serial_number = config.serial_number

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height
        
        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.exposure_duration_ms = config.exposure_duration_ms
        self.channels = config.channels
        self.color_mode = config.color_mode

        self.white_balance = config.white_balance
        self.gain = config.gain
        self.gamma = config.gamma
        self.contrast = config.contrast
        
        self.use_depth = config.use_depth
        self.mock = config.mock
        self.use_output_trigger = config.use_output_trigger
        self.set_emitter = config.set_emitter
        
        # Frame alignment
        self.use_align_frames = config.use_align_frames
        self.align = None

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}
        
        # Store the device object for advanced options
        self.device = None
        self.depth_sensor = None

        self.use_log = False

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def find_serial_number_from_name(self, name):
        camera_infos = find_cameras()
        camera_names = [cam["name"] for cam in camera_infos]
        this_name_count = Counter(camera_names)[name]
        if this_name_count > 1:
            # TODO(aliberts): Test this with multiple identical cameras (Aloha)
            raise ValueError(
                f"Multiple {name} cameras have been detected. Please use their serial number to instantiate them."
            )

        name_to_serial_dict = {cam["name"]: cam["serial_number"] for cam in camera_infos}
        cam_sn = name_to_serial_dict[name]

        return cam_sn

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is already connected."
            )

        if self.mock:
            import tests.cameras.mock_pyrealsense2 as rs
        else:
            import pyrealsense2 as rs

        config = rs.config()
        config.enable_device(str(self.serial_number))

        if self.fps and self.capture_width and self.capture_height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.capture_width and self.capture_height:
                config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
            else:
                config.enable_stream(rs.stream.depth)
    

        self.camera = rs.pipeline()

        pipeline_profile = config.resolve(rs.pipeline_wrapper(self.camera))
        if not pipeline_profile:
            raise RuntimeError("Failed to resolve pipeline.")
        
        self.device = pipeline_profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        if self.exposure_duration_ms is not None:
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # disable auto exposure for depth sensor
            self.depth_sensor.set_option(rs.option.exposure, self.exposure_duration_ms)  # set exposure/ms for depth sensor
        
        self.rgb_sensor = self.device.first_color_sensor()

        # set exposure time
        if self.exposure_duration_ms is not None:
            self.rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.rgb_sensor.set_option(rs.option.exposure, self.exposure_duration_ms)  # set exposure/ms for color sensor
        
        # set white balance
        if self.white_balance is not None:
            self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
            self.rgb_sensor.set_option(rs.option.white_balance, self.white_balance)  # 2800-6500
        # set gain
        if self.gain is not None:
            self.rgb_sensor.set_option(rs.option.gain, self.gain)  # 0~128
        #set gamma
        if self.gamma is not None:
            self.rgb_sensor.set_option(rs.option.gamma, self.gamma)  # 100~500
        # set contrast
        if self.contrast is not None:
            self.rgb_sensor.set_option(rs.option.contrast, self.contrast)  # 0~100
        

        # Apply sync mode and output trigger and emitter settings if provided in config
        if self.config.sync_mode is not None:
            try:
                self.depth_sensor.set_option(rs.option.inter_cam_sync_mode, self.config.sync_mode)
                print(f"Camera {self.serial_number} sync mode set to {self.config.sync_mode}")
            except Exception as e:
                print(f"Failed to set sync mode: {e}")
            
        if self.use_output_trigger:
            try:
                self.depth_sensor.set_option(rs.option.output_trigger_enabled, OUTPUT_TRIGGER_ENABLED)
                print(f"Camera {self.serial_number} output trigger enabled.")
            except Exception as e:
                print(f"Failed to set output trigger: {e}")

        if self.set_emitter is not None:
            if self.set_emitter not in [0, 1, 2]:
                raise ValueError(f"`set_emitter` is expected to be 0, 1 or 2, but {self.set_emitter} is provided.")
            try:
                self.depth_sensor.set_option(rs.option.emitter_enabled, self.set_emitter)
                emitter_state = {0: "disabled", 1: "enabled", 2: "auto"}
                print(f"Camera {self.serial_number} emitter {emitter_state[self.set_emitter]}.")
            except Exception as e:
                print(f"Failed to set emitter: {e}")

        try:
            profile = self.camera.start(config)
            is_camera_open = True
            
            # Create alignment processor if needed
            if self.use_align_frames and self.use_depth:
                self.align = rs.align(rs.stream.color)
                print(f"Frame alignment enabled: depth frames will be aligned to color frames")
            
        except RuntimeError:
            is_camera_open = False
            traceback.print_exc()

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `serial_number` is valid before printing the traceback
            camera_infos = find_cameras()
            serial_numbers = [cam["serial_number"] for cam in camera_infos]
            if self.serial_number not in serial_numbers:
                raise ValueError(
                    f"`serial_number` is expected to be one of these available cameras {serial_numbers}, but {self.serial_number} is provided instead. "
                    "To find the serial number you should use, run `python lerobot/cameras/intelrealsense.py`."
                )

            raise OSError(f"Can't access IntelRealSenseCamera({self.serial_number}).")
        

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = color_stream.as_video_stream_profile()
        actual_fps = color_profile.fps()
        actual_width = color_profile.width()
        actual_height = color_profile.height()

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set fps {self.fps} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_fps}."
            )
        if self.capture_width is not None and self.capture_width != actual_width:
            raise OSError(
                f"Can't set width {self.capture_width} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_width}."
            )
        if self.capture_height is not None and self.capture_height != actual_height:
            raise OSError(
                f"Can't set height {self.capture_height} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_height}."
            )

        self.fps = round(actual_fps)
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)

        self.is_connected = True

    def read(self, temporary_color: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 480 x 640 x 3)
        of type `np.uint8`, contrarily to the pytorch format which is float channel first.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 480 x 640) of type np.uint16.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        # if self.mock:
        #     import tests.cameras.mock_cv2 as cv2
        # else:
        #     import cv2

        start_time = time.perf_counter()
        # print(f"realsense start read(), timestamp:{(time.perf_counter() * 1000)%1000:.2f} ms")

        frame = self.camera.wait_for_frames(timeout_ms=5000)
        # new_time = frame.get_timestamp()%1000
        if self.use_log:
            print(f"\nRealsense: frame getting finish: {(time.perf_counter()*1000)%1000:.2f} ms\t\t\t\t<-----get\n")

        
        # Align frames if alignment is enabled
        if self.use_align_frames and self.use_depth:
            frame = self.align.process(frame)

        color_frame = frame.get_color_frame()

        if not color_frame:
            raise OSError(f"Can't capture color image from IntelRealSenseCamera({self.serial_number}).")

        color_image = np.asanyarray(color_frame.get_data())

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # IntelRealSense uses RGB color mode by default.
        if requested_color_mode == "bgr":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        h, w, _ = color_image.shape
        
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.capture_height} x {self.capture_width}). ({h} x {w}) returned instead."
            )
        
        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            # print(f"Reading depth map from IntelRealSenseCamera({self.serial_number})...")
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(f"Can't capture depth image from IntelRealSenseCamera({self.serial_number}).")

            depth_map = np.asanyarray(depth_frame.get_data())

            h, w = depth_map.shape
            if h != self.capture_height or w != self.capture_width:
                raise OSError(
                    f"Can't capture depth map with expected height and width ({self.capture_height} x {self.capture_width}). ({h} x {w}) returned instead."
                )

            if self.rotation is not None:
                depth_map = cv2.rotate(depth_map, self.rotation)

            # log the number of seconds it took to read the image
            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

            return color_image, depth_map
        else:
            # log the number of seconds it took to read the image
            self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

            return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            # not lock while waiting for frames, mean thread can 
            # get self.color_image and self.depth_map when waiting for next frame
            if self.use_depth:
                color_image, depth_map = self.read()
            else:
                color_image = self.read()
            # lock while updating self.color_image and self.depth_map
            with self.lock:
                self.color_image = color_image
                if self.use_depth:
                    self.depth_map = depth_map

    def async_read(self):
        """Access the latest color image"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )
        # print(f"realsense start async_read, timestamp:{(time.perf_counter() * 1000)%1000:.2f} ms")
        if self.thread is None:
            self.lock = threading.Lock()
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            # TODO(rcadene, aliberts): intelrealsense has diverged compared to opencv over here
            num_tries += 1
            print(f"realsense async_read try: {num_tries}, timestamp:{(time.perf_counter() * 1000)%1000:.2f} ms")
            time.sleep(0.001)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )
            
        # Copy under the lock to avoid concurrent modification by the read loop.
        with self.lock:
            if self.use_depth:
                return self.color_image.copy(), self.depth_map.copy()
            else:
                return self.color_image.copy()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.stop()
        self.camera = None
        self.device = None
        self.depth_sensor = None
        self.rgb_sensor = None

        self.is_connected = False
            
    def get_sync_mode(self):
        """
        Get the current inter-camera synchronization mode.
        
        Returns:
            The current sync mode (0, 1, 2, or 3)
        """
        # if not self.is_connected:
        #     raise RobotDeviceNotConnectedError(
        #         f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
        #     )
            
        if self.mock:
            import tests.cameras.mock_pyrealsense2 as rs
        else:
            import pyrealsense2 as rs
            
        try:
            return self.depth_sensor.get_option(rs.option.inter_cam_sync_mode)
        except Exception as e:
            print(f"Failed to get sync mode: {e}")
            return None
            
    def set_output_trigger(self, enabled):
        """
        Enable or disable the output trigger signal.
        
        Args:
            enabled: One of the following values:
                - OUTPUT_TRIGGER_DISABLED (0): Disable output trigger
                - OUTPUT_TRIGGER_ENABLED (1): Enable output trigger
                
        Notes:
            By default, the output trigger signal is aligned with depth frames.
            If you need the trigger to be synchronized with RGB frames instead,
            you should:
            
            1. Enable depth streaming (set use_depth=True in the config)
            2. Enable frame alignment (set align_frames=True in the config)
            
            When frame alignment is enabled, depth frames are spatially aligned to
            match the color frames, and the trigger signal will effectively be
            synchronized with the color frames.
            
            Example:
            ```python
            config = IntelRealSenseCameraConfig(
                serial_number=123456789,
                use_depth=True,
                align_frames=True,
                output_trigger=OUTPUT_TRIGGER_ENABLED
            )
            camera = IntelRealSenseCamera(config)
            camera.connect()
            # Now the trigger signal will be aligned with RGB frames
            ```
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
            )
            
        if self.mock:
            import tests.cameras.mock_pyrealsense2 as rs
        else:
            import pyrealsense2 as rs
            
        if enabled not in [OUTPUT_TRIGGER_DISABLED, OUTPUT_TRIGGER_ENABLED]:
            raise ValueError(f"Invalid value: {enabled}. Must be 0 or 1.")
            
        try:
            self.depth_sensor.set_option(rs.option.output_trigger_enabled, enabled)
            print(f"Camera {self.serial_number} output trigger {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            print(f"Failed to set output trigger: {e}")
            
    def get_output_trigger(self):
        """
        Get the current output trigger setting.
        
        Returns:
            The current output trigger setting (0 or 1)
        """
        # if not self.is_connected:
        #     raise RobotDeviceNotConnectedError(
        #         f"IntelRealSenseCamera({self.serial_number}) is not connected. Try running `camera.connect()` first."
        #     )
            
        if self.mock:
            import tests.cameras.mock_pyrealsense2 as rs
        else:
            import pyrealsense2 as rs
            
        try:
            return self.depth_sensor.get_option(rs.option.output_trigger_enabled)
        except Exception as e:
            print(f"Failed to get output trigger setting: {e}")
            return None

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `IntelRealSenseCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--serial-numbers",
        type=int,
        nargs="*",
        default=None,
        help="List of serial numbers used to instantiate the `IntelRealSenseCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--exposure-duration-ms",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Set the exposure time for all cameras. If not provided, use the default exposure time of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        default="rgb",
        help="Set the color mode. Default rgb.",
    )
    parser.add_argument(
        "--white-balance",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Default None (auto white balance). Range: 2800-6500.",
    )
    parser.add_argument(
        "--gain",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Set the gain for RGB camera. Range: 0-128.",
    )
    parser.add_argument(
        "--gamma",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Set the gamma for RGB cameras. Range: 100-500.",
    )
    parser.add_argument(
        "--contrast",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Set the contrast for RGB cameras. Range: 0-100.",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        default=False,
        help="Set whether to use depth frames. Default False.",
    )
    parser.add_argument(
        "--rotation",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Set the rotation of the frames. Default None. Can be -90, 90 or 180.",
    )
    parser.add_argument(
        "--sync-mode",
        type=int,
        default=0,
        help="Set the inter-camera synchronization mode. Default None (no sync). "
             "Options: 0 (default mode), 1 (master mode), 2 (slave mode), 3 (full slave mode).",
    )
    parser.add_argument(
        "--set-emitter",
        type=int,
        default=0,
        help="Set the emitter mode. Default 0. "
             "Options: 0 (off), 1 (laser), 2 (laser auto).",
    )
    parser.add_argument(
        "--use-output-trigger",
        action="store_true",
        default=False,
        help="Set the output trigger mode. Default False (disabled). "
    )
    parser.add_argument(
        "--use-align-frames",
        action="store_true",
        default=False,
        help="Set whether to align depth frames to color frames. Default False. "
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_intelrealsense_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    print(args)
    save_images_from_cameras(**vars(args))
