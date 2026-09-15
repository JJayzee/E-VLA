import argparse
import concurrent.futures
import datetime
import json
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

import cv2
import dv_processing as dv
import numpy as np

from lerobot.cameras.configs import InivationDavisCameraConfig
from lerobot.cameras.davis_utils import (
    apply_white_balance,
    compute_white_balance_gains,
    save_image_and_events,
)
from lerobot.utils.robot_utils import RobotDeviceNotConnectedError
from lerobot.utils.utils import capture_timestamp_utc


def save_images_from_cameras(
    images_dir: Path,
    fps=30,
    exposure_duration_ms=10,
    color_mode="rgb",
    record_time_s=2,
    work_mode="davis",
    white_balance=0,
    first_frame_wb=0,
    events_range="all",
    **kwargs,
):
    """
    Initializes the camera and saves images to the directory. Useful to visually identify the camera
    """
    assert fps is not None, "Fps must be provided for none blocking read"

    print("Connecting davis camera")
    config = InivationDavisCameraConfig(
        fps=fps, exposure_duration_ms=exposure_duration_ms, color_mode=color_mode, 
        work_mode=work_mode, white_balance=white_balance, first_frame_wb=first_frame_wb, events_range=events_range
    )
    camera = InivationDavisCamera(config)
    camera.connect()

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images and events to {images_dir}")
    frame_index = 0

    # clear old frames in buffer
    while True:
        if camera.camera.getNextFrame() is None:
            break

    while True:
        if camera.camera.getNextFrame() is not None:
            # print(f"Timestamp: get first frame timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
            break
    # at this time, the camera has just output a frame,
    # and this frame is immediately consumed by .getNextFrame().
    # so the buffer is empty now.
    
    # busy wait 55ms to ensure next frame is ready
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < 0.055:
        pass

    img_timestamps_start = []
    img_timestamps_end = []
    ev_tempstamps_start = []
    ev_tempstamps_end = []
    ev_num = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            episode_start_time = time.perf_counter()
            frame_start_time = time.perf_counter()
            frame_end_time = frame_start_time + 1 / fps
            # using thread pool to save images, avoid blocking the main thread
            # print(f"Timestamp: episode start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
            while time.perf_counter() - episode_start_time <= record_time_s:
                # print(f"Timestamp: frame loop start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
                image, timestamps, events = camera.read() if fps is None else camera.async_read()
                
                img_timestamps_start.append(timestamps[0])
                img_timestamps_end.append(timestamps[1])
                if events is not None:
                    ev_tempstamps_start.append(events['timestamp'][0])
                    ev_tempstamps_end.append(events['timestamp'][-1])
                    ev_num.append(len(events))
                else:
                    ev_tempstamps_start.append(-1)
                    ev_tempstamps_end.append(-1)
                    ev_num.append(0)

                executor.submit(
                    save_image_and_events,
                    image,
                    events,
                    frame_index,
                    images_dir,
                )

                # print(f"Timestamp: waiting start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
                dt = frame_end_time - time.perf_counter()
                if dt > 0:
                    print(f"Frame: {frame_index:04d}\tWaiting for next frame (ms): {dt * 1000:.2f}")
                    time.sleep(dt)
                final_time = time.perf_counter() - frame_start_time
                print(f"Frame: {frame_index:04d}\tFinal Time (ms): {final_time * 1000:.2f}")
                # print(f"Timestamp: frame final timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
                frame_index += 1
                frame_start_time = frame_end_time
                frame_end_time = frame_start_time + 1 / fps

            camera.disconnect()
    finally:
        # write a timestamps.json file to the images_dir
        N = len(img_timestamps_start)
        with open(images_dir / "timestamps.jsonl", "w") as f:
            for i in range(N):
                timestamp_line = {}
                timestamp_line["frame_index"] = i
                timestamp_line["start"] = int(img_timestamps_start[i])
                timestamp_line["end"] = int(img_timestamps_end[i])
                timestamp_line["ev_start"] = int(ev_tempstamps_start[i])
                timestamp_line["ev_end"] = int(ev_tempstamps_end[i])
                timestamp_line["num_events"] = int(ev_num[i])
                f.write(json.dumps(timestamp_line) + "\n")

        # the executor will wait for all tasks to finish before exiting
        if white_balance:
            print(f"Using white balance gains: {camera.gains}")
        else:
            print(f"Not using white balance.")
        print(f"Images and events have been saved to {images_dir}")

class InivationDavisCamera:
    def __init__(
        self,
        config: InivationDavisCameraConfig,
    ):
        self.config = config
        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.logs = {}

        self.fps = config.fps
        self.exposure_duration_ms = config.exposure_duration_ms
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.channels = config.channels
        self.work_mode = config.work_mode
        self.white_balance = config.white_balance
        self.first_frame_wb = config.first_frame_wb
        self.events_range = config.events_range
        self.gains = None
        # self.pre_cal_gains = np.array([1.2366, 0.6378, 1.6042])             # for room lighe
        self.pre_cal_gains = np.array([1.2891957, 0.6607546, 1.4066648])    # for lamp illumination
        self.timestamp_base = None

        self.image = None
        self.events = None
        self.image_timestamp = None
        self.flag = 0   # flag to indicate if the frame is old

        self.use_log = True

    def connect(self):
        print(f"DAVIS work mode: {self.work_mode}")
        # Use dv_processing to connect to the DAVIS camera
        self.camera = dv.io.CameraCapture("", type=dv.io.CameraCapture.CameraType.DAVIS)

        if not self.camera.isEventStreamAvailable() or not self.camera.isFrameStreamAvailable():
            # If no event stream is available, we cannot proceed
            raise RuntimeError("Camera does not support event stream and frame stream.")

        print(f"Connecting to DAVIS camera...")

        # Configure frame capture
        if self.work_mode == "image":
            self.camera.setDavisReadoutMode(dv.io.CameraCapture.DavisReadoutMode.FramesOnly)
        elif self.work_mode == "events":
            self.camera.setDavisReadoutMode(dv.io.CameraCapture.DavisReadoutMode.EventsOnly)
        elif self.work_mode == "davis":
            self.camera.setDavisReadoutMode(dv.io.CameraCapture.DavisReadoutMode.EventsAndFrames)
        else:
            raise ValueError(f"Invalid work_mode: {self.work_mode}. Choose from 'image', 'events', 'davis'.")

        if self.work_mode in ["image", "davis"]:
            # Set frame rate based on config
            frame_interval_ms = datetime.timedelta(milliseconds=1000/self.fps)
            exposure_duration_ms = datetime.timedelta(milliseconds=int(self.exposure_duration_ms))
            self.camera.setDavisFrameInterval(frame_interval_ms)  # ms
            if self.exposure_duration_ms is not None:
                self.camera.setDavisExposureDuration(exposure_duration_ms)    # ms
                print(f"fps set to {self.fps} fps, exposure time set to {self.exposure_duration_ms} ms")
            else:
                self.camera.enableDavisAutoExposure()
                print(f"fps set to {self.fps} fps, exposure time set to auto")

        if self.work_mode in ["events", "davis"]:
            # Set the noise filter
            self.noise_filter = dv.noise.BackgroundActivityNoiseFilter(self.camera.getFrameResolution())
            self.event_buffer = dv.EventStore()

        if self.work_mode in ["image", "davis"] and self.white_balance:
            if self.first_frame_wb:
                print("Getting first frame for white balance.")
                # get one frame to set white balance
                while True:
                    raw_frame = self.camera.getNextFrame()
                    if raw_frame is not None:
                        break
                    time.sleep(0.001)
                image = np.asanyarray(raw_frame.image)
                self.gains = compute_white_balance_gains(image)
            else:
                print("Using pre-calibrated white balance gains.")
                self.gains = self.pre_cal_gains
            print(f"White balance gains: {self.gains}")
        
        # set timestamp base
        if self.work_mode in ["image", "davis"]:
            while True:
                raw_frame = self.camera.getNextFrame()
                if raw_frame is not None:
                    break
            # print(f"Set timestamp base to {raw_frame.timestamp}")
            self.timestamp_base = raw_frame.timestamp
        else:
            events = None
            while events is None:
                events = self.camera.getNextEventBatch()
            # print(f"Set timestamp base to {events[0].timestamp()}")
            self.timestamp_base = events[0].timestamp()

        print(f"Connected to DAVIS camera: {self.config.name}, work_mode: {self.work_mode}, fps: {self.fps}.")
        self.is_connected = True
        
        return True
        
    def get_latest_frame(self, frame):
        next_frame = self.camera.getNextFrame()
        if next_frame is None:
            return frame
        else:
            print(f"\nWARNING: Skipping old frame, get next frame")
            return self.get_latest_frame(next_frame)

    def read(self, temporary_color: str | None = None):
        """
        Read DAVIS image and event data. Assuming that the camera is in "davis" or "image" mode.
        The method is a blocking method, which means not immediately return if no new data is available.
        But it will not busy wait to lock the PIL.
        Returns the latest frame produced by the camera.
        If many frames in buffer, skip all old frames.
        if no frames in buffer, sleep and retry until new data is available.

        Returns:
            image (numpy.array, uint8): The captured image as a numpy array of shape (H, W, C).
            timestamp (list): The start and end timestamp of the frame in microseconds.
            events (structure array): The captured events as a structure array:
            {'names': ['timestamp', 'x', 'y', 'polarity'], 'formats': ['<i8', '<i2', '<i2', 'i1'],
            'offsets': [0, 8, 10, 12], 'itemsize': 16}
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DAVISCamera({self.config.name}) is not connected. Try running `camera.connect()` first."
            )

        if self.work_mode not in ["image", "davis"]:
            raise RuntimeError(f"Camera is not in 'image' or 'davis' mode, but in '{self.work_mode}' mode.")
        
        data_start_time = time.perf_counter()
        # t1 = (time.perf_counter() * 1000)%1000
        # print(f"Timestamp: read() start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")

        # Get the latest frame in buffer.
        # start_time = time.perf_counter()
        while True:
            raw_frame = self.camera.getNextFrame()
            if raw_frame is not None:
                raw_frame = self.get_latest_frame(raw_frame)
                break
            else:
                time.sleep(0.001)  # avoid busy waiting

        # t2 = (time.perf_counter() * 1000)%1000
        # if self.use_log:
        #     print(f"\nDAVIS: frame getting finish: {((time.perf_counter() * 1000)%1000):.2f} ms\t\t\t\t<-----get")
        # wait_time = t2 - t1
        # print(f"Wait time for frame: {wait_time:.2f} ms")
        
        image = np.asanyarray(raw_frame.image)
        start_time_ori = raw_frame.timestamp    # ori ts
        exposure_time_ori = int(raw_frame.exposure.total_seconds() * 1000000)  # us
        end_time_ori = int(start_time_ori + exposure_time_ori)
        start_time = start_time_ori - self.timestamp_base
        end_time = end_time_ori - self.timestamp_base

        if self.white_balance:
            image = apply_white_balance(image, self.gains)

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided.")
        
        # Inivation DAVIS camera use BGR color mode by default.
        if requested_color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.work_mode == "image":
            delta_timestamp_s = time.perf_counter() - data_start_time
            timestamp_utc = capture_timestamp_utc()
            self.logs["timestamp_utc"] = timestamp_utc
            self.logs["delta_timestamp_s"] = delta_timestamp_s
            return image, [start_time, end_time], None

        # if self.use_log:
        #     print(f"DAVIS: event getting start: {((time.perf_counter() * 1000)%1000):.2f} ms")
        events_result = None
        
        if self.events_range == "in_exposure":
            events = self.camera.getNextEventBatch()
            while (events is not None) and (events[0].timestamp()<=end_time_ori):
                # print(f"event_timestamp: {int(events[0].timestamp()/1000)}")
                # print(f"event_timestamp: {int(events[len(events)-1].timestamp()/1000)}")
                if events[len(events)-1].timestamp() > start_time_ori:
                    self.noise_filter.accept(events)
                    events = self.noise_filter.generateEvents()
                    self.event_buffer.add(events)
                # print(f"get batch begin: {((time.perf_counter() * 1000)%1000):.2f} ms")
                events = self.camera.getNextEventBatch()
            if start_time_ori is not None and end_time_ori is not None:
                events_result = self.event_buffer.sliceTime(start_time_ori, end_time_ori).numpy()
        elif self.events_range == "all":
            events = self.camera.getNextEventBatch()
            while events is not None:
                self.noise_filter.accept(events)
                events = self.noise_filter.generateEvents()
                self.event_buffer.add(events)
                events = self.camera.getNextEventBatch()
            events_result = self.event_buffer.numpy()
        else:
            raise ValueError(f"Invalid events_range: {self.events_range}. Choose from 'all', 'in_exposure'.")

        self.event_buffer = dv.EventStore()  # clear buffer after reading

        num_events = 0 if events_result is None else len(events_result)
        # if self.use_log:
        #     print(f"DAVIS: Number of events: {num_events}.")
        
        # reset the timestamp base
        if num_events > 0:
            events_result['timestamp'] -= self.timestamp_base
        else:
            events_result = np.empty(0, dtype=[('timestamp', '<i8'), ('x', '<i2'), ('y', '<i2'), ('polarity', 'i1')])

        if self.work_mode == "davis":
            timestamp_utc = capture_timestamp_utc()
            delta_timestamp_s = time.perf_counter() - data_start_time
            self.logs["timestamp_utc"] = timestamp_utc
            self.logs["delta_timestamp_s"] = delta_timestamp_s
            return image, [start_time, end_time], events_result

    def read_loop(self):
        """ 
        This method will always write the latest frame to self.image and self.events.
        It is a blocking method.
        If many frames in buffer, skip all old frames.
        if no frames in buffer, block until new data is available. (implemented by self.read())
        """
        while not self.stop_event.is_set():
            # if self.use_log:
            #     print(f"Timestamp: thread read_loop() start timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
            image, image_timestamp, events_result = self.read() # it returns every 33ms.

            # lock while updating
            with self.lock:
                # if self.use_log:
                #     print(f"DAVIS: update self.image: {((time.perf_counter() * 1000)%1000):.2f} ms\t\t\t\t<-----up\n")
                self.image = image
                self.events = events_result
                self.image_timestamp = image_timestamp
                self.flag = 1
            
    def async_read(self):
        """Return the latest DAVIS sample without adding a batch dimension.

        Returns:
            ``(image, timestamps, events)`` where image is a copied NumPy HWC uint8 RGB array,
            timestamps is ``[exposure_start_us, exposure_end_us]``, and events is a copied structured
            NumPy array with fields ``timestamp, x, y, polarity`` in ``davis`` mode. Events is ``None``
            in ``image`` mode. The first call starts the background reader and waits for its first frame.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DAVISCamera({self.config.name}) is not connected. Try running `camera.connect()` first."
            )
        if self.work_mode == "events":
            raise NotImplementedError(
                "InivationDavisCamera.async_read() does not yet support events-only mode. "
                "Use 'davis' mode for synchronized frames and events."
            )

        # print(f"async read begin Timestamp: {((time.perf_counter() * 1000)%1000):.2f} ms")
        if self.thread is None:
            self.lock = threading.Lock()
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.image is None:
            num_tries += 1
            time.sleep(0.001)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        # Copy under the lock to avoid concurrent modification by the read loop.
        with self.lock:
            if self.flag == 0:
                print("WARNING: No fresh DAVIS frame; returning the latest buffered frame.")
            self.flag = 0
            if self.work_mode == "davis":
                return self.image.copy(), self.image_timestamp.copy(), self.events.copy()
            elif self.work_mode == "image":
                return self.image.copy(), self.image_timestamp.copy(), None
            else:
                raise RuntimeError(f"Camera is not in 'image' or 'davis', but in '{self.work_mode}' mode.")
    
    def disconnect(self):
        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None
        
        self.event_buffer = None

        if self.camera is not None:
            self.camera = None
        
        self.image = None
        self.image_timestamp = None
        self.events = None
        self.flag = 0

        self.is_connected = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `InivationDavisCamera`."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--exposure-duration-ms",
        type=int,
        default=None,
        help="Set the exposure time for all cameras. If not provided, use the default exposure time of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_inivationdavis_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    parser.add_argument(
        "--work-mode",
        type=str,
        default="davis",
        help="Set the work mode. Options are 'davis' or 'image'.",
    )
    parser.add_argument(
        "--white-balance",
        type=int,
        default=0,
        help="If set, apply white balance.",
    )
    parser.add_argument(
        "--first-frame-wb",
        type=int,
        default=0,
        help="If set, use the first frame to compute white balance gains and apply to all subsequent frames.",
    )
    args = parser.parse_args()
    print(args)
    save_images_from_cameras(**vars(args))
