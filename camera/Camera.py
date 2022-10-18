from threading import Thread
from collections import deque
import cv2
import time

from datetime import datetime, timedelta, timezone
import os
import gc

class BaseCamera(object):
    def __init__(self, threadname="VideoStream"):

        # Initialize thread
        self.thread = Thread(target=self.update, name=threadname, args=())
        self.thread.daemon = True
        self.thread_process = None
        # Initialize frame
        self.frame = None

        # Initialize thread flag
        self.stopped = False

    def start(self):
        """Start the frame grab thread"""
        # raise RuntimeError('Must be implemented by subclasses.')
        self.thread.start()
        time.sleep(1)
        return self

    def update(self):
        """Continuously Grab Next frame from the camera"""
        raise RuntimeError("Must be implemented by subclasses.")

    def read(self):
        """Return the next frame from the queue to the user"""
        try:
            return self.frame
        except:
            raise RuntimeError("Must implement Update Method by subclasses.")

    def stop(self):
        """Stop the frame grab thread"""
        self.stopped = True
        self.thread.join()


class OpencvCamera(BaseCamera):
    def __init__(self, src, threadname="VideoStream"):
        super(OpencvCamera, self).__init__(threadname)

        # initialize the camera thread
        self.stream = cv2.VideoCapture(src)

        # read the first frame
        (self.iscaptured, self.frame) = self.stream.read()

    def start(self):
        """Start the frame grab thread"""
        return super(OpencvCamera, self).start()

    def update(self):
        """Continuously Grab Next frame from the camera"""
        while True:

            # Stop the thread when the stop method is triggered
            if self.stopped:
                return

            # Get next frame
            (self.iscaptured, self.frame) = self.stream.read()

        time.sleep(0.1)
        self.stream.release()

    def read(self):
        """Return the next frame from the queue to the user"""
        return super(OpencvCamera, self).read()

    def stop(self):
        """Stop the frame grab thread"""
        super(OpencvCamera, self).stop()


DeviceMapper = {
    "webcam": 0,
    "logitech": "/dev/video4",
}


class LiveStream(BaseCamera):
    # Set the parameters needed
    queue_size = 256

    def __init__(self, device : str, threadname="LiveStream"):

        # Default: Built-in Webcam
        self.device = DeviceMapper.get(device, 0)

        self.stream = cv2.VideoCapture(self.device)
        # read the first frame
        (self.iscaptured, self.frame) = self.stream.read()

        if self.iscaptured:
            print("Initializing Video Stream...")
        
        # initialize the queue used to store frames read from
        # the video file
        self.deq = deque(maxlen=self.queue_size)
        self.deqRead = deque(maxlen=self.queue_size)

        super().__init__(threadname=threadname)

    def __enter__(self):
        return super().start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().stop()
        self.stream.release()
        self.deq.clear()
        self.deqRead.clear()
        gc.collect()


    def update(self) -> None:
        """Continuously Grab Next frame from the camera"""
        while True:

            # Stop the thread when the stop method is triggered
            if self.stopped:
                return

            if not self.stream.isOpened():
                raise IOError("Camera not opened.")
            # Get next frame
            (self.iscaptured, self.frame) = self.stream.read()
            if not self.iscaptured:
                continue

            time_captured = datetime.now(tz=timezone(timedelta(hours=8)))
            self.deq.appendleft({"image": self.frame, "time_captured": time_captured})

        time.sleep(0.1)
        

    def read(self):
        """Return the next frame from the queue to the user"""

        # return next frame in the queue
        while len(self.deq) <= 0:
            time.sleep(0.1)

        current_frame = self.deq.pop()

        return current_frame



if __name__ == "__main__":


    with LiveStream("webcam") as vid:
        time.sleep(1)

        while not vid.stopped:
            frame = vid.read()

            cv2.imshow("main", frame['image'])

            k = cv2.waitKey(1) & 0xFF
            if k==27:    # Esc key to stop
                print("Exit requested.")
                break
        pass
    pass
