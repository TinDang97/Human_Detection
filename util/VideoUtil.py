import time
import cv2
from threading import Thread
from queue import Queue


class VideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src, time_out=60, max_items=60, get_latest=False, retries=-1):
        assert type(get_latest) is bool, "Type of 'get_latest' must be bool. But got %s" % str(type(get_latest))

        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = not self.stream.isOpened()
        self.queue_frame = Queue(max_items) if not get_latest else None
        self.time_out = time_out
        self.thread = Thread(target=self.parallel_read_video, args=())
        self.get_latest = get_latest
        self.latest_frame = None
        self.retries = retries

    def isOpened(self):
        return self.stream.isOpened()

    def read_frame(self):
        _, frame = self.stream.read()
        retries = 0
        max_try = self.retries if self.retries >= 0 else 1
        while frame is None and retries < max_try:
            if self.retries >= 0:
                retries += 1
            self.stream = cv2.VideoCapture(self.src)
            time.sleep(0.5)
            _, frame = self.stream.read()

        return frame 

    def parallel_read_video(self):
        while not self.stopped:
            frame = self.read_frame()

            if not self.get_latest:
                self.queue_frame.put(frame, timeout=self.time_out)
            else:
                self.latest_frame = frame

        if not self.get_latest:
            self.queue_frame.task_done()
        self.stream.release()

    def getCaptureProps(self, propId):
        return self.stream.get(propId)

    def get(self):
        if self.get_latest:
            return self.latest_frame

        return self.queue_frame.get(timeout=self.time_out)

    def __iter__(self):
        if not self.thread.isAlive():
            self.thread.start()
        time.sleep(1)
        return self

    def __next__(self):
        if self.stopped:
            raise StopIteration

        frame = self.get()
        if frame is None:
            self.stop()
            raise StopIteration

        return frame

    def stop(self):
        self.stopped = True


class VideoWriter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, file_path, frame_size, fps, fourcc='MJPG', time_out=60, max_items=60):
        assert type(frame_size) is tuple and len(frame_size) == 2,  "Frame size must be tuple (width, height)"
        if type(fourcc) == str:
            fourcc = cv2.VideoWriter_fourcc(*fourcc)

        self.writer = cv2.VideoWriter(file_path, int(fourcc), fps, frame_size)
        self.queue_frame = Queue(max_items)
        self.stopped = False
        self.time_out = time_out
        self.thread = Thread(target=self.parallel_write_frame, args=())

    def start(self):
        self.thread.start()
        return self

    def parallel_write_frame(self):
        try:
            while not self.stopped or not self.queue_frame.empty():
                self.writer.write(self.queue_frame.get(timeout=self.time_out))
        except:pass
        finally:
            self.release()
            self.writer.release()
            self.queue_frame.task_done()

    def put(self, frame):
        if not self.thread.isAlive():
            self.thread.start()

        try:
            self.queue_frame.put(frame, timeout=self.time_out)
        except:
            self.release()
            return False
        return True

    def release(self):
        self.stopped = True

