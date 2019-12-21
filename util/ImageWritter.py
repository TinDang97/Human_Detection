import threading
import time
from multiprocessing import Queue

import cv2
from datetime import datetime


class ImageWriter:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, folder_path):
        self.__folder_path = folder_path
        self.__thread = threading.Thread(target=self.parallel_write, args=())
        self.__queue_frame = Queue()
        self.__started = False
        self.__stopped = False

    def parallel_write(self):
        try:
            while not self.__queue_frame.empty() or not self.__stopped:
                filename = f"capture_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                cv2.imwrite(f"{self.__folder_path}/{filename}", self.__queue_frame.get())
        except:
            pass

    def put(self, frame):
        if not self.__started:
            self.__thread.start()
            self.__started = True

        self.__queue_frame.put(frame)
        return True

    def release(self):
        self.__stopped = True

        while not self.__queue_frame.empty():
            time.sleep(0.5)
