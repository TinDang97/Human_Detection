import cv2

from util import show_image
from yolo import YOLO


class ObjectDetector:
    def __init__(self, weights_path, config_path, history_frame=10, detect_shadows=False, min_object_size=400):
        self.detector = YOLO(weights_path, config_path)
        self.background_sub = cv2.createBackgroundSubtractorKNN(history=history_frame, detectShadows=detect_shadows)
        self.min_object_size = min_object_size

    def detect(self, frame, display_bg=False):
        """
        Detect and respond box, score ones
        :param frame:
        :return:
        """
        boxes, scores = [], []
        fgMask = self.background_sub.apply(frame)

        # find motion objects and remove small objects
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) > self.min_object_size]

        # Detect human if have motion object
        if blobs:
            boxes, scores = self.detector.detect_person(frame)

        if display_bg:
            show_image('Background', fgMask, wait=False)

        for box, score in zip(boxes, scores):
            yield box, score
