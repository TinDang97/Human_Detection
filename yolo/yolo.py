import cv2
import numpy
import numpy as np


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


class YOLO:
    def __init__(self, weights_path, config_path, image_output=(416, 416), scale=1/255):
        self.image_output = image_output
        self.scale = scale
        self.label = "person"
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = 0.9
        self.nms_threshold = 0.4

    def detect_person(self, image):
        blob = cv2.dnn.blobFromImage(image, self.scale, self.image_output, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        confidences = []
        boxes = []

        h, w, _ = image.shape

        outs = self.net.forward(get_output_layers(self.net))
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != 0:
                    continue

                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    w = int(detection[2] * w)
                    h = int(detection[3] * h)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        if len(boxes) <= 0:
            return boxes, confidences

        keep = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        if not list(keep):
            return [], []

        return numpy.array(boxes)[keep[:, 0]].tolist(), numpy.array(confidences)[keep[:, 0]].tolist()
