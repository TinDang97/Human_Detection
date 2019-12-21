import os
from util.VideoUtil import VideoGetter
import cv2
from datetime import datetime
from argparse import ArgumentParser

from util import draw_prediction, show_image
from yolo import YOLO
from util.ImageWritter import ImageWriter


args_parser = ArgumentParser(description="This repo using Opencv and Yolo_v3 to detect human.\r\n"
                                         "To know options, please type -h.\r\n"
                                         "Press q to stop.")
args_parser.add_argument("--source", default="0",
                         help="Path of source detection. Number for camera device or video or stream")
args_parser.add_argument("--weight", default="./data/yolov3.weights", help="Path of model weight file")
args_parser.add_argument("--config", default="./data/yolov3.cfg", help="Path of model config file")
args_parser.add_argument("--show_score", default=True, help="if True, show score with label")
args_parser.add_argument("--display", default=False, const=True, action="store_const", help="if True, display process")
args_parser.add_argument("--output-folder", default="", help="folder output path")
args_parser.add_argument("--min-size-object", default=400, help="folder output path")
args = args_parser.parse_args()

print(args_parser.description)

if args.output_folder and not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

if __name__ == '__main__':
    yolo = YOLO(args.weight, args.config)
    backSub = cv2.createBackgroundSubtractorKNN(history=10, detectShadows=False)

    source = args.source
    try:
        source = int(source)
    except:
        pass

    vid = VideoGetter(source, get_latest=type(source) is int or "rtsp://" in source.lower())

    if args.output_folder:
        writer = ImageWriter(args.output_folder)

    if args.display:
        cv2.startWindowThread()

    for frame in vid:
        if cv2.waitKey(5) & 0xFF == ord('q'):
            vid.stop()
            break

        fgMask = backSub.apply(frame)

        # find motion objects and remove small objects
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) > args.min_size_object]

        # Detect human if have motion object
        if blobs:
            boxes, scores = yolo.detect_person(frame)

            for box, score in zip(boxes, scores):
                label = f"person"
                if args.show_score:
                    label += f" - {score:{.4}}"

                # Draw bounding rect human
                frame = draw_prediction(frame, label, box)
                print(f"Have human!!! Alert!")

            if args.output_folder and boxes:
                writer.put(frame)

        if args.display:
            show_image('Capture', frame, wait=False)
            show_image('Background', fgMask, wait=False)

    vid.stop()
    writer.release()
    cv2.destroyAllWindows()
