import os
import time

import cv2
from flask import Flask, Response, send_file
from ObjectDetector import ObjectDetector
from util.VideoUtil import VideoGetter
from argparse import ArgumentParser
from util import draw_prediction, show_image
from util.ImageWritter import ImageWriter
from threading import Thread
from multiprocessing import Array, Lock

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
args_parser.add_argument("--host", default="localhost", help="folder output path")
args_parser.add_argument("--port", default=4342, help="folder output path")
args = args_parser.parse_args()

print(args_parser.description)

if args.output_folder and not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)


def tracking(args, vid, detector, writer, msg, file_path, lock):
    if args.display:
        cv2.startWindowThread()

    for frame in vid:
        if cv2.waitKey(5) & 0xFF == ord('q'):
            vid.stop()
            break

        for box, score in detector.detect(frame, display_bg=args.display):
            label = f"person"
            if args.show_score:
                label += f" - {score:{.4}}"

            # Draw bounding rect human
            frame = draw_prediction(frame, label, box)
            with lock:
                msg.value = f"Have human!!! Alert!"
                print(msg.value)

        if args.output_folder and msg.value:
            latest_file = writer.put(frame)
            with lock:
                file_path.value = latest_file

        if args.display:
            show_image('Capture', frame, wait=False)

    vid.stop()
    writer.release()
    cv2.destroyAllWindows()


app = Flask(__name__)

if __name__ == '__main__':
    source = args.source
    try:
        source = int(source)
    except:
        pass

    vid = VideoGetter(source, get_latest=type(source) is int or "rtsp://" in source.lower())
    writer = None
    msg = Array('B', range(100))
    file_path = Array('B', range(100))
    lock = Lock()

    if args.output_folder:
        writer = ImageWriter(args.output_folder)

    if not vid.isOpened():
        print(f"Can't open video/stream at link '{args.source}'")
        exit(0)

    detector = ObjectDetector(args.weight, args.config)
    thresh_tracking = Thread(target=tracking, args=(args, vid, detector, writer, msg, file_path, lock))
    thresh_tracking.start()

    @app.route("/preview")
    def view_alert():
        def yield_msg():
            while 1:
                if not msg.value:
                    time.sleep(0.5)
                    continue

                http_link = f"http://{args.host}:{args.port}/capture/{file_path.value}"
                yield f"{msg.value} <a href='{http_link}' target='_blank'>{http_link}</a><br>"
                msg.value = ""
        return Response(yield_msg())


    @app.route("/capture/<filename>")
    def view_capture(filename):
        return send_file(f"{args.output_folder}/{filename}", mimetype='image/gif')

    @app.route("/stop")
    def stop_vid():
        vid.stop()
        return "DONE"

    app.run(args.host, port=args.port)