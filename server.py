import os
import signal
import traceback

# pylint: disable=no-member
import cv2 as cv
from flask import Flask, Response, render_template

from app import *
from main import detect_object, draw_objects

app = Flask(__name__)

cap = cv.VideoCapture(0)

point = (0.5,) * 2
prev_gray_frame = None


model_folder = "./ssd_mobilenet_v2_coco_2018_03_29"

model = "frozen_inference_graph.pb"
config = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classes = "coco_labels.txt"

with open(classes, "r", encoding="utf-8") as f:
    labels = [f.strip() for f in f.readlines()]


object_detector = cv.dnn.readNetFromTensorflow(
    os.path.join(model_folder, model),
    config,
)


def proccess_img(frame):
    global point, prev_gray_frame
    scalar = 0.5
    small_frame = cv.resize(frame, (0, 0), fx=scalar, fy=scalar)

    if prev_gray_frame is None:
        prev_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
        return frame

    next_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    flow = get_optical_flow(prev_gray_frame, next_gray_frame)

    # magnitudes = np.sum(flow**2, axis=2) ** 0.5

    # mag_img = get_mag_img(small_frame, flow

    median_flow = flow[rel_to_abs(point, flow)]

    dx, dy = abs_to_rel(median_flow, flow, True)
    new_x = min(max(0, point[0] + dx), frame.shape[1] - 1)
    new_y = min(max(0, point[1] + dy), frame.shape[0] - 1)
    point = (new_x, new_y)

    if point[0] > 1 or point[0] <= 0 or point[1] > 1 or point[1] <= 0:
        point = (0.5, 0.5)

    prev_gray_frame = next_gray_frame

    cv.circle(frame, rel_to_abs(point, frame, True), 5, (0, 0, 255), -1)

    objects = detect_object(object_detector, frame)
    draw_objects(frame, objects)
    # draw_optical_flow(frame, flow)

    return frame


def generate_frame():
    global cap
    while True:
        try:
            success, frame = cap.read()
            if not success:
                break
            frame = proccess_img(frame)
            ret, buffer = cv.imencode(".jpg", frame)
            if not ret:
                break
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        except Exception as error:
            print(
                "ERROR in generate_frame:",
                str(error),
                "\nTraceback:\n" + "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            )
            break


def release_camera(signal, frame):
    global cap
    if cap:
        cap.release()
    print("release camara, exiting ...")
    exit(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


def video():
    global cap


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    signal.signal(signal.SIGINT, release_camera)
