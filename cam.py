# pylint: disable=no-member
import cv2 as cv
import numpy as np

from app import *
from main import detect_object, object_detector

cap = cv.VideoCapture(0)


prev_gray_frame = None
point: tuple[float, float] | None = None


while True:
    ret, frame = cap.read()
    if not ret:
        continue
    scalar = 0.5
    small_frame = cv.resize(frame, (0, 0), fx=scalar, fy=scalar)
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if point is None:
        point = (0.5, 0.5)

    if prev_gray_frame is None:
        prev_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
        continue

    next_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    edge_img = get_edges_img(next_gray_frame)
    flow = get_optical_flow(prev_gray_frame, next_gray_frame)

    magnitudes = np.sum(flow**2, axis=2) ** 0.5

    mag_img = get_mag_img(small_frame, flow)

    median_flow = flow[rel_to_abs(point, flow)]

    dx, dy = abs_to_rel(median_flow, flow, True)
    new_x = min(max(0, point[0] + dx), frame.shape[1] - 1)
    new_y = min(max(0, point[1] + dy), frame.shape[0] - 1)
    point = (new_x, new_y)

    # point = point + median_flow[:-1] / flow.shape[:2]

    prev_gray_frame = next_gray_frame

    # cv.circle(frame, rel_to_abs(point, frame, True), 5, (0, 0, 255), -1)

    objects = detect_object(object_detector, frame)
    draw_objects(frame, objects)

    cv.imshow("optical_flow", frame)
    cv.imshow("magnitudes", mag_img)
    cv.imshow("edges", edge_img)
    cv.imshow("frame", small_frame)

    cv.setMouseCallback("optical_flow", on_mouse, frame)

    # listen events
    key = cv.waitKeyEx(1)
    if key == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
