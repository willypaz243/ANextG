# pylint: disable=no-member
import cv2 as cv
import numpy as np

from app import draw_optical_flow, get_edges_img, get_optical_flow

cap = cv.VideoCapture("Camino.mp4")


prev_gray_frame = None
point: tuple[int, int] | None = None


def on_mouse(event, x, y, _, __):
    global point
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)


flow = None

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (0, 0), fx=0.23, fy=0.23)

    if point is None:
        point = (frame.shape[1] // 2, frame.shape[0] // 2)

    if prev_gray_frame is None:
        prev_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    next_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edge_img = get_edges_img(next_gray_frame)
    flow = get_optical_flow(prev_gray_frame, next_gray_frame)

    magnitudes = np.sum(flow**2, axis=2) ** 0.5
    frame = draw_optical_flow(frame, flow)

    mag_img = np.zeros((frame.shape[0], frame.shape[1]), np.float32)
    mag_img += magnitudes
    mag_img = np.clip(mag_img, 0, 255).astype(np.uint8)

    mag_img = cv.cvtColor(mag_img, cv.COLOR_GRAY2BGR)
    mag_img = cv.normalize(
        mag_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )

    median_flow = flow[point[1], point[0]]

    dx, dy = median_flow
    x = min(max(round(point[0] + dx), 0), frame.shape[1] - 1)
    y = min(max(round(point[1] + dy), 0), frame.shape[0] - 1)
    point = (x, y)

    prev_gray_frame = next_gray_frame

    cv.circle(frame, point, 5, (0, 0, 255), -1)

    cv.imshow("optical_flow", frame)
    cv.imshow("magnitudes", mag_img)
    cv.imshow("edges", edge_img)

    cv.setMouseCallback("optical_flow", on_mouse)

    # listen events
    event = cv.waitKeyEx(1)
    if event == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
