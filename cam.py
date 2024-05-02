# pylint: disable=no-member
import cv2 as cv
import numpy as np

from app import draw_optical_flow, get_edges_img, get_optical_flow
from main import detect_object, draw_objects, object_detector

cap = cv.VideoCapture(0)


prev_gray_frame = None
point: tuple[int, int] | None = None


def on_mouse(event, x, y, _, img: cv.Mat):
    global point  # pylint: disable=global-statement
    assert len(img.shape) >= 2
    if event != cv.EVENT_LBUTTONDOWN:
        return
    y, x = abs_to_rel((x, y), img)
    point = (x, y)


def rel_to_abs(pt: np.ndarray | tuple, mat: cv.Mat, reverse: bool = False):
    assert len(pt) == 2
    x = min(round(pt[0] * mat.shape[1]), mat.shape[1] - 1)
    y = min(round(pt[1] * mat.shape[0]), mat.shape[0] - 1)
    if reverse:
        return x, y
    return y, x


def abs_to_rel(pt: np.ndarray | tuple, mat: cv.Mat, reverse: bool = False):
    assert len(pt) == 2
    x = pt[0] / mat.shape[1]
    y = pt[1] / mat.shape[0]
    if reverse:
        return x, y
    return y, x


flow = None


def get_mag_img(small_img: cv.Mat) -> cv.Mat:
    magn_img = np.zeros((small_img.shape[0], small_img.shape[1]), np.float32)
    magn_img += magn_img
    magn_img = np.clip(magn_img, 0, 255).astype(np.uint8)
    magn_img = cv.cvtColor(magn_img, cv.COLOR_GRAY2BGR)
    magn_img = cv.normalize(
        magn_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    magn_img = draw_optical_flow(magn_img, flow)
    return magn_img


while True:
    ret, frame = cap.read()
    if not ret:
        continue
    scalar = 0.5
    small_frame = cv.resize(frame, (0, 0), fx=scalar, fy=scalar)
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if point is None:
        point = np.array((0.5, 0.5))

    if prev_gray_frame is None:
        prev_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
        continue

    next_gray_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    edge_img = get_edges_img(next_gray_frame)
    flow = get_optical_flow(prev_gray_frame, next_gray_frame)

    magnitudes = np.sum(flow**2, axis=2) ** 0.5

    mag_img = get_mag_img(small_frame)

    median_flow = flow[rel_to_abs(point, flow)]

    dx, dy = abs_to_rel(median_flow, flow, True)
    new_x = min(max(0, point[0] + dx), frame.shape[1] - 1)
    new_y = min(max(0, point[1] + dy), frame.shape[0] - 1)
    point = (new_x, new_y)

    # point = point + median_flow[:-1] / flow.shape[:2]

    prev_gray_frame = next_gray_frame

    cv.circle(frame, rel_to_abs(point, frame, True), 5, (0, 0, 255), -1)

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
