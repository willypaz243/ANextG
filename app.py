# pylint: disable=no-member
import cv2 as cv
import numpy as np


def get_edges_img(gray_frame: cv.Mat) -> cv.Mat:
    assert len(gray_frame.shape) == 2
    img = cv.GaussianBlur(gray_frame, (3, 3), 0)
    v_img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    h_img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    img = cv.convertScaleAbs((h_img**2 + v_img**2) ** (0.5))
    img = cv.threshold(img, 200, 255, cv.THRESH_OTSU)[1]
    return img


def get_optical_flow(prv_gray: cv.Mat, nxt_gray: cv.Mat) -> cv.Mat:
    assert len(prv_gray.shape) == 2 and len(nxt_gray.shape) == 2
    flow = cv.calcOpticalFlowFarneback(
        prv_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow


def draw_optical_flow(img: cv.Mat, flow: cv.Mat) -> cv.Mat:
    step = 32
    for y in range(0, img.shape[0], step):
        for x in range(0, img.shape[1], step):
            pt1 = (x, y)
            dx, dy = flow[y, x]
            pt2 = (int(x + dx), int(y + dy))
            cv.arrowedLine(img, pt1, pt2, (0, 255, 0), 1)
    return img


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


def get_mag_img(small_img: cv.Mat, flow: cv.Mat) -> cv.Mat:
    magn_img = np.zeros((small_img.shape[0], small_img.shape[1]), np.float32)
    magn_img += magn_img
    magn_img = np.clip(magn_img, 0, 255).astype(np.uint8)
    magn_img = cv.cvtColor(magn_img, cv.COLOR_GRAY2BGR)
    magn_img = cv.normalize(
        magn_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    magn_img = draw_optical_flow(magn_img, flow)
    return magn_img
