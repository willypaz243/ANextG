# pylint: disable=no-member
import cv2 as cv


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
