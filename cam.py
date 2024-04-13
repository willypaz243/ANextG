# pylint: disable=no-member
import cv2 as cv

from app import draw_optical_flow, get_edges_img, get_optical_flow

cap = cv.VideoCapture(0)

prev_gray_frame = None

while True:
    ret, frame = cap.read()
    if prev_gray_frame is None:
        prev_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue
    next_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edge_img = get_edges_img(next_gray_frame)
    flow = get_optical_flow(prev_gray_frame, next_gray_frame)
    frame = draw_optical_flow(frame, flow)

    cv.imshow("optical_flow", frame)
    cv.imshow("edges", edge_img)

    prev_gray_frame = next_gray_frame
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
