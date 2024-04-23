# pylint: disable=no-member
import os

import cv2 as cv

model_folder = "./ssd_mobilenet_v2_coco_2018_03_29"

model = "frozen_inference_graph.pb"
config = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classes = "coco_labels.txt"

with open(classes, "r", encoding="utf-8") as f:
    labels = [f.strip() for f in f.readlines()]


object_detector = cv.dnn.readNetFromTensorflow(
    os.path.join(model_folder, model), config
)


def detect_object(net, img):
    dim = 300
    blob = cv.dnn.blobFromImage(
        img,
        1.0,
        size=(dim, dim),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    objects = net.forward()
    return objects


def put_text(img, text, x, y):
    sizetext = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    dim = sizetext[0]
    baseline = sizetext[1]

    cv.rectangle(
        img,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (255, 0, 0),
        1,
    )
    cv.putText(
        img,
        text,
        (x, y - 5),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        1,
    )


def draw_objects(img: cv.Mat, objects: cv.Mat, threshold: float = 0.5):
    rows = img.shape[0]
    cols = img.shape[1]

    for i in range(objects.shape[2]):
        class_id = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        x = round(objects[0, 0, i, 3] * cols)
        y = round(objects[0, 0, i, 4] * rows)
        w = round(objects[0, 0, i, 5] * cols - x)
        h = round(objects[0, 0, i, 6] * rows - y)

        if score > threshold:
            put_text(img, labels[class_id], x, y)
            cv.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )


def loop(cap: cv.VideoCapture):
    ret, frame = cap.read()
    objects = detect_object(object_detector, frame)
    draw_objects(frame, objects)
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        return False
    return ret


def main():
    cap = cv.VideoCapture(0)
    monitoring = True
    while monitoring:
        monitoring = loop(cap)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
