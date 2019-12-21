import cv2


def draw_prediction(img, label, box):
    box = [int(p) for p in box]
    cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
    return cv2.putText(img, label, (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def show_image(winname, image, wait=True, window_size=(800, 600)):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, image)
    cv2.resizeWindow(winname, *window_size)
    if wait:
        cv2.waitKey(0)