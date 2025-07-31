import cv2
import numpy as np

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: {x}, {y}")

cv2.namedWindow("Test")
cv2.setMouseCallback("Test", on_mouse)

img = 255 * np.ones((400, 640, 3), dtype=np.uint8)

while True:
    cv2.imshow("Test", img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()