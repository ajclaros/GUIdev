import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


# Create a black img, a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow("img")

# create trackbars for color change
cv2.createTrackbar("H Lower", "img", 0, 255, nothing)
cv2.createTrackbar("H Higher", "img", 0, 255, nothing)
cv2.createTrackbar("S Lower", "img", 0, 255, nothing)
cv2.createTrackbar("S Lower", "img", 0, 255, nothing)
cv2.createTrackbar("V Lower", "img", 0, 255, nothing)
cv2.createTrackbar("V Higher", "img", 0, 255, nothing)
switch = "0 : OFF \n1 : ON"
cv2.createTrackbar(switch, "img", 0, 1, nothing)

while 1:
    cv2.imshow("img", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    hL = cv2.getTrackbarPos("H Lower", "img")
    hH = cv2.getTrackbarPos("H Higher", "img")
    sL = cv2.getTrackbarPos("S Lower", "img")
    sH = cv2.getTrackbarPos("S Higher", "img")
    vL = cv2.getTrackbarPos("V Lower", "img")
    vH = cv2.getTrackbarPos("V Higher", "img")

    lowerRegion = np.array([hL, sL, vL], np.uint8)
    upperRegion = np.array([hH, sH, vH], np.uint8)

    redObject = cv2.inRange(hsv, LowerRegion, upperRegion)
    kernal = np.ones((1, 1), "uint8")

    red = cv2.morphologyEx(redObject, cv2.MORPH_OPEN, kernal)
    red = cv2.dilate(red, kernal, iterations=1)

    res1 = cv2.bitwise_and(img, img, mask=red)

    s = cv2.getTrackbarPos(switch, "img")
    cv2.imshow("Masking ", res1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

cv2.destroyAllWindows()
