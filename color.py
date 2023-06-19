import cv2
import numpy as np

def empty(a):
    pass

# Create a window
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty) # Hue range is 0-179
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty) # Saturation range is 0-255
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty) # Value range is 0-255
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:


    
    img = cv2.imread('resources/lambo.webp')
    img = cv2.resize(img, (640, 480))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars") # Get the value of the trackbar
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min]) # Create a numpy array for the lower bound
    upper = np.array([h_max, s_max, v_max]) # Create a numpy array for the upper bound
    mask = cv2.inRange(imgHSV, lower, upper) # Create a mask
    imgResult = cv2.bitwise_and(img, img, mask=mask) # Apply the mask to the image
    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    cv2.waitKey(1)
    

