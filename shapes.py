import cv2
import numpy as np

def getContours(img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find the contours
    for cnt in countours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3) # Draw the contours
            cevre = cv2.arcLength(cnt, True) # Calculate the perimeter
            print(cevre)
            approx = cv2.approxPolyDP(cnt, 0.02*cevre, True) # Approximate the shape
            print(len(approx))
            nesneKose = len(approx)
            x, y, w, h = cv2.boundingRect(approx) # Create a bounding box
            if nesneKose == 3: # If the shape has 3 corners
                objectType = "Ucgen"
            elif nesneKose == 4: # If the shape has 4 corners
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05: # If the shape is a square
                    objectType = "Kare"
                else: # If the shape is a rectangle
                    objectType = "Dikdortgen"
            elif nesneKose > 4: # If the shape has more than 4 corners
                objectType = "Cember"
            else:
                objectType = "Tanimlanamadi"
            cv2.rectangle(imgContour, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2) # Draw the bounding box
            cv2.putText(imgContour, objectType, (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # Put the text on the image


img = cv2.imread('resources/shapes.jpeg')
img = cv2.resize(img, (640, 480))
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgBlank = np.zeros_like(img)
getContours(imgCanny)


cv2.imshow('Original', img)
cv2.imshow('Gray', imgGray)
cv2.imshow('Blur', imgBlur)
cv2.imshow('Canny', imgCanny)
cv2.imshow('Contour', imgContour)
cv2.waitKey(0)