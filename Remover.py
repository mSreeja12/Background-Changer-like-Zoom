import cv2
import cvzone 
import cvzone.FPS
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os 
from cvzone.FPS import FPS

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,360)
segmentor = SelfiSegmentation(model=1)
fpsReader = FPS()
imgBg = cv2.imread(".venv/These Cozy & Chic Zoom Home Backgrounds Are Like Virtual Makeovers For Your Space (2).jpeg")

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgBg,cutThreshold=0.3)

    Stacked = cvzone.stackImages([img,imgOut],2,1)
    p, Stacked = fpsReader.update(Stacked)
    cv2.imshow("Image",Stacked)
    
    cv2.waitKey(1)