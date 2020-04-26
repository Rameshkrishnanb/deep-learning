#!/usr/bin/env python
import numpy as np
import cv2
cars=cv2.CascadeClassifier('cars.xml')
image=cv2.imread('cars.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

dcars =cars.detectMultiScale(gray)
print(len(dcars))
for (x,y,w,h) in dcars:
	cv2.rectangle(image,(x,y),(x+w,y+h),(250,0,0),2)
cv2.imshow('dst',image)
print(gray.shape)
cv2.waitKey(0)

