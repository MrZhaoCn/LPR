#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:13:54 2018
@author: mrzhaocn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
imagePath = "./change_size/2.jpg"
img = cv2.imread(imagePath)

#China car plate size: 440mm*140mm，aspect 3.142857
aspect = 3.142857
errorRate = 0.2
minArea = 44 * 14 * 0.2
maxArea = 44 * 14 * 10
rateMin = aspect - aspect * errorRate#最小宽高比
rateMax = aspect + aspect * errorRate #最大宽高比



def isUsefulContour(contour):
    rect = cv2.minAreaRect(contour)
    width,height = rect[1]
    if width <= 0 or height <= 0:
        return False,rect
    rate = float(width) / height
    if rate < 1 :
        rate = float(height) / width
    if (rate >= rateMin) and (rate <= rateMax):
        area = width * height
        if area < minArea or area > maxArea:
            return False,rect
        angle = rect[2]
        if angle > -30:
            print(rect[2])
            return True,rect
    return False,rect
    
#高斯平滑
gaussion = cv2.GaussianBlur(img,(5,5),0,0,cv2.BORDER_DEFAULT)
#中值滤波
#median = cv2.medianBlur(gaussion,5)
#灰度化
grayImage = cv2.cvtColor(gaussion,cv2.COLOR_BGR2GRAY)
#sobel算子
x = cv2.Sobel(grayImage,cv2.CV_8U,1,0,ksize=3)  
y = cv2.Sobel(grayImage,cv2.CV_8U,0,1,ksize=3)  
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
sobel = cv2.addWeighted(absX,0.5,absY,0.5,0)

#二值化
ret,binary = cv2.threshold(sobel,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU )

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,3))

colose = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算

#轮廓截取
_ ,contours,_ = cv2.findContours(colose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
userfulContours = []
Box = []
for contour in contours: 
    userful,rect = isUsefulContour(contour)
    box = np.int0(cv2.boxPoints(rect))
    Box.append(box)
    if userful:
        userfulContours.append(contour)
        #box = np.int0(cv2.boxPoints(rect))
        #Box.append(box)
        
imag = cv2.drawContours(img.copy(),Box,-1,(0,0,255),1)
cv2.imshow("imag",imag)
cv2.waitKey(1000) 
