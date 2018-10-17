#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:13:54 2018
@author: mrzhaocn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
imagePath = "./images/浙GZB388.jpg"
img = cv2.imread(imagePath)

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
cv2.imshow("gray",binary)

cv2.waitKey(1000)

