#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:13:54 2018
@author: mrzhaocn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#China car plate size: 440mm*140mm，aspect 3.142857 136,36
aspect = 3.142857
errorRate = 0.5
minArea = 44 * 14 * 0.4
maxArea = 44 * 14 * 10
rateMin = aspect - aspect * errorRate * errorRate #最小宽高比
rateMax = aspect + aspect * errorRate #最大宽高比
maxAngle = 20
minAngle = -20

def isUsefulContour(contour):
    rect = cv2.minAreaRect(contour)
    width,height = rect[1]
    if width <= 0 or height <= 0:
        return False,rect
    rate = float(width) / height
    angle = rect[2]
    if rate < 1 :
        rate = float(height) / width
        angle = angle + 90
    if (rate >= rateMin) and (rate <= rateMax):
        area = width * height
        if area < minArea or area > maxArea:
            return False,rect
        if angle >=0 and angle <= maxAngle:
            return True,rect
        
        if angle <=0 and angle >= minAngle:
            return True,rect
    return False,rect

def detectCarPlate(imagePath):
    img = cv2.imread(imagePath)
    #高斯平滑
    gaussion = cv2.GaussianBlur(img,(3,3),0,0,cv2.BORDER_DEFAULT)
    #中值滤波
    #median = cv2.medianBlur(gaussion,5)
    #灰度化
    grayImage = cv2.cvtColor(gaussion,cv2.COLOR_BGR2GRAY)
    #sobel算子
    x = cv2.Sobel(grayImage,cv2.CV_8U,1,0,ksize=3)  
    y = cv2.Sobel(grayImage,cv2.CV_8U,0,1,ksize=3)  
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX,0.6,absY,0.6,0)
    
    #二值化
    ret,binary = cv2.threshold(sobel,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(12,3))
    
    colose = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算
    # perform a series of erosions and dilations
    #closed = cv2.erode(closed, None, iterations=4)
    #closed = cv2.dilate(closed, None, iterations=4)
    #轮廓截取
    _ ,contours,_ = cv2.findContours(colose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    userfulContours = []
    cropImages = []
    i = 0
    for contour in contours: 
        userful,rect = isUsefulContour(contour)
        #box = np.int0(cv2.boxPoints(rect))
        #Box.append(box)
        if userful:
            i = i + 1
            userfulContours.append(contour)
            box = np.int0(cv2.boxPoints(rect))
            try:
                # step7：裁剪。box里保存的是绿色矩形区域四个顶点的坐标。我将按下图红色矩形所示裁剪昆虫图像。
                # 找出四个顶点的x，y坐标的最大最小值。新图像的高=maxY-minY，宽=maxX-minX。
                Xs = [i[0] for i in box]
                Ys = [i[1] for i in box]
                x1 = min(Xs)
                x2 = max(Xs)
                y1 = min(Ys)
                y2 = max(Ys)
                hight = y2 - y1
                width = x2 - x1
                cropImg = img[y1:y1+hight, x1:x1+width]
                cv2.imwrite("./testCropImages/"+ str(i) + '.jpg',cropImg)
                cropImages.append(cropImg)
                #Box.append(box)
            except:
                print("cropImageError")
    return cropImages
# =============================================================================
#     imag = cv2.drawContours(img.copy(),Box,-1,(0,0,255),1)
#     cv2.imshow("imag",imag)
#     cv2.waitKey(1000) 
# =============================================================================