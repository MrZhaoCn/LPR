#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:19:25 2018

@author: mrzhaocn
"""

import cv2
import os
#import detect

image_size = 900
source_path = "./images/"
target_path = "./change_size/"

carPlateImagePath = "./testCropImages/"
targetCarPlateImage = "./testCropImagesTemp/"
carPlateWidth = 136
carPlateHeight = 36

def resizeImages():
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    image_list = os.listdir(source_path)
    i = 0
    for file in image_list:
        i = i+ 1
        image_soure = cv2.imread(source_path+file)
        if image_soure is None:
            continue
        if image_soure.any() == False:
            continue
        (height,width) = image_soure.shape[:2]
        
        if width > image_size:
            rate = float(width) / height
            resize_height =  int(image_size / rate)
            image = cv2.resize(image_soure,(image_size,resize_height),0,0,cv2.INTER_LINEAR)
            cv2.imwrite(target_path+str(i)+".jpg",image)
            
def resizeCarPlateImage(): 
    if not os.path.exists(targetCarPlateImage):
        os.makedirs(targetCarPlateImage)
    image_list = os.listdir(carPlateImagePath)
    
    for file in image_list:
        filePath = carPlateImagePath+file
        tartgetFilePath = targetCarPlateImage+file
        image_soure = cv2.imread(filePath)
        if image_soure is None:
            continue
        if image_soure.any() == False:
            continue
        try:
            image = cv2.resize(image_soure,(carPlateWidth,carPlateHeight),0,0,cv2.INTER_LINEAR)
            cv2.imwrite(tartgetFilePath,image)
        except:
            print(tartgetFilePath)
            
def resizeCarPlateImages(images): 
    resizeImages = []
    for image in images:
        try:
            image_soure = cv2.resize(image,(carPlateWidth,carPlateHeight),0,0,cv2.INTER_LINEAR)
            resizeImages.append(image_soure)
        except:
            print("resizeImageError")      
    return resizeImages
            
def detectCarPlates():
    image_list = os.listdir(target_path)
    for file in image_list:
      #detectCarPlate(target_path+file)
      print(target_path+file)
        
if __name__ == "__main__":
    #detectCarPlates()
    resizeCarPlateImage()
    
    
            
        
