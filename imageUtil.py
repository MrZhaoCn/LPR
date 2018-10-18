#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:19:25 2018

@author: mrzhaocn
"""

import cv2
import os
import detect
image_size = 900
source_path = "./images/"
target_path = "./change_size/"

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

def detectCarPlates():
    image_list = os.listdir(target_path)
    for file in image_list:
      detectCarPlate(target_path+file)
      print(target_path+file)
        
if __name__ == "__main__":
    detectCarPlates()
    
    
            
        
