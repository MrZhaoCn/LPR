#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:14:05 2018

@author: mrzhaocn
"""

import numpy as np
import cv2
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

carPlateImagePath = "./cropImagesTemp/"
carPlateWidth = 136
carPlateHeight = 36
def getCarPlateImages():
     image_list = os.listdir(carPlateImagePath)
     images = []
     labels = []
     i = 0
     for file in image_list:
        filePath = carPlateImagePath+file
        image_soure = cv2.imread(filePath)
        if image_soure is None:
            continue
        if image_soure.any() == False:
            continue
        try:
            images.append(image_soure)
            if "no" in file:
                labels.append(0)
            else:
                labels.append(1)
        except:
            print(filePath)
     dataArray = np.array(images)
     n_samples = dataArray.shape[0]
     print(dataArray.shape)
     data = dataArray.reshape((n_samples, -1))
     print(data.shape)
if __name__ == "__main__":
    getCarPlateImages()