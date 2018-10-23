#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:26:15 2018

@author: mrzhaocn
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import detect
import imageUtil
import svm_train
import os
from sklearn.externals import joblib #jbolib模块
imagePath = "./images/粤A5DP12.jpg"
n_components = 700
def predict(images):
    if len(images) <= 0:
        return
    dataArray = np.array(images)
    n_samples = dataArray.shape[0]
    data = dataArray.reshape((n_samples, -1))
    x_predict_pca = svm_train.dimensionalityPCA(n_components,data)
    if x_predict_pca.shape[1] != n_components:
        print("数据有问题")
        print(x_predict_pca.shape)
        return
    clf3 = joblib.load('save/svm.pkl')
    if clf3:
        predicted = clf3.predict(x_predict_pca)
    else:
        print("模型文件不存在")
    return predicted

if __name__ == "__main__":
    cropImages = detect.detectCarPlate(imagePath)
    resizeImage = imageUtil.resizeCarPlateImages(cropImages)
# =============================================================================
#     i = 0
#     for image in resizeImage:
#         i = i + 1
#         cv2.imwrite("./testCropImages/" + str(i) + ".jpg",image)
# =============================================================================
    predicted = predict(resizeImage)
    print(predicted)
      