#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:14:05 2018

@author: mrzhaocn
"""

import numpy as np
import cv2
import os
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib #jbolib模块
carPlateImagePath = "./cropImagesTemp/"
carPlateWidth = 136
carPlateHeight = 36

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()
    
def getCarPlateImages():
     image_list = os.listdir(carPlateImagePath)
     #将数据打乱有利于测试准确率
     random.shuffle(image_list)
     images = []
     labels = []
     file_label = ""
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
                file_label = file_label + file + "0  \n"
                labels.append(0)
                i = i + 1
            else:
                file_label = file_label +  file + "1  \n"
                labels.append(1)
        except:
            print(filePath)
     dataArray = np.array(images)
     n_samples = dataArray.shape[0]

     data = dataArray.reshape((n_samples, -1))
     npLabels = np.array(labels)
     
     #将名称和对应的label写入文件
     save_to_file("fileLabels",file_label)
     print(i)
     return data,npLabels,image_list

def train():
    data,labels,image_list = getCarPlateImages()
    
# =============================================================================
#     print(image_list[2000:2010])
#     print(labels[2000:2010])
# =============================================================================
    
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    classifier = svm.SVC(gamma=0.001)#默认是rbf
    classifier.fit(x_train,y_train)
    print (u"训练集准确率:%.2f%%" % (classifier.score(x_train, y_train) * 100))
    print (u"测试集准确率:%.2f%%" % (classifier.score(x_test, y_test) * 100))
    
    #保存Model(注:save文件夹要预先建立，否则会报错)
    joblib.dump(classifier, './save/svm.pkl')
    
if __name__ == "__main__":
    getCarPlateImages()
    #train()
    #读取Model
# =============================================================================
#     clf3 = joblib.load('save/svm.pkl')
#     if clf3:
#         predicted = clf3.predict(x_test)
#     else:
# =============================================================================
        