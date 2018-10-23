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
import detect
import imageUtil
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib #jbolib模块
carPlateImagePath = "./testCropImagesTemp/"
imagePath = "./images/湘G60009.jpg"
carPlateWidth = 136
carPlateHeight = 36
n_components = 700

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()
#
def getCarPlateImages(path):
     image_list = os.listdir(path)
     #将数据打乱有利于测试准确率
     random.shuffle(image_list)
     images = []
     labels = []
     file_label = ""
     files = []
     for file in image_list:
        filePath = path+file
        image_soure = cv2.imread(filePath)
        if image_soure is None:
            continue
        if image_soure.any() == False:
            continue
        try:
            #image_soure = cv2.cvtColor(image_soure,cv2.COLOR_BGR2GRAY)
            images.append(image_soure)
            files.append(file)
            if "no" in file:
                file_label = file_label + file + "0  \n"
                labels.append(0)
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
     print(data.shape)
     return data,npLabels,files

def train():
    data,labels,files = getCarPlateImages(carPlateImagePath)
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    #降维，除了使用PCA外，还有一种思路是利用CNN抽取特征做分类
    pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    #通过网格交叉验证，最优参数为{'C': 1000.0, 'gamma': 0.001}，准确率分贝为：训练集准确率:100.00%
     #测试集准确率:97.54%
    #classifier = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier = SVC(C=1000.0,gamma=0.001,kernel='rbf', class_weight='balanced')#默认是rbf
    classifier.fit(x_train_pca, y_train)
    
    print (u"训练集准确率:%.2f%%" % (classifier.score(x_train_pca, y_train) * 100))
    print (u"测试集准确率:%.2f%%" % (classifier.score(x_test_pca, y_test) * 100))
    
    #保存Model(注:save文件夹要预先建立，否则会报错)
    joblib.dump(classifier, './save/svm.pkl')
def dimensionalityPCA(n_components,x_predict):
    #降维，除了使用PCA外，还有一种思路是利用CNN抽取特征做分类
    pca = PCA(n_components=700, svd_solver='randomized',
          whiten=True).fit(x_predict)
    x_predict_pca = pca.transform(x_predict)
    return x_predict_pca
def predict(clf3): 
    #cropImages = detect.detectCarPlate(imagePath)
    
    #resizeImage = imageUtil.resizeCarPlateImage()
    data,labels,_ = getCarPlateImages(carPlateImagePath)
    #降维，除了使用PCA外，还有一种思路是利用CNN抽取特征做分类
    x_train_pca = dimensionalityPCA(700,data)
    predicted = clf3.predict(x_train_pca)
    return predicted,labels
     
if __name__ == "__main__":
    #train()

    #读取Model
    clf3 = joblib.load('save/svm.pkl')
    if clf3:
         predicted,labels = predict(clf3)
         print(predicted[:20])
         print(labels[:20])
    else:
        train()