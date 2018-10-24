#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:10:16 2018

@author: mrzhaocn
"""

import os
import numpy as np
import tensorflow as tf
import time
import datetime
#import model
import cv2
img_w = 136
img_h = 36
num_labels = 7
batch_size = 8
epoch = 1000
learning_rate = 0.0001
image_holder = tf.placeholder(tf.float32,[batch_size,img_h,img_w,3])
label_holder = tf.placeholder(tf.float32,[batch_size,7])
keep_prob = tf.placeholder(tf.float32)
logs_train_dir = './logs/'
source_path = './generatePlate/'
def get_batch(i,images,labels):
    batch_images = images[i:i*batch_size]
    batch_labels = labels[i:i*batch_size]
    return batch_images,batch_labels

train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7= model.inference(image_holder,keep_prob)

train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7 = model.losses(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder)
 
train_op1,train_op2,train_op3,train_op4,train_op5,train_op6,train_op7 = model.trainning(train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7,learning_rate)

train_acc = model.evaluation(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder)
input_image=tf.summary.image('input',image_holder)
#tf.summary.histogram('label',label_holder) #label的histogram,测试训练代码时用，参考:http://geek.csdn.net/news/detail/197155

summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

image_list = os.listdir(source_path)
images = []
labels = []
for file in image_list:
    image_soure = cv2.imread(source_path+file)
    if image_soure is None:
        continue
    if image_soure.any() == False:
        continue
    try:
        label = file[0:7]
        images.append(image_soure)
        labels.append(label)
    except:
        print("fileError")
train_datas = np.array(images)
train_labels = np.array(labels)

start_time1 = time.time()    
for step in range(epoch):
    start_time2 = time.time()   
    time_str = datetime.datetime.now().isoformat()
    batch = len(train_datas) / batch_size
    for i in range(batch):
        x_batch,y_batch = get_batch(i,train_datas,train_labels)
        feed_dict ={image_holder:x_batch,label_holder:y_batch,keep_prob:0.5}
        _,_,_,_,_,_,_,tra_loss1,tra_loss2,tra_loss3,tra_loss4,tra_loss5,tra_loss6,tra_loss7,acc,summary_str= sess.run([train_op1,train_op2,train_op3,train_op4,train_op5,train_op6,train_op7,train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7,train_acc,summary_op],feed_dict)
        train_writer.add_summary(summary_str,step)
        tra_all_loss =tra_loss1+tra_loss2+tra_loss3+tra_loss4+tra_loss5+tra_loss6+tra_loss7
    duration = time.time()-start_time2
    if step % 10== 0:
        sec_per_batch = float(duration)
        print("time %d",duration)
        print("loss %d",tra_all_loss)
    if step % 1000==0 or (step+1) == epoch:
        checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
        saver = tf.train.Saver()
        saver.save(sess,checkpoint_path,global_step=step)
sess.close()       
print(time.time()-start_time1)
    
    
    