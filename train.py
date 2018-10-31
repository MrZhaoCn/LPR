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
import model
import random
import cv2
img_w = 136
img_h = 36
num_labels = 7
batch_size = 64
epoch = 100
learning_rate = 0.0005
decay_rate_one= 0.9
decay_rate_two=0.999
image_holder = tf.placeholder(tf.float32,[batch_size,img_h,img_w,3])
label_holder = tf.placeholder(tf.int32,[batch_size,7])
keep_prob = tf.placeholder(tf.float32)
logs_train_dir = './logs/'
log_file_name = './logs/log'
max_acc=0
source_path = './generatePlate/'

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64};
def get_batch (i, images, labels,files):
    batch_images = images[i*batch_size:(i + 1)*batch_size]
    batch_labels = labels[i*batch_size:(i + 1)*batch_size]
    batch_files = files[i * batch_size:(i + 1) * batch_size]
    return batch_images,batch_labels,batch_files
def save_to_file(file_name, contents):
    fh = open(file_name, 'a+')
    fh.write(contents)
    fh.close()
train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7= model.inference(image_holder,keep_prob)

# correct1 = tf.nn.in_top_k(train_logits1, label_holder, 1)
# accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
#
# correct2 = tf.nn.in_top_k(train_logits2, label_holder, 1)
# accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32))
#
# correct3 = tf.nn.in_top_k(train_logits3, label_holder, 1)
# accuracy3 = tf.reduce_mean(tf.cast(correct3, tf.float32))
#
# correct4 = tf.nn.in_top_k(train_logits4, label_holder, 1)
# accuracy4 = tf.reduce_mean(tf.cast(correct4, tf.float32))
#
# correct5 = tf.nn.in_top_k(train_logits5, label_holder, 1)
# accuracy5 = tf.reduce_mean(tf.cast(correct5, tf.float32))
#
# correct6 = tf.nn.in_top_k(train_logits6, label_holder, 1)
# accuracy6 = tf.reduce_mean(tf.cast(correct6, tf.float32))
#
# correct7 = tf.nn.in_top_k(train_logits7, label_holder, 1)
# accuracy7 = tf.reduce_mean(tf.cast(correct7, tf.float32))

train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7 = model.losses(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder)
train_op1,train_op2,train_op3,train_op4,train_op5,train_op6,train_op7 = model.trainning(train_loss1,train_loss2,train_loss3,train_loss4,train_loss5,train_loss6,train_loss7,learning_rate)

train_acc = model.evaluation(train_logits1,train_logits2,train_logits3,train_logits4,train_logits5,train_logits6,train_logits7,label_holder)

input_image=tf.summary.image('input',image_holder)
#tf.summary.histogram('label',label_holder) #label的histogram,测试训练代码时用，参考:http://geek.csdn.net/news/detail/197155

summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
image_list = os.listdir(source_path)
random.shuffle(image_list)
images = []
labels = []
files = []
for file in image_list:
    image_soure = cv2.imread(source_path+file)
    if image_soure is None:
        continue
    if image_soure.any() == False:
        continue
    try:
        label = file[0:7]
        indexs = []
        for i in range(7):
            labelIndex = index[label[i]]
            indexs.append(labelIndex)

        images.append(image_soure)
        labels.append(indexs)
        files.append(file)
    except:
        print("fileError",file)
train_datas = np.array(images)
train_labels = np.array(labels)

start_time1 = time.time()
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    ckpt = tf.train.get_checkpoint_state('./logs/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for step in range(epoch):
        start_time2 = time.time()
        batch = int(len(train_datas) / batch_size)
        file_label = ""
        for i in range(batch - 1):
            x_batch,y_batch,batch_file = get_batch(i,train_datas,train_labels,files)

            feed_dict = {image_holder: x_batch, label_holder: y_batch, keep_prob: 0.5}
            _, _, _, _, _, _, _, tra_loss1, tra_loss2, tra_loss3, tra_loss4, tra_loss5, tra_loss6, tra_loss7, acc, summary_str = sess.run(
                [train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7, train_loss1, train_loss2,
                 train_loss3, train_loss4, train_loss5, train_loss6, train_loss7, train_acc, summary_op], feed_dict)
            # accur1,accur2,accur3,accur4,accur5,accur6,accur7 = sess.run(accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7,feed_dict={label_holder: y_batch})
            train_writer.add_summary(summary_str, step)
            mean_loss = (tra_loss1 + tra_loss2 + tra_loss3 + tra_loss4 + tra_loss5 + tra_loss6 + tra_loss7) / 7
            # mean_accur = (accur1 + accur2 + accur3 + accur4 + accur5 + accur6 + accur7) / 7
            duration = time.time()-start_time2
            file_label = file_label + "tra_loss1:" + str(tra_loss1) + " tra_loss2:" + str(tra_loss2)+ " tra_loss3:" + str(tra_loss3)+ "tra_loss4:" + str(tra_loss4)+ "tra_loss5:" + str(tra_loss5)+ "tra_loss6:" + str(tra_loss6)+ "tra_loss7:" + str(tra_loss7) + "mean_loss:" + str(mean_loss) + "\n"
            file_label = file_label + str(duration) + "\n"

            print("tra_loss1:%g,tra_loss2:%g,tra_loss3:%g,tra_loss4:%g,tra_loss5:%g,tra_loss6:%g,tra_loss7:%g,mean_loss:%g"%(tra_loss1, tra_loss2,
                 tra_loss3, tra_loss4, tra_loss5, tra_loss6, tra_loss7,mean_loss))
            # print("accur1:%g,accur2:%g,accur3:%g,accu41:%g,accur5:%g,accur6:%g,accur7:%g,mean_accur:%g"%(
            #     accur1, accur2, accur3, accur4, accur5, accur6, accur7, mean_accur))


        print("step: %d,loss %g" % (step, mean_loss))
        file_label = file_label + "step:" + str(step) + "mean_loss:" + str(mean_loss)
        save_to_file(log_file_name, file_label)
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        # if (step + 1) % 10 == 0 or (step + 1) == epoch:

    print(time.time()-start_time1)