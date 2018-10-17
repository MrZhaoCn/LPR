#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:25:18 2018

@author: mrzhaocn
"""

import cv2
import detect as dt
if __name__ == '__main__':
    path = './images/xianga.jpg'
    img = cv2.imread(path)
    img = dt.findContoursAndDrawBoundingBox(img)
    cv2.imshow('lena',img)
    k=cv2.waitKey(0)