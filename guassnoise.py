# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:21:26 2019

@author: wretched
"""
#对图像产生高斯噪声

import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

def gaussNoise(img, mu, sigma):
    # 首先创建一个新的图像用于存放处理后的结果
    img2 = np.zeros(img.shape, np.uint8)
    # 获取原始图像的宽高
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            # 依次循环图像中的每个像素，BGR三个通道分别处理
            
            
            img2[i,j]=img[i,j]+random.gauss(mu,sigma)
            # 判断处理后的值是否越界
            if img[i,j] > 255:
                img[i,j]= 255
            if img[i,j] < 0:
                img[i,j] = 0
          
    return img2

if __name__ == "__main__":
    images=[]
    for i in range(1,21):
        img=cv2.imread("F:\\DSP\pictures\%d_treatd1.bmp"%i,0)
        images.append(img)
    
    for i in range(1,21):
        print("第",i,"次开始")
        th1=gaussNoise(images[i-1],0,25)
        cv2.imwrite("F:\DSP\guassnoise\%d_guassnoise.bmp"%i,th1)
        print("第%d次结束"%i)
   
