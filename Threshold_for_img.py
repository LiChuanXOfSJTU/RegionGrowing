# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:33:36 2019

@author: Nile Lee

for digital image processing course
"""
#实现或测试至少二种阈值分割方法，观察二者的异同。观察复杂图像与简单图像的阈值分割效果。
import cv2
import matplotlib.pyplot as plt
import numpy as np
#读入灰度图
images=[]
for i in range(1,21):
    img=cv2.imread("F:\\DSP\Sobel\%d_sobel.bmp"%i,0)
    images.append(img)
# 阈值分割
print(len(images))

#固定阈值分割

for i in range(1,21):
    ret1, th1 = cv2.threshold(images[i-1], 85, 255, cv2.THRESH_BINARY)
    cv2.imwrite("F:\\DSP\Global(v=127)\%d_sobel_global127.bmp"%i,th1)
  

#自适应阈值
'''
参数1：要处理的原图
参数2：最大阈值，一般为255
参数3：小区域阈值的计算方式
ADAPTIVE_THRESH_MEAN_C：小区域内取均值
ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
参数4：阈值方式（跟前面讲的那5种相同）
参数5：小区域的面积，如11就是11*11的小块
参数6：最终阈值等于小区域计算出的阈值再减去此值
'''
'''

for i in range(1,21):
    th2=cv2.adaptiveThreshold(
    images[i-1], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
    th3 = cv2.adaptiveThreshold(
    images[i-1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 6)
    cv2.imwrite("F:\\DSP\Adaptive Mean\%d.bmp"%i,th2)
    cv2.imwrite("F:\\DSP\Adaptive Gaussian\%d.bmp"%i,th3)

#Ostu阈值
for i in range(1,21):
    ret4, th4 = cv2.threshold(images[i-1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Ostu阈值:",ret4)
    cv2.imwrite("F:\\DSP\Ostu\%d.bmp"%i,th4)
    


'''
'''
titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian','Ostu']
images = [img, th1, th2, th3,th4]


cv2.imshow('Original',img)
cv2.imshow("Global(v = 127)",th1)
cv2.imshow("Adaptive Mean",th2)
cv2.imshow("Adaptive Gaussian",th3)
cv2.imshow("Ostu",th4)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''