# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:54:35 2019

@author: Nile Lee
实现或测试区域生长算法，理解自递归的实现；观察不同阈值下的生长结果变化。观察复杂图像与简单图像的分割效果。
"""


import numpy as np
import cv2
 
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects():
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]#8邻域
   
        
    return connects
 
def regionGrow(img,seeds,thresh):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects()
    
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark
 
if __name__ == '__main__':
    
    img = cv2.imread('F://DSP/pictures/6_treatd1.bmp',0)
    ret1, th1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    seeds = [Point(10,10)]
    binaryImg = regionGrow(th1,seeds,10)
    cv2.imshow(' ',binaryImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    













