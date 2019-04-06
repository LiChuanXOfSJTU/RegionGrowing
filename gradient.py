# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:17:59 2019

@author: Nile Lee

实现或测试Robert、Prewitt、Sobel梯度算子、Laplacian算子、高斯Laplacian算子、Canny算子。
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.cm as cm
###2*2算子
def Robert_gradient(img):
    x,y=img.shape
    Robb=[[-1,-1],[1,1]]
    new_img=np.zeros((x, y), dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            if(j+2<=y) and (i+2<=x):
                img_roi=img[i:i+2,j:j+2]
                list_robert=Robb*img_roi  
                new_img[i,j]=abs(list_robert[0][0]+list_robert[1][1])+abs(list_robert[0][1]+list_robert[1][0])
    new_img = new_img*(255.0/new_img.max())           
    return np.uint8(new_img)

def Prewitt_gradient(img):
    x,y=img.shape
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    new_img=np.zeros((x, y), dtype=np.uint8)
    prewitt_x=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    prewitt_y=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    for i in range(1,x+1):
        for j in range(1,y+1):
            img_roi=img[i-1:i+2,j-1:j+2]
            list_x=prewitt_x*img_roi
            list_y=prewitt_y*img_roi
            new_img[i-1,j-1]=abs(sum(list_x))+abs(sum(list_y))
    new_img = new_img*(255.0/new_img.max()) 
    return np.uint8(new_img)           
   
def Sobel_gradient(img):
    x,y=img.shape
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    new_img=np.zeros((x, y), dtype=np.uint8)
    sobel_x=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    for i in range(1,x+1):
        for j in range(1,y+1):
            img_roi=img[i-1:i+2,j-1:j+2]
            list_x=sobel_x*img_roi
            list_y=sobel_y*img_roi
            new_img[i-1,j-1]=abs(sum(list_x))+abs(sum(list_y))
    new_img = new_img*(255.0/new_img.max()) 
    return np.uint8(new_img)

def Laplace_gradient(img):
    x,y=img.shape
    img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    new_img=np.zeros((x, y), dtype=np.uint8)
    laplace=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    for i in range(1,x+1):
        for j in range(1,y+1):
            img_roi=img[i-1:i+2,j-1:j+2]
            
            list_lap=laplace*img_roi
            new_img[i-1,j-1]=abs(sum(list_lap))
    new_img = new_img*(255.0/new_img.max()) 
    return np.uint8(new_img)
   
def LOG(img):
    x,y=img.shape
    img=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_REPLICATE)
    new_img=np.zeros((x,y),dtype=np.uint8)
    log=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    for i in range(1,x+1):
        for j in range(1,y+1):
            img_roi=img[i-1:i+4,j-1:j+4]
            list_LOG=log*img_roi
            new_img[i-1,j-1]=abs(sum(list_LOG))
    new_img=new_img*(255.0/new_img.max())
    return np.uint8(new_img)

def canny(img):
    minVal=0
    maxVal=255
    result = cv2.Canny(img, minVal, maxVal)
    return result

if __name__ == "__main__":
    images=[]
    for i in range(1,21):
        img=cv2.imread("F:\\DSP\guassnoise\%d_guassnoise.bmp"%i,0)
        images.append(img)
    print(len(images))
    '''
    th1=Robert_gradient(images[1])
    cv2.imwrite("F:\\DSP\\robert\2.bmp",th1)
    '''
    for i in range(1,21):
        print("第",i,"次开始")
        th1=Laplace_gradient(images[i-1])
        cv2.imwrite("F:\DSP\Laplace\%d_Laplace_guassnoise.bmp"%i,th1)
        print("第%d次结束"%i)
   
    
    ''' 
    th2=Prewitt_gradient(img) 
    th3=Sobel_gradient(img)
    th4=Laplace_gradient(img)
    '''
    
    '''
    cv2.imshow("origin",img)
    cv2.imshow("robert",th1)
    cv2.imshow("prewitt",th2)
    cv2.imshow("sobel",th3)
    #cv2.imshow("laplace",th4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''








