# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:18:00 2021

@author: user
"""
import configparser
import json
config = configparser.ConfigParser()    # 注意大小寫
config.read("config.ini")   # 配置檔案的路徑

weather_path = config['main']['origin_path']
extract_path = config['main']['destination_path']
suffix = config['main']['suffix']
image_type = config['main']['image_type']

y =int( config['main']['cut_y'])
x = int(config['main']['cut_x'])
y_len = int(config['main']['y_len'])
x_len = int(config['main']['x_len'])

lonRange = json.loads(config['main']['lonRange'])
latRange = json.loads(config['main']['latRange'])


import os

work_path = os.getcwd()

def mkdir(create_path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(create_path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(create_path)
        print('-----建立成功-----')

    else:
        #如果目錄已存在，則不建立，提示目錄已存在
        print(create_path+'目錄已存在')

import numpy as np
import cv2
from PIL import Image


mkdir(extract_path)


for file in os.listdir(weather_path):
    #path = "scc201606090000.jpg"
    dir_path = os.path.abspath(weather_path)
    path = dir_path + "\\" + file
    print(path)
    
    # 讀取圖檔
    #img = cv2.imread(path)
    #img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gif = cv2.VideoCapture(path)
    ret,frame = gif.read() # ret=True if it finds a frame else False. Since your gif contains only one frame, the next read() will give you ret=False
    img = Image.fromarray(frame)
    img = img.convert('RGB')
    
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray =cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    gray = gray[y:y+y_len,:]
    gray_invert = cv2.bitwise_not(gray)
    #cv2.imshow("img",gray)
    cv2.waitKey()
    
    
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=3)
    
    #cv2.imshow("dil",dilation)
    cv2.waitKey()
    
    
    erosion  = cv2.erode(dilation, kernel, iterations=4)
    
    #cv2.imshow("erosion",erosion )
    cv2.waitKey()
    
    ret,thresh1 = cv2.threshold(erosion,50,255,cv2.THRESH_BINARY_INV)
    
    #cv2.imshow("thresh",thresh1)
    cv2.waitKey()
    
    # set my output img to zero everywhere except my mask
    output_img = gray_invert.copy()
    output_img[np.where(thresh1==0)] = 1
     
    
    #cv2.imshow('result', output_img)
    cv2.waitKey()
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=8)
    
    #cv2.imshow('extract', extract_img)
    #cv2.waitKey()
    
    from numpy import interp

    
    
    # the range of y and x pixels
    yRange = [0, gray.shape[0]]
    xRange = [0, gray.shape[1]]
    gray_three_channel = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    
    for k in range(centroids.shape[0]-1):
        xPixel = centroids[k+1][0]
        yPixel = centroids[k+1][1]
        
        lat = latRange[1] - interp(yPixel, yRange, latRange) # flipped again
        lon = interp(xPixel, xRange, lonRange)
        
        #origin_cmp = cv2.drawContours(gray_invert.copy(),[box],0,(0,0,255),5)
        
        text2 = '(' + str(format(lon,'.2f')) + ', ' + str(format(lat,'.2f')) + ')'
        
        cv2.putText(gray_three_channel, text2, (int(xPixel), int(yPixel)), cv2.FONT_HERSHEY_TRIPLEX,
          0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    #cv2.imshow('origin_cmp', gray_three_channel)
    cv2.waitKey()
    out_path = extract_path + '\\' + file.split('.')[0] + suffix +'.' +image_type
    print(out_path)
    cv2.imwrite(out_path, gray_three_channel)
    
    cv2.destroyAllWindows()
    
    # 寫入圖檔
    #out_path = extract_path + '\\' + file.split('.')[0] + suffix +'.' +image_type
    #print(out_path)
    #cv2.imwrite(out_path, extract_img)