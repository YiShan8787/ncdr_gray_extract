# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:18:00 2021

@author: user
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from tensorflow.keras.preprocessing import image

import cv2
from PIL import Image




gif = cv2.VideoCapture('dailysfc_20130502.gif')
ret,frame = gif.read() # ret=True if it finds a frame else False. Since your gif contains only one frame, the next read() will give you ret=False
img = Image.fromarray(frame)
img = img.convert('RGB')

y = 200
x = 100
y_len = 1000
x_len = 600

img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
gray =cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
gray = gray[y:y+y_len,:]
gray_invert = cv2.bitwise_not(gray)
cv2.imshow("img",gray)
cv2.waitKey()


kernel = np.ones((4, 4), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=3)

cv2.imshow("dil",dilation)
cv2.waitKey()


erosion  = cv2.erode(dilation, kernel, iterations=4)

cv2.imshow("erosion",erosion )
cv2.waitKey()

ret,thresh1 = cv2.threshold(erosion,127,255,cv2.THRESH_BINARY_INV)

cv2.imshow("thresh",thresh1)
cv2.waitKey()

# set my output img to zero everywhere except my mask
output_img = gray_invert.copy()
output_img[np.where(thresh1==0)] = 1
 

cv2.imshow('result', output_img)
cv2.waitKey()

# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=8)

# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
print('stats = ',stats)
# 连通域的中心点
print('centroids = ',centroids)
# 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
print('labels = ',labels)

# 不同的连通域赋予不同的颜色
output = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
for i in range(1, num_labels):

    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
cv2.imshow('oginal', output)
cv2.waitKey()


# Now finding Contours         ###################
'''
_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
coordinates = []
for cnt in contours:
        # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
    approx = cv2.approxPolyDP(
        cnt, 0.07 * cv2.arcLength(cnt, True), True)
    if len(approx) == 3:
        coordinates.append([cnt])
        cv2.drawContours(gray, [cnt], 0, (0, 0, 255), 3)

cv2.imwrite("result.png", gray)
'''


cv2.destroyAllWindows()

# 寫入圖檔
#out_path = "extract.jpg"
#cv2.imwrite(out_path, extract_img)