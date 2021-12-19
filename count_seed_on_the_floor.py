# Objective: To determine the number of seed in the corn cross-section

import os
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import math
import matplotlib.pyplot as plt

path = r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\11.09 -- Count CORN\imageDataset\6f8a1343-01ab-4e1c-9d8f-2487fb15c458.jpg"

I = cv2.imread(path)
M = I.shape[0]
N = I.shape[1]
r = 0.8
I = cv2.resize(I,[int(r*N), int(r*M)])
I = I[300:650, 310:700,:]
M = I.shape[0]
N = I.shape[1]

aSeed = I[110:150,0:30,:]
cv2.imshow("I",I)
cv2.imshow("A seed",aSeed)

# LAB = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
# HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# # LABDisp = np.hstack(LAB[:,:,0],LAB[:,:,1])
# # LABDisp = np.hstack(LABDisp,LAB[:,:,2])

# print(LAB.shape)

# # cv2.imshow("L",LAB[:,:,0])
# # cv2.imshow("A",LAB[:,:,1])
# # cv2.imshow("B",LAB[:,:,2])

# # cv2.imshow("H",HSV[:,:,0])
# # cv2.imshow("S",HSV[:,:,1])
# # cv2.imshow("V",HSV[:,:,2])
# V = HSV[:,:,2]
# ksize = (5, 5)
# # V = cv2.blur(V,ksize)
# V = cv2.GaussianBlur(V, (3,3), 0)




# ret, th1 = cv2.threshold(V, 120, 255, cv2.THRESH_BINARY)



# kernel = np.ones((3,3),np.uint8)
# solidSeg = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)

# solidSeg_RGB = np.zeros((M,N,3), dtype=np.uint8)
# for k in range(3):
#     solidSeg_RGB[:,:,k] = solidSeg

# cornCroSecEdge = cv2.Canny(solidSeg, 100, 150)
# sumI = 0
# sumJ = 0
# numPix = 0
# for i in range(M):
#     for j in range(N):
#         if cornCroSecEdge[i,j] > 0:
#             sumI +=i
#             sumJ +=j
#             numPix +=1

# iCent = int(sumI/numPix)
# jCent = int(sumJ/numPix)
# cornCroSecEdge_RGB = np.zeros((M,N,3), dtype=np.uint8)
# for k in range(3):
#     cornCroSecEdge_RGB[:,:,k] = cornCroSecEdge

# for k in range(-5,5,1):
#     for l in range(-5,5,1):
#         cornCroSecEdge_RGB[iCent+k, jCent+l, 0] = 0
#         cornCroSecEdge_RGB[iCent+k, jCent+l, 1] = 255
#         cornCroSecEdge_RGB[iCent+k, jCent+l, 2] = 255



# cv2.imshow("th1",th1)
# cv2.imshow("SolidSeg",solidSeg_RGB)
# cv2.imshow("cornCSEdge",cornCroSecEdge_RGB)

# # =======================================
# RMax = 200


# def centroidOfTheSeed(img):
#     M = img.shape[0]
#     N = img.shape[1]

#     sumI = 0
#     sumJ = 0
#     NPix = 0

#     iArr =[]
#     jArr =[]
#     RArr =[]
    
#     for i in range(M):
#         for j in range(N):
#             if img[i,j]>127:
#                 sumI += i
#                 sumJ += j
#                 NPix += 1

#                 #Rval = np.sqrt(i**2 + j**2)

#                 iArr.append(i)
#                 jArr.append(j)
#                 #RArr.append(Rval)                
    
#     cent = [0,0]
#     cent[0] = sumI/NPix
#     cent[1] = sumJ/NPix

#     return cent, iArr, jArr, RArr


# iAvg = sum(cornCroSecEdge)
# [cent, iArr, jArr, RArr] = centroidOfTheSeed(cornCroSecEdge)

# print(cent)

# NArr = len(iArr)
# print(NArr)


# deltaArr =[]

# for i in range(NArr):
#     dI = iArr[i] - cent[0]
#     dJ = jArr[i] - cent[1]

#     delta = np.arctan(abs(dI)/abs(dJ))*180/(math.pi)

#     if dI<0:
#         if dJ>=0:
#             delta = delta
#         if dJ<0:
#             delta = 180-delta

#     if dI>0:
#         if dJ<0:
#             delta = 180 + delta
#         if dJ>=0:
#             delta = 360 - delta
    
#     Rval = np.sqrt(dI**2 + dJ**2)
#     Rval = Rval/RMax
    
#     RArr.append(Rval)    
#     deltaArr.append(delta)


# plt.figure(figsize=(12, 6))

# x = deltaArr
# y = RArr
# z = np.polyfit(x, y, 100)

# print(z)

# # xp = np.linspace(0, 360, 100)
# # p = np.poly1d(z)

# # _ = plt.plot(x, y, '.', xp, p(xp), '-')

# #yMax = max(edge.shape[0], edge.shape[1])
# yMax = 1
# N = cornCroSecEdge.shape[1]//2
# plt.xlim(0,360)
# plt.ylim(0,yMax)

# plt.scatter(deltaArr, RArr, s=0.5)
# plt.show()

# os.system("cls")
# print(len(x))
# print(len(y))























cv2.waitKey(0)
cv2.destroyAllWindows()