import cv2
import os
import numpy as np
import random

os.system("cls")

fullPath = r"C:\Users\Esa\Documents\a1ocvCodes\uav_corn\malai.jpg"
# fullPath = r"C:\Users\Esa\Documents\a1ocvCodes\uav_corn\samp.png"
img0 = cv2.imread(fullPath)
# ratio = 1

M0 = img0.shape[0]
N0 = img0.shape[1]

M = 600
N = 500
img = np.zeros((M, N, 3), dtype=np.uint8)

i0 = 478 #(random.randint(0, M0-M))
j0 = 2601 #(random.randint(0, N0-N))

for i in range(M):
    for j in range(N):
        for k in range(3):
            img[i,j,k] = img0[i0+i, j0+j, k]

# cv2.imshow("Original RGB", img)



# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
# L,A,B=cv2.split(lab)

blueImg = img[:,:,1]

cv2.imshow("Blue Ori", blueImg)


ksize = (3, 3)
blueImg = cv2.blur(blueImg, ksize) 
blueImg = cv2.blur(blueImg, ksize) 
ret,thresh1 = cv2.threshold(blueImg,120,255,cv2.THRESH_BINARY)

rgb_BW = np.zeros((thresh1.shape[0], thresh1.shape[1],3), dtype=np.uint8)
for i in range(thresh1.shape[0]):
    for j in range(thresh1.shape[1]):
        for k in range(3):
            rgb_BW[i,j,k] = thresh1[i,j]

# cv2.imshow("Segmented BW", rgb_BW)




img2 = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8)
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if thresh1[i,j] == 255:
            for k in range(3):
                img2[i,j,k] = img[i,j,k]


finalIMG = np.hstack((img, rgb_BW))
finalIMG = np.hstack((finalIMG, img2))




cv2.imshow("i:"+str(i0)+" j:"+str(j0), finalIMG)
cv2.imshow("Blue Blur", blueImg)



cv2.waitKey(0)
cv2.destroyAllWindows()