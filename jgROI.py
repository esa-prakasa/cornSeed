import numpy as np
import pandas as pd
import cv2
import random
import math

 
# Load an color image in grayscale
#img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\IMG_5011.jpg',1)
#img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\IMG_5020.#jpg',1)
#img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\01.jpg',1)
#img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\02.jpg',1)
#img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\03.jpg',1)
img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\IMG_5023.jpg',1)







print('Original Dimensions : ',img.shape)
height = img.shape[0]
width = img.shape[1]

print('Height: '+str(height)+' Width: '+str(width))

scale = 0.1
dst = cv2.resize(img,None,fx=scale,fy=scale)

hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
print(hsv.shape)

#side = 300
#ic = math.floor(0.5*height)
#jc = math.floor(0.5*width)
#dst = img[1000:1400,1250:1650,:]
#dst = cv2.resize(img,None,fx=0.1,fy=0.1)


#cv2.imshow('Resized image',dst)
#cv2.imshow('HSV image 0',hsv[:,:,0])
#cv2.imshow('HSV image 1',hsv[:,:,1])
#cv2.imshow('HSV image 2',hsv[:,:,2])


gray = hsv[:,:,2]
ret, bwImg = cv2.threshold(gray,100,255,cv2. THRESH_BINARY)
#cv2. imshow("Binary Image",bwImg)

height = hsv.shape[0]
width = hsv.shape[1]


iBW = []
jBW = []
totPix = 0
for i in range(height):
	for j in range(width):
		#print(gray[i,j])
		if bwImg[i,j]==255:
			totPix=totPix + 1
			iBW.append(i)
			jBW.append(j)
			#print("%d %d %d "% (totPix, iBW[totPix-1],jBW[totPix-1]))


print("Total white pixels: "+str(totPix))

print("Pixel for each seed: %2.3f "%(totPix/7.0) )

df = pd.DataFrame({'iBW':iBW, 'jBW':jBW})
#df.head()
print(df.ix[0:5,('iBW','jBW')])
print("done")

aSingleSeedArea = (800)*(scale/0.1)
#NoS = int(totPix/860)
#NoS = int(totPix/900)
NoS = int(totPix/aSingleSeedArea)

from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters=7)
kmeans = KMeans(n_clusters=NoS)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
centFloor = np.floor(centroids)

N = len(centroids)
c = [[0] * 2 for i in range(N)]

for i in range(N):
	for j in range(2):
		c[i][j] = int(centFloor[i,j])

print(centroids)
print(c)

#def getImage(i,j):


#for k in range(7):
for k in range(NoS):
	ic = c[k][0] 
	jc = c[k][1]

	for m in range(-2,2,1):
		for n in range(-2,2,1): 
			dst[ic+m,jc+n, 0]= 0
			dst[ic+m,jc+n, 1]= 0
			dst[ic+m,jc+n, 2]= 255

cv2. imshow("Dotted image",dst)



'''
BlockId = np.random.permutation(9)

kMat = [[0,0],[0,1],[0,2], [1,0],[1,1],[1,2], [2,0],[2,1],[2,2]]

idxI = [0,333,666]
idxJ = [0,333,666]


def addBrightness(img,i,j):
	delta = 50
	blue   =  img[i,j,0] + delta
	if (blue>255):
		blue = 255s
	green =  img[i,j,1] + delta
	if (green>255):
		green = 255
	red  =  img[i,j,2] + delta
	if (red>255):
		red = 255

	img[i,j,0] = blue
	img[i,j,1] = green
	img[i,j,2] = red

	return img
  



MaxBlock = 9
for K in range(0,MaxBlock):
	print(K)
	kI = kMat[BlockId[K]][0] #random.randrange(3) #2
	kJ = kMat[BlockId[K]][1] #random.randrange(3) #0
	for i in range(idxI[kI],(idxI[kI]+332)):
		for j in range(idxJ[kJ],(idxJ[kJ]+332)):
			addBrightness(img,i,j)

			
			#img[i,j,0] = img[i,j,0]  + 10 # 0  # Blue channel
			#img[i,j,1] = img[i,j,1]  + 10# 0  # Green channel
			#img[i,j,2] = img[i,j,2]  + 10# 0  # Green channel

dst = cv2.resize(img,None,fx=0.3,fy=0.3)
#dst2 = cv2.resize(imgF,None,fx=0.3,fy=0.3)


cv2.imshow('image',dst)

#cv2.imshow('imageF',dst2)

'''
cv2.waitKey(0)
