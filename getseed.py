import numpy as np
import pandas as pd
import cv2
import random
import math
import giveroi as gvroi
import os

os.system('cls')
 
# Load an color image in grayscale
fileNmSet = ['IMG_5002.JPG',   'IMG_5003.JPG',
'IMG_5004.JPG','IMG_5005.JPG','IMG_5006.JPG','IMG_5007.JPG','IMG_5008.JPG','IMG_5009.JPG','IMG_5010.JPG',
'IMG_5011.JPG','IMG_5013.JPG','IMG_5014.JPG','IMG_5015.JPG','IMG_5016.JPG','IMG_5017.JPG','IMG_5018.JPG',
'IMG_5019.JPG','IMG_5020.JPG','IMG_5021.JPG','IMG_5022.JPG','IMG_5023.JPG','IMG_5024.JPG','IMG_5025.JPG',
'IMG_5026.JPG','IMG_5027.JPG','IMG_5028.JPG','IMG_5029.JPG','IMG_5030.JPG','IMG_5031.JPG','IMG_5032.JPG',
'IMG_5033.JPG','IMG_5034.JPG','IMG_5035.JPG','IMG_5036.JPG','IMG_5037.JPG','IMG_5038.JPG','IMG_5039.JPG',
'IMG_5040.JPG','IMG_5041.JPG','IMG_5042.JPG','IMG_5043.JPG','IMG_5044.JPG','IMG_5045.JPG','IMG_5046.JPG',
'IMG_5047.JPG','IMG_5048.JPG']


maxNumOfFile = len(fileNmSet)

fileIdx = int(input("What is the image file index that needs to be extracted? (max: "+str(maxNumOfFile)+") "))

pathToRead = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\"+fileNmSet[fileIdx]
print(pathToRead)

Ns = len(pathToRead)
imgFile = pathToRead[(Ns-12):(Ns-4)] 


imgOri = cv2.imread(pathToRead,1)
img = imgOri.copy()

print('Original Dimensions : ',img.shape)
height = img.shape[0]
width = img.shape[1]
print('Height: '+str(height)+' Width: '+str(width))

scale = 0.1
dst = cv2.resize(img,None,fx=scale,fy=scale)
dst0 = dst.copy()

bwImg, hsv = gvroi.convertToGray(dst)
#hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
#print(hsv.shape)
#gray = hsv[:,:,2]
#ret, bwImg = cv2.threshold(gray,100,255,cv2. THRESH_BINARY)
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

#print(centroids)
#print(c)

#def getImage(i,j):

for k in range(NoS):
	ic = c[k][0] 
	jc = c[k][1]

	for m in range(-2,2,1):
		for n in range(-2,2,1): 
			dst[ic+m,jc+n, 0]= 0
			dst[ic+m,jc+n, 1]= 0
			dst[ic+m,jc+n, 2]= 255

#cv2.imshow("Dotted image",dst)
path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\centroid\\"
fullPath = path+imgFile+"_cent.jpg"
cv2.imwrite(fullPath, dst)



vcRg =  2
hzRg = 11

for k in range(NoS):
	ic = c[k][0]
	jc = c[k][1]

	iTop = gvroi.findHZLine(bwImg, ic, jc, vcRg, hzRg, -1)
	iBot = gvroi.findHZLine(bwImg, ic, jc, vcRg, hzRg, 1)

	jLft = gvroi.findVCLine(bwImg, ic, jc, vcRg, hzRg, -1) 
	jRgt = gvroi.findVCLine(bwImg, ic, jc, vcRg, hzRg, +1)

	print("Centroid "+str(k)+" "+str(iTop)+" "+str(iBot)+" "+str(jLft)+" "+str(jRgt))

	img = gvroi.drawLines(dst, iTop, iBot, jLft, jRgt)


	path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\small\\"
	if (k<10):
		nm = "0"+str(k)
	else:
		nm = str(k)
	nm = imgFile+"_"+nm
	fullPath = path+nm+".jpg"

	img2 = dst0[iTop:iBot, jLft:jRgt, :]
	cv2.imwrite(fullPath, img2)



	iTOP = int(np.round(iTop/scale))
	iBOT = int(np.round(iBot/scale))
	jLFT = int(np.round(jLft/scale))
	jRGT = int(np.round(jRgt/scale))

	path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\large\\"
	fullPath = path+nm+".jpg"
	
	img3 = imgOri[iTOP:iBOT, jLFT:jRGT, :] 
	cv2.imwrite(fullPath, img3)




#cv2.imshow("Box", dst)
path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\box\\"
fullPath = path+imgFile+"_box.jpg"
cv2.imwrite(fullPath, dst)



cv2.waitKey(0)
cv2.destroyAllWindows()

