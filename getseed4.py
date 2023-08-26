import numpy as np
import pandas as pd
import cv2
import random
import math
import giveroi as gvroi
import os

os.system('cls')
 


# pathToRead = "C:\\Users\\INKOM06\\Pictures\\_DATASET\\BIMAseed\\_bima20bad\\"
pathToRead = "C:\\Users\\IPT\\Pictures\\corn\\"
pathToRead_ori = pathToRead
fileNmSet = os.listdir(pathToRead+"\\data\\")
print(fileNmSet)

maxNumOfFile = len(fileNmSet)
print("maxNumOfFile %d"%(maxNumOfFile))

#for fileIdx in range(3):
for fileIdx in range(maxNumOfFile):

##pathToRead = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\"+fileNmSet[fileIdx]
	# pathToRead = "C:\\Users\\INKOM06\\Pictures\\_DATASET\\BIMAseed\\_bima20bad\\"+fileNmSet[fileIdx]
	pathToRead = pathToRead+"\\data\\"+fileNmSet[fileIdx]

	print(pathToRead)

	Ns = len(pathToRead)
	imgFile = pathToRead[(Ns-12):(Ns-4)] 

	imgOri = cv2.imread(pathToRead)


	# cv2.imshow("imgOri",imgOri[0:200, 0:200,:])


	img = imgOri.copy()
	###-----#print('Original Dimensions : ',img.shape)
	height = img.shape[0]
	width = img.shape[1]
	###-----#print('Height: '+str(height)+' Width: '+str(width))

	scale = 0.1
	dst = cv2.resize(img,None,fx=scale,fy=scale)
	dst0 = dst.copy()

	bwImg, hsv = gvroi.convertToGray(dst)
	#hsv = cv2.cvtColor(dst,cv2.COLOR_BGR2HSV)
	#print(hsv.shape)
	#gray = hsv[:,:,2]
	#ret, bwImg = cv2.threshold(gray,100,255,cv2. THRESH_BINARY)
	#cv2.imshow("Binary Image",bwImg)

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


	###-----#print("Total white pixels: "+str(totPix))

	#print("Pixel for each seed: %2.3f "%(totPix/7.0) )
	###-----#print("Pixel for each seed: %2.3f "%(totPix/5.0) )

	df = pd.DataFrame({'iBW':iBW, 'jBW':jBW})
	###-----#print(df.head())
	#print(df.ix[0:5,('iBW','jBW')])
	###-----#print("done")

	#aSingleSeedArea = (800)*(scale/0.1)
	aSingleSeedArea = (1500)*(scale/0.1)
	#NoS = int(totPix/860)
	#NoS = int(totPix/900)
	NoS = 30 #int(totPix/aSingleSeedArea)

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
	#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\centroid\\"
	path = pathToRead_ori+"centroid\\"



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

		###-----#print("Centroid "+str(k)+" "+str(iTop)+" "+str(iBot)+" "+str(jLft)+" "+str(jRgt))

		img = gvroi.drawLines(dst, iTop, iBot, jLft, jRgt)


		#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\small\\"
		path = pathToRead_ori+"small\\"




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

		#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\large\\"
		path = pathToRead_ori+"large\\"

		fullPath = path+nm+".jpg"
		
		img3 = imgOri[iTOP:iBOT, jLFT:jRGT, :] 
		cv2.imwrite(fullPath, img3)




	#cv2.imshow("Box", dst)
	#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\box\\"
	path = pathToRead_ori+"box\\"

	fullPath = path+imgFile+"_box.jpg"
	cv2.imwrite(fullPath, dst)

	print(str(fileIdx)+" "+fullPath)


print("Everythings are done!!")

cv2.waitKey(0)
cv2.destroyAllWindows()

