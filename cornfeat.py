import cv2
import os
import numpy as np
import math

os.system("cls")

def getAvgIntensity(img, i0, j0, fSz):
	avgVal = 0
	N = 0
	for i in range(fSz):
		for j in range(fSz):
			N = N + 1
			avgVal = avgVal + img[(i0+i),(j0+j)]
	avgVal = avgVal/N
	return avgVal


def getCentroidOfMass(img):
	M = img.shape[0]
	N = img.shape[1]
	iC = 0
	jC = 0
	Nimg = 0
	for i in range(M):
		for j in range(N):
			if (img[i,j] == 255):
				Nimg = Nimg + 1
				iC = iC + i
				jC = jC + j
	iC = int(iC/Nimg)
	jC = int(jC/Nimg)
	return iC,jC


def getRadiusAndAngle(i,j,iC,jC):

	dI = i - iC
	dJ = j - jC



	if ((dI==0)&(dJ>0)):  #Position at 0 deg
		R = dJ
		deg = 0

	if ((dI<0)&(dJ==0)):  #Position at 90 deg
		R = abs(dI)
		deg = 90

	if ((dI==0)&(dJ<0)):  #Position at 180 deg
		R = abs(dJ)
		deg = 180

	if ((dI>0)&(dJ==0)):  #Position at 270 deg
		R = abs(dI)
		deg = 270

	if ((dI!=0)&(dJ!=0)):

		R = math.sqrt((dI**2)+(dJ**2))
		deg = abs(math.atan(dI/dJ)*180/3.14)

		if ((dI<0)&(dJ>0)):
			deg = deg

		if ((dI<0)&(dJ<0)):
			deg = 180 - deg

		if ((dI>0)&(dJ<0)):
			deg = 180 + deg

		if ((dI>0)&(dJ>0)):
			deg = 360 - deg

	return R, deg 







path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\truet\\"
path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\model\\"
path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\false\\"

subFoldPath = path[-6:-1]

#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\"

fileList = []
with os.scandir(path) as entries:
    for entry in entries:
        fileList.append(entry.name)

for i in range (1,5,1):
	print(str(i)+"  "+fileList[i])


imgIdx = int(input("Give the image index!: "))
img = cv2.imread(path+fileList[imgIdx])

M = img.shape[0]
N = img.shape[1]
ratio = 0.5

img = cv2.resize(img,(int(ratio*M), int(ratio*N)) , interpolation = cv2.INTER_AREA)   
#cv2.imshow("RED",img[:,:,2])
#cv2.imshow("GREEN",img[:,:,1])
#cv2.imshow("BLUE",img[:,:,0])


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.imshow("HUE",hsv[:,:,0])
#cv2.imshow("SAT",hsv[:,:,1])
#cv2.imshow("VAL",hsv[:,:,2])

avg = []
M = img.shape[0]
N = img.shape[1]

fSz = 7
avg.append(getAvgIntensity(img[:,:,2], 0, 0, fSz))
avg.append(getAvgIntensity(img[:,:,2], (M-fSz), 0, fSz))
avg.append(getAvgIntensity(img[:,:,2], 0, (N-fSz), fSz))
avg.append(getAvgIntensity(img[:,:,2], (M-fSz), (N-fSz), fSz))

finalAvg = 0
for i in range(4):
	finalAvg = finalAvg + avg[i]
	print(avg[i])

finalAvg = int(finalAvg/4)
print(finalAvg)



thr,imgBW = cv2.threshold(img[:,:,2],int(finalAvg*5),255,cv2.THRESH_BINARY)
#cv2.imshow("imgBW",imgBW)


centPt = [0, 0]
centPt = getCentroidOfMass(imgBW)


imgBWrgb = np.zeros((M,N,3), dtype=np.uint8)
for i in range(3):
	imgBWrgb[:,:,i] = imgBW

print(centPt[0])
print(centPt[1])


for k in range(-3,4,1):
	for l in range(-3,4,1):
		imgBWrgb[ (centPt[0]+k), (centPt[1]+l), 0] = 0
		imgBWrgb[ (centPt[0]+k), (centPt[1]+l), 1] = 0
		imgBWrgb[ (centPt[0]+k), (centPt[1]+l), 2] = 255


# Canny Edge Detection
edges = cv2.Canny(imgBW,100,200)
#cv2.imshow("Edges image", edges)


#Get dataPoint R(delta)
iC = centPt[0]
jC = centPt[1]
phi = 3.14
NPt = 0 

radArr = []
degArr  = []
#R = 0
#deg = 0

for i in range(M):
	for j in range(N):
		if edges[i,j] == 255:
			NPt = NPt + 1
			#print(NPt)
			[R, deg] = getRadiusAndAngle(i,j,iC,jC)

			radArr.append(R)			
			degArr.append(deg)








#cv2.imshow("imgBW RGB",imgBWrgb)


#kernel = np.ones((5,5), np.uint8) 
#imgEro = cv2.erode(imgBW, kernel, iterations=1) 
#cv2.imshow("img Erroded",imgEro)


import matplotlib.pyplot as plt


fig, axs = plt.subplots(1,2,figsize=(10,4))
#fig= plt.figure(figsize=(600,300))

fig.suptitle("Profile extraction of image "+str(imgIdx)+": "+fileList[imgIdx])
axs[0].set_title('Segmented image')
axs[0].imshow(imgBW, cmap='gray', vmin=0, vmax=255)

yMax = max(M,N)

axs[1].set_title('Boundary profile: Radius vs. Degree')
axs[1].axis([0, yMax, 0, 300])
axs[1].plot(degArr,radArr, 'r.')

plt.show()


pathToSave = "C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.02-011-IDENTIFIKASI Jagung 2020\\Eksperimen Feature Extraction\\figures\\"

figPath = pathToSave+subFoldPath+"\\fig_"+fileList[imgIdx]
print(figPath)
#fig.savefig(figPath, dpi=fig.dpi)






cv2.waitKey(0)
cv2.destroyAllWindows()

