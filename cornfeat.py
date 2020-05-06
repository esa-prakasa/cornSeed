import cv2
import os
import numpy as np

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




path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\"

fileList = []
with os.scandir(path) as entries:
    for entry in entries:
        fileList.append(entry.name)

for i in range (1,11,1):
	print(str(i)+"  "+fileList[i])


imgIdx = int(input("Give the image index!: "))
img = cv2.imread(path+fileList[imgIdx])

M = img.shape[0]
N = img.shape[1]
ratio = 0.5

img = cv2.resize(img,(int(ratio*M), int(ratio*N)) , interpolation = cv2.INTER_AREA)   
cv2.imshow("RED",img[:,:,2])
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
cv2.imshow("imgBW",imgBW)

#kernel = np.ones((5,5), np.uint8) 
#imgEro = cv2.erode(imgBW, kernel, iterations=1) 
#cv2.imshow("img Erroded",imgEro)

cv2.waitKey(0)
cv2.destroyAllWindows()

