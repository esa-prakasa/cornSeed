import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import getdeg 



path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"

idx = int(input("Image index that needs to be analysed? "))


files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
	for file in f:
		files.append(os.path.join(r, file))


img = cv2.imread(files[idx])
print("File name is "+files[idx])
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray = hsv[:,:,2]

ret,img2 = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

kernel = np.ones((5,5),np.uint8)
img3 = cv2.dilate(img2,kernel,iterations = 3)

img4 = cv2.Canny(img3,100,200)


height = img4.shape[0]
width = img4.shape[1]

iArr = []
jArr = []

for i in range(height):
	for j in range(width):
		if img4[i,j] > 0:
			iArr.append(i)
			jArr.append(j)

acI = 0
acJ = 0
N = len(iArr)
for i in range(N):
	print("i: "+str(iArr[i])+"  j: "+str(jArr[i]))
	acI += iArr[i]
	acJ += jArr[i]

iMean = int(np.floor(acI/N))
jMean = int(np.floor(acJ/N))

print(iMean)
print(jMean)

for i in range(-5,5):
	for j in range(-5,5):
		img[int(iMean+i), int(jMean+j),0] = 0
		img[int(iMean+i), int(jMean+j),1] = 0
		img[int(iMean+i), int(jMean+j),2] = 255


angleArr = []
radiusArr = []

for i in range(N):
	#print("i: "+str(iArr[i])+"  j: "+str(jArr[i]))
	dIVal = -(iArr[i] - iMean)
	dJVal = jArr[i] - jMean

	#print(" %f  %f "%(dIVal, dJVal))


	angleVal  = getdeg.getTanAngle(dIVal, dJVal)
	radiusVal = getdeg.getRadius(dIVal, dJVal)
	print(" i: %d angle: %f  radius: %f"%(i, angleVal, radiusVal))

	if (angleVal<=360):
		angleArr.append(angleVal)
		radiusArr.append(radiusVal)




# time= np.arange(0, 10, 0.1);
# amplitude = np.sin(time)
#plt.scatter(x, y)

'''
plt.subplot(121)
plt.scatter(angleArr, radiusArr)
plt.title('radius vs angle')

plt.ylim(top=300)  # adjust the top leaving bottom unchanged
plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged
plt.show()

plt.subplot(122)

plt.axis("off")
#plt.title('Original image')
plt.imshow(img)
plt.show()
'''

#fig, (ax1, ax2) = plt.subplots(1, 2)
#fig.suptitle('Radius vs angle')

#ax1.scatter(angleArr, radiusArr)
#ax1.plt.show()
#ax2.scatter(angleArr, radiusArr)
#ax2.plt.show()


b,g,r = cv2.split(img)       # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb


fig, axs = plt.subplots(1,2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(rgb_img)

axs[1].scatter(angleArr, radiusArr)
axs[1].grid(True)
plt.ylim(50, 300)
plt.xlim(0, 360)
#axs[1].ylim(top=300)  # adjust the 0 leaving 360 unchanged
#axs[1].ylim(bottom=0)  # adjust the bottom leaving top unchanged



plt.show()





#cv2.imshow("Hello",img)

#print(len(files)-pyplot.1)

cv2.waitKey(0)
cv2.destroyAllWindows()
