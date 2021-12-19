import os
import cv2
import numpy as np
import math

path = r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\11.09 -- Count CORN\imageDataset\d8b0418d-6cef-47e0-884d-896177587683.jpg"
path = r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\11.09 -- Count CORN\imageDataset\b9ee7d85-fafd-4a37-8d0a-eb43bd19cc3c.jpg"
# path = r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\11.09 -- Count CORN\imageDataset\b9ee7d85-fafd-4a37-8d0a-eb43bd19cc3cRot.jpg"






I = cv2.imread(path)
M = I.shape[0]
N = I.shape[1]
r = 0.4
I = cv2.resize(I,[int(r*N), int(r*M)])

cv2.imshow("I",I)

LAB = cv2.cvtColor(I, cv2.COLOR_BGR2LAB)
HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# LABDisp = np.hstack(LAB[:,:,0],LAB[:,:,1])
# LABDisp = np.hstack(LABDisp,LAB[:,:,2])

print(LAB.shape)

# cv2.imshow("L",LAB[:,:,0])
# cv2.imshow("A",LAB[:,:,1])
# cv2.imshow("B",LAB[:,:,2])

# cv2.imshow("H",HSV[:,:,0])
# cv2.imshow("S",HSV[:,:,1])
# cv2.imshow("V",HSV[:,:,2])
V = HSV[:,:,2]
ksize = (7, 7)
V = cv2.GaussianBlur(V, ksize, 0)
V = cv2.GaussianBlur(V, ksize, 0)

ret, th1 = cv2.threshold(V, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("th1",th1)

[M2, N2] = th1.shape
iAcc = 0
jAcc = 0
nPix = 0
for i in range(M2):
    for j in range(N2):
        if th1[i,j] == 255:
            iAcc = iAcc + i
            jAcc = jAcc + j
            nPix = nPix + 1

iAvg = int(iAcc/nPix)
jAvg = int(jAcc/nPix)

rgb = np.zeros((int(r*M),int(r*N),3), dtype=np.uint8)
rgb[:,:,0] = th1
rgb[:,:,1] = th1
rgb[:,:,2] = th1

delta = 3
for i in range(-delta,delta,1):
    for j in range(-delta,delta,1):
        rgb[iAvg+i, jAvg+j, 0] = 0
        rgb[iAvg+i, jAvg+j, 1] = 0
        rgb[iAvg+i, jAvg+j, 2] = 255



edg = cv2.Canny(th1,0.1,0.2)

cv2.imshow("Edge", edg)

edgArr=[]
Rmax = 1e-5
Rmin = 1e+5
for i in range(10,M2,1):
    for j in range(N2):
        if edg[i,j] == 255:
            di = (i-iAvg)**2
            dj = (j-jAvg)**2
            R = math.sqrt(di + dj)
            if R>Rmax:
                iMax = i
                jMax = j
                Rmax = R
            if R<Rmin:
                iMin = i
                jMin = j
                Rmin = R

Rmax2 = 1e-5
Rmin2 = 1e+5
for i in range(10,M2,1):
    for j in range(N2):
        if edg[i,j] == 255:
            di = (i-iAvg)**2
            dj = (j-jAvg)**2
            R = math.sqrt(di + dj)
            if ((R>Rmax2)and(R<Rmax)and(i>iAvg)):
                iMax2 = i
                jMax2 = j
                Rmax2 = R

            # if ((R<Rmin2)and(R>Rmin)and(j>jAvg)):
            if ((R<Rmin2)and(j>jAvg)):
                iMin2 = i
                jMin2 = j
                Rmin2 = R


x1 = jAvg
y1 = iAvg
x2 = jMax
y2 = iMax
cv2.line(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.line(rgb, (x1, y1), (jMin, iMin), (0, 0, 255), 2)

cv2.line(rgb, (x1, y1), (jMax2, iMax2), (255, 0, 0), 2)
cv2.line(rgb, (x1, y1), (jMin2, iMin2), (255, 0, 255), 2)

cv2.imshow("rgb",rgb)


cv2.waitKey(0)
cv2.destroyAllWindows()