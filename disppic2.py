#import numpy as np
#import pandas as pd
import cv2
#import random
#import math

 
img = cv2.imread('C:\\Users\\INKOM06\\Pictures\\jagung2020\\0304\\mod.bmp',1)
#cv2.imshow('Original image',img)


print("Done!")



ic = 200
jc = 180


def findHZLine(ic, jc, vcRg, hzRg, d):
	val = 1e3
	val2 = 1e3
	
	result = ic
	while ((val>0) | (val2>0)):
		val = int(0)
		val2 = int(0)
		for k in range(-vcRg,vcRg,1):
			val = val + img[result+k, jc, 0]
		for l in range(-hzRg,hzRg,1):
			val2 = val2 + img[result, jc+l,0]
		result = result + d
	return result


def findVCLine(ic, jc, vcRg, hzRg, d):
	val = 1e3
	val2 = 1e3
	
	result = jc
	while ((val>0) | (val2>0)):
		val = int(0)
		val2 = int(0)
		for k in range(-vcRg,vcRg,1):
			val = val + img[ic, result+k, 0]
		for l in range(-hzRg,hzRg,1):
			val2 = val2 + img[ic+l, result,0]
		result = result + d
	return result



vcRg =  2
hzRg = 11

val = 1e3
val2 = 1e3


iTop = findHZLine(ic, jc, vcRg, hzRg, -1)
iBot = findHZLine(ic, jc, vcRg, hzRg, 1)

jLft = findVCLine(ic, jc, vcRg, hzRg, -1) 
jRgt = findVCLine(ic, jc, vcRg, hzRg, +1) 

iTop = iTop - 10
iBot = iBot + 10

jLft = jLft - 10
jRgt = jRgt + 10


# Drawing coordinate (j,i) 
color = (0, 255, 0) 
thickness = 1

sp = []
sp.append([0, iTop])
sp.append([0, iBot])
sp.append([jLft,0])
sp.append([jRgt,0])

ep = []
ep.append([400, iTop])
ep.append([400, iBot])
ep.append([jLft, 400])
ep.append([jRgt, 400])



for i in range(0,2):
	start_point = tuple(sp[i]) 
	end_point = tuple(ep[i]) 
	img = cv2.line(img, start_point, end_point, color, thickness) 


for i in range(2,4):
	start_point = tuple(sp[i]) 
	end_point = tuple(ep[i]) 
	img = cv2.line(img, start_point, end_point, color, thickness) 



'''
start_point = (0, iTop) 
end_point = (400, iTop) 
img = cv2.line(img, start_point, end_point, color, thickness) 

start_point = (0, iBot) 
end_point = (400, iBot) 
img = cv2.line(img, start_point, end_point, color, thickness) 

start_point = (jLft, 0) 
end_point = (jLft, 400)
print(type(start_point)) 
img = cv2.line(img, start_point, end_point, color, thickness) 

start_point = (jRgt, 0) 
end_point = (jRgt, 400) 
img = cv2.line(img, start_point, end_point, color, thickness) 

'''
cv2.imshow("Centroid ", img)


cv2.waitKey()
cv2.destroyAllWindows()





