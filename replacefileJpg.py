import numpy as np 
import json
import os
import cv2



path = r"C:\Users\Esa\Pictures\_DATASET\Augmented_2\Augmented_2\Plank30\img"

'''
y = os.listdir(path)
N = len(y)
for i in range(N):
	if y[i].endswith(".jp"):
		oldFile = os.path.join(path,y[i])
		newFile = oldFile+"g"
		print("\n")
		print(oldFile)
		print(newFile)
		os.rename(oldFile,newFile)
'''

y = os.listdir(path)
N = len(y)
for i in range(N):
	oldFile = os.path.join(path,y[i])
	newFile = oldFile+".jpg"
	print("\n")
	print(oldFile)
	print(newFile)
	os.rename(oldFile,newFile)
