import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread(r'C:\Users\Esa\Pictures\_DATASET\testObject\mug.jpg')



mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (0,0,img.shape[0],img.shape[1])
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

cv.imshow("Results", img)

cv.waitKey(0)
cv.destroyAllWindows()