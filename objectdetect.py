import cv2
import os

path = r"C:\Users\Esa\Pictures\_car\car.png"

img = cv2.imread(path)
cv2.imshow("car", img)

cv2.waitKey(0)
cv2.destroyAllWindows()