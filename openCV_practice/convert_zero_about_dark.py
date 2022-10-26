import numpy as np
import cv2 as cv

img = cv.imread('sample.jpg', cv.IMREAD_GRAYSCALE)

if img is None:
    raise Exception("Image load failed")

mean = np.mean(img)
temp = img<mean
result = img[:]

for i in range(len(temp)):
    for j in range(len(temp[0])):
        if temp[i][j]:
            result[i][j] = 0

cv.imwrite('output.jpg', result)