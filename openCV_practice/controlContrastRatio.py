import numpy as np
import cv2 as cv

src = cv.imread('sample.jpg', cv.IMREAD_GRAYSCALE)

if src is None:
    raise Exception("Image load failed")

def saturated(value):
    if value>255:
        value = 255
    elif value<0:
        value = 0
    return value

a = 2.0

mean = np.mean(src)
temp = src>mean

dst = np.empty(src.shape, dtype=src.dtype)
for i in range(len(temp)):
    for j in range(len(temp[0])):
        if temp[i][j]:
            dst[i,j] = saturated(src[i,j]*a)
        else:
            dst[i,j] = src[i,j]

cv.imwrite('contrast.jpg', dst)