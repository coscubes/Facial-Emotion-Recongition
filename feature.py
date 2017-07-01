import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def LBPH(image, numPoints = 24, radius = 8, eps = 1e-7):
	if len(image.shape) > 2:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	lbp = local_binary_pattern(image, numPoints, radius, method="uniform")
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
	
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	
	return hist

def GABOR(image, ksize = 31, theta = 0):
	if len(image.shape) > 2:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	kern = kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
	fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
	return fimg.ravel()

def HOG(image):
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
	return fd


