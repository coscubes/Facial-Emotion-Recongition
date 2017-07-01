import numpy as np
import cv2
import os

def drldp(img):
	h, w = img.shape
	
	if (h % 3) != 0:
		if (h % 3) == 1:
			b = np.zeros(w, np.uint8)
			img = np.vstack((b, img, b))
		else:
			b = np.zeros(h, np.uint8)
			img = np.vstack((img, b))
	h, w = img.shape

	if (w % 3 ) != 0:
		if (w % 3) == 1:
			b = np.zeros((h, 1), np.uint8)
			print b.shape
			img = np.hstack((b, img, b))
		else:
			b = np.zeros((h, 1), np.uint8)
			b = b. transpose()
			img = np.hstack((img, b))

	newimg = np.zeros((img.shape[0] / 3, img.shape[1] / 3), np.uint8)
	
	windowsize_r = 3
	windowsize_c = 3

	for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
	    for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
	        window = img[r:r+windowsize_r,c:c+windowsize_c]
	        window = window.ravel()
	        a = 0
	        for i in window:
	        	a = a ^ i

	        newimg[r / 3, c / 3] = a

	return newimg

images = os.listdir("ldp_of_jaffefaces/")

for i in images:
	print i
	img = cv2.imread("ldp_of_jaffefaces/" + i, 0)
	temp = drldp(img)
	cv2.imwrite("drldp_jaffefaces/" + i, temp)