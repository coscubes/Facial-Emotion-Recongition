# calculate histogram of image

import numpy as np

def histimg(image, windowsize_r = 5, windowsize_c = 5, bins = 16):
	# Crop out the window and calculate the histogram
	ar = []
	for r in range(0, image.shape[0] - windowsize_r, windowsize_r):
	    for c in range(0, image.shape[1] - windowsize_c, windowsize_c):
	        window = image[r:r+windowsize_r,c:c+windowsize_c]
	        hist, bins = np.histogram(window.ravel(),bins=bins, [0,256])
	        ar.append(hist)

	return np.array(ar)
