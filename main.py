from skimage.feature import hog
import cv2
import numpy as np
import os
import pickle as pk
from sklearn import cross_validation
from sklearn.svm import SVC


feature_vector = []
label = []

images = os.listdir("jaffe/")
for i in images:
	img = cv2.imread("jaffe/" + i, 0)
	fd, feature = hog(img, orientations=10, pixels_per_cell=(64, 64), cells_per_block=(1, 1), visualise=True)
	print i
	feature = feature.ravel()
	#print feature.shape
	#feature_vector.append(hog)
	feature_vector.append(feature)
	label.append(i[3:5])

feature_vector_array = np.asarray(feature_vector, dtype='uint8')
#print "as array successful"
#f = open('blah.pkl', 'wb')
#pk.dump(feature_vector, f)
#np.savetxt('out.csv', feature_vector_array, delimiter = ',')
np.savetxt('hog.csv', feature_vector_array, delimiter=',')
