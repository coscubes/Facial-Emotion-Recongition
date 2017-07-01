import cv2
import numpy as np
from skimage.feature import hog
import os
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.lda import LDA
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib.pyplot as plt
from collections import Counter
import ldp
import feature
from skimage.filters import gabor
from math import pi
from math import sqrt
import os

frequency = [1.5707, 1.117, 0.7853, 0.5553, 0.3926]
angles = [0, pi / 8, pi / 4, pi * 3 / 8, pi / 2, pi * 5 / 8, pi * 0.75, pi * 7 / 8]

feature_normal = []
labels = []
images = os.listdir("jaffefaces/")

for freq in frequency:
	for theta in angles:
		for i in images:
			img = cv2.imread("jaffefaces/" + i, 0)
			temp = gabor(img, freq, theta)
			temp = np.array(temp).ravel()
			feature_normal.append(temp)
			labels.append(i[3:5])

		clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
		scores = cross_val_score(clf1, feature_normal, labels, cv=10, n_jobs = -1)
		print scores
		print "angle, frequency, accuracy", theta, freq, scores.mean()

		feature_normal = []
		labels = []


