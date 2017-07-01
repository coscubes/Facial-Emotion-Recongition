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
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib.pyplot as plt
from collections import Counter
import ldp
import feature

# variable for no preprocessed data
feature_normal = []

# variable for GABOR preprocessing
feature_gabor = []

# variable for HOG preprocessing 
feature_hog = []

# variable for LDP preprocessing
feature_ldp = []

# variable for LBP preprocessing 
feature_lbp = []

# variable for labels
labels = []

images = os.listdir("jaffefaces/") # list of images in the folder

for i in images:
	print i
	img = cv2.imread("jaffefaces/" + i, 0) # read the image as grayscale
	feature_normal.append(img.ravel()) # add normal raveled image to feature vector
	feature_gabor.append(feature.GABOR(img)) # add GABOR to image to feature vector
	feature_lbp.append(feature.LBPH(img)) # add LBP to image to feature vector
	feature_ldp.append(ldp.histogram_descriptor(ldp.ldp(img))) # add LDP to image to feature vector
	feature_hog.append(feature.HOG(img)) # add HOG of image to feature vector
	labels.append(i[3:5]) # list of labels

# Convert all the vectors into np.array
feature_normal = np.array(feature_normal)
feature_hog = np.array(feature_hog)
feature_ldp = np.array(feature_ldp)
feature_lbp = np.array(feature_lbp)
feature_gabor = np.array(feature_gabor)
labels = np.array(labels)

# classifiers for without any preprocessing
print "\nstart of No preprocessing results\n"
'''
print "Linear SVM"
clf = svm.SVC(kernel = 'linear', C=1)
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs = -1)
print scores
print "accuracy", scores.mean()

print "\nNo preprocessing + rbf SVM"
clf = svm.LinearSVC()
scores = cross_val_score(clf, feature_normal, labels, cv=10)
print scores
print "accuracy", scores.mean()

print "\nSGDClassifier loss = hinge"
clf = SGDClassifier(loss="hinge", penalty="l2")
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs=-1)
print scores
print "accuracy", scores.mean()

print "\nSGDClassifier SGDClassifier(loss=log)"
clf = SGDClassifier(loss="log")
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs=-1)
print scores
print "accuracy", scores.mean()

print "\nusing RBF sampler"
clf = SGDClassifier() 
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(feature_normal)
scores = cross_val_score(clf, X_features, labels, cv=10, n_jobs = -1)
print scores
print "accuracy", scores.mean()

print "\nKNN n = 1"
clf = KNeighborsClassifier(n_neighbors = 1)
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs=-1)
print scores
print "accuracy", scores.mean()
'''
print "\nKNN n = 2"
clf = KNeighborsClassifier(n_neighbors = 2)
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs=-1)
print scores
print "accuracy", scores.mean()

print "\nEnd of No preprocessing results\n"
