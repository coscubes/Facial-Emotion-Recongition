import cv2
import numpy as np
from skimage.feature import hog
import os
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import SGDClassifier
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

images = os.listdir("jaffefaces/")

feature_ldp = []
feature_gabor = []
feature_lbp = []
label = []

for i in images:
	img = cv2.imread("jaffefaces/" + i, 0)
	print i
	hist = ldp.histogram_descriptor(ldp.ldp(img))
	feature_ldp.append(hist)
	hist = feature.LBPH(img)
	feature_lbp.append(hist)
	gabor = feature.GABOR(img)
	feature_gabor.append(gabor)
	label.append(i[3:5])

feature_ldp = np.array(feature_ldp)
feature_lbp = np.array(feature_lbp)
feature_gabor = np.array(feature_gabor)
labels = np.array(label)

X_train1, X_test1, y_train1, y_test1 = train_test_split(feature_ldp, labels, test_size=0.3, random_state = 10, stratify = labels)
X_train2, X_test2, y_train2, y_test2 = train_test_split(feature_lbp, labels, test_size=0.3, random_state = 10, stratify = labels)

'''
print "Using Linear SVM LBP"
clf_lin_svm1 = MLPClassifier(solver='lbfgs',alpha=1e-05,hidden_layer_sizes=(5,2),random_state=1)
scores = cross_val_score(clf_lin_svm1, feature_lbp, labels, cv=10)
print scores

print "Using Linear SVM LDP"
clf_lin_svm2 = MLPClassifier(solver='lbfgs',alpha=1e-05,hidden_layer_sizes=(5,2),random_state=1)
scores = cross_val_score(clf_lin_svm2, feature_ldp, labels, cv=10)
print scores
'''
print "Using Linear SVM RBF"
clf_lin_svm3 = KNeighborsClassifier(n_neighbors=2)
scores = cross_val_score(clf_lin_svm3, feature_gabor, labels, cv=10)
print scores
