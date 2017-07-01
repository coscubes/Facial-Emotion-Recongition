import cv2
import numpy as np
import os
from ldp_new import ldp_new
from histimg import histimg

from sklearn.svm import LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

images = os.listdir("ldp_imgs/")

def get_labels(value):
	'''
	AF = afraid
	AN = angry
	DI = disgusted
	HA = happy
	NE = neutral
	SA = sad
	SU = surprised
	'''
	if value[4:6] == 'AF':
		return 1
	elif value[4:6] == 'AN':
		return 2
	elif value[4:6] == 'DI':
		return 3
	elif value[4:6] == 'HA':
		return 4
	elif value[4:6] == 'NE':
		return 5
	elif value[4:6] == 'SA':
		return 6
	else:
		return 7

feature_normal = []
labels = []

for i in images:
	img = cv2.imread("ldp_imgs/" + i, 0)
	hist = histimg(img)
	print hist.shape
	feature_normal.append(hist)
	labels.append(get_labels(i))


X_train, X_test, y_train, y_test = train_test_split(feature_normal, labels, test_size=0.2, random_state=0, stratify=labels)
'''
print "starting PCA"
pca = PCA(n_components=12000)
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)
'''
print "Starting SVM"
clf = SVC(kernel='linear', C=1).fit(X_t_train, y_train)
print clf.score(X_t_test, y_test)

	
