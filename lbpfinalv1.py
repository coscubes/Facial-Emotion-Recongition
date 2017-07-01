'''
AdaBoostClassifier
BaggingClassifier
BernoulliNB
CalibratedClassifierCV ----------------------------useless
DecisionTreeClassifier
ExtraTreesClassifier
GaussianNB
GaussianProcessClassifier ------------------------gives error
GradientBoostingClassifier
KNeighborsClassifier
LDA-----------------------------------------------nahi mila
LabelPropagation----------------------------------not working
LabelSpreading------------------------------------not working
LinearDiscriminantAnalysis
LinearSVC
LogisticRegression
LogisticRegressionCV
MLPClassifier
MultinomialNB
NearestCentroid
NuSVC
PassiveAggressiveClassifier
Perceptron
QDA
QuadraticDiscriminantAnalysis
RadiusNeighborsClassifier
RandomForestClassifier
RidgeClassifier
RidgeClassifierCV
SGDClassifier
SVC
'''

import cv2
import numpy as np
import os
import feature
import ldp

# Scikit-learn classifiers import
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import label_propagation
from sklearn.svm import LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

def find_label(temp):
	temp = temp[3:5]
	if temp == 'NE':
		return 1
	elif temp == 'HA':
		return 2
	elif temp == 'SA':
		return 3
	elif temp == 'SU':
		return 4
	elif temp == 'FE':
		return 5
	elif temp == 'DI':
		return 6
	elif temp == 'AN':
		return 7
	else:
		return 8

# Scikit-learn metrics import
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel as C

images = os.listdir("jaffefaces/")

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

images = os.listdir("ldp_of_jaffe/") # list of images in the folder

for i in images:
	img = cv2.imread("ldp_of_jaffe/" + i, 0)
	xarr=np.squeeze(np.array(img).astype(np.float32))
	m,v=cv2.PCACompute(xarr, np.mean(xarr, axis=0).reshape(1,-1))
	arr= np.array(v)
	flat_arr= arr.ravel()
	feature_normal.append(flat_arr)
	labels.append(find_label(i))

feature_normal = np.float32(feature_normal)
responses = np.float32(labels)

print "Classifiers applied for No preprocessing data\n"
print "\n Adaboost Classifier"
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
scores = cross_val_score(bdt, feature_normal, labels, cv=10, n_jobs = -1)
print scores
print "Accuracy", scores.mean()


print "\nBagging Classifier"
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, feature_normal, labels, cv=10, n_jobs = -1)
print scores
print "Accuracy", scores.mean()

print "\nBernoulliNB classifier"
nb = BernoulliNB()
scores = cross_val_score(nb, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nDecision Tree Classifier"
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nExtra trees classifier"
etsc = ExtraTreesClassifier()
scores = cross_val_score(etsc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nGaussian NB"
gnb = GaussianNB()
scores = cross_val_score(gnb, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

# Gaussian Process Classifier gives an unknown error
#print "\nGaussianProcessClassifier"
#kernel = C(0.1, (1e-5, np.inf)) * DotProduct(sigma_0=0.1) ** 2
#gp = GaussianProcessClassifier(kernel=kernel)
#scores = cross_val_score(gp, feature_normal, labels, cv=10, n_jobs = 4)
#print scores
#print "Accuracy", scores.mean()

print "\nGradientBoostingClassifier"
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
scores = cross_val_score(gbc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nKNeighborsClassifier n = 1"
knn1 = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn1, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nKNeighborsClassifier n = 2"
knn2 = KNeighborsClassifier(n_neighbors=2)
scores = cross_val_score(knn2, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nKNeighborsClassifier n = 5"
knn5 = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn5, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

# This hangs my computer for some reason
#print "\nLinearDiscriminantAnalysis n_components = 7"
#lda = LinearDiscriminantAnalysis(n_components=7)
#scores = cross_val_score(lda, feature_normal, labels, cv=10, n_jobs = 2)
#print scores
#print "Accuracy", scores.mean()

# LabelPropagation not working
#print "\nLabelPropagation"
#label_prop_model = LabelPropagation()
#scores = cross_val_score(label_prop_model, feature_normal, labels, cv=10, n_jobs = 4)
#print scores
#print "Accuracy", scores.mean()

# Label Spreading not working
#print "\nLabel Spreading"
#lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
#scores = cross_val_score(lp_model, feature_normal, labels, cv=10)
#print scores
#print "Accuracy", scores.mean()

print "\nLinearSVC"
lin_svc = LinearSVC()
scores = cross_val_score(lin_svc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing Logistic regression"
clf_LG = LogisticRegression()
scores = cross_val_score(clf_LG, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

# Logistic regression CV takes too much time to compile
#print "\nUsing Logistic regression CV"
#clf_LGCV = LogisticRegressionCV()
#scores = cross_val_score(clf_LGCV, feature_normal, labels, cv=10, n_jobs = 4)
#print scores
#print "Accuracy", scores.mean()

print "\nUsing MLPClassifier single hidden layer"
mlp = MLPClassifier(alpha = 1)
scores = cross_val_score(mlp, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing the perceptron"
per = Perceptron(fit_intercept=False, n_iter=10, shuffle=False)
scores = cross_val_score(per, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing MLPClassifier 3 hidden layer"
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
scores = cross_val_score(mlp, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing Passive aggressive Classifier"
pac = PassiveAggressiveClassifier()
scores = cross_val_score(pac, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing nearest centroid"
nc = NearestCentroid()
scores = cross_val_score(nc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nnusvc"
nusvc = NuSVC()
scores = cross_val_score(nusvc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

# This hangs my computer for some reason
#print "\n Using quadratic discriminant analysis"
#qda = QuadraticDiscriminantAnalysis(store_covariances=True)
#scores = cross_val_score(qda, feature_normal, labels, cv=10, n_jobs = 2)
#print scores
#print "Accuracy", scores.mean()

print "\nUsing Random Forest classifiers"
rfc = RandomForestClassifier(n_estimators=25)
scores = cross_val_score(rfc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

# multiproceassing error
#print "\nUsing Radius Neighbours classifier R = 100.0"
#rneigh = RadiusNeighborsClassifier(radius=10.0)
#scores = cross_val_score(rneigh, feature_normal, labels, cv=10, n_jobs = 4)
#print scores
#print "Accuracy", scores.mean()

print "\nUsing Ridge Classifier"
rgc = RidgeClassifier(tol=1e-2, solver="lsqr")
scores = cross_val_score(rgc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nUsing Stochastic gradient descent"
sgdc = SGDClassifier(loss="hinge", penalty="l2")
scores = cross_val_score(sgdc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nSupport vector Classifier kernel = rbf"
svcc = SVC(kernel='rbf', probability=True)
scores = cross_val_score(svcc, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nSupport vector classifier kernel = linear"
svcl = SVC(kernel='linear', C=1)
scores = cross_val_score(svcl, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()

print "\nSupport Vector classifier kernel = poly"
svcp = SVC(kernel='poly', gamma = 2)
scores = cross_val_score(svcp, feature_normal, labels, cv=10, n_jobs = 4)
print scores
print "Accuracy", scores.mean()


