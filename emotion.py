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
#from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from collections import Counter

c = Counter()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



images = os.listdir("jaffe/")

feature = []
label = []

for i in images:
	img = cv2.imread("jaffe/" + i, 0)
	#print i
	temp = img.ravel()
	feature.append(temp)
	label.append(i[3:5])

features = np.array(feature)
#print array.shape
labels = np.array(label)
print Counter(labels)
print np.unique(labels)
#print labels.shape
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state = 10, stratify = labels)
print Counter(y_train)
#print Counter(y_test)
print Counter(y_test)
#print Counter(X_test)

#print X_train.shape
#print y_train.shape
'''
print "Using Stochastic Gradient Descent "
clf_svm_lin = MLPClassifier(solver='lbfgs',alpha=1e-05,hidden_layer_sizes=(5,2),random_state=1)
clf_svm_lin.fit(X_train, y_train)
print "The score for Stochastic Gradient Descent ", clf_svm_lin.score(X_test, y_test)
scores = cross_val_score(clf_svm_lin, array, labels, cv=5)
print scores


print "Using RBF SVM"
clf_svm_rbf = svm.SVC(kernel='rbf', C=1)
clf_svm_rbf.fit(X_train, y_train)
print "The score for RBF SVM", clf_svm_rbf.score(X_test, y_test)
scores = cross_val_score(clf_svm_rbf, array, labels, cv=5)
print scores

print "Using Linear SVM"
clf_lin_svm = svm.SVC(kernel='linear')
scores = cross_val_score(clf_lin_svm, features, labels, cv=10)

print "Using KNN, n=5"
clf_lin_svm = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(clf_lin_svm, features, labels, cv=10)

print scores.mean(), scores.std() * 2
print scores

print "Using Logistic regression"
clf_LG = LogisticRegression()
scores = cross_val_score(clf_LG, features, labels, cv=10)
#scores = cross_val_score(clf_lin_svm, features, labels, cv=10)

print scores.mean(), scores.std() * 2
print scores

clf_LDA = LDA()
scores = cross_val_score(clf_LDA, features, labels, cv=10)

print scores.mean(), scores.std() * 2
print scores
'''
print "Using Linear SVM"
clf_lin_svm = LogisticRegression()
scores = cross_val_score(clf_lin_svm, features, labels, cv=10)
print scores
clf_lin_svm.fit(X_train, y_train)
y_pred = clf_lin_svm.predict(X_test)
class_names = list(set(y_test))
cnf_matrix = confusion_matrix(y_test, y_pred, labels = class_names)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()