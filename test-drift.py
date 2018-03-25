'''
run with 
python test-de-grupos.py <number of group> <classifier>

1:'KNN'
2:'MLP'
3:'RF'
4:'SVM-RBF'
5:'SVM-RBF'
6:'GNB'
7:'SDG'
8:'Tree'

'''

'''
ploting
'''
#import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

'''

Machine Learning Models
'''
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing


##System
import csv
import numpy as np
import sys, time
import pandas as pd

#concept drift

from classifier.detector_classifier import DetectorClassifier
from concept_drift.adwin import Adwin
from concept_drift.page_hinkley import PageHinkley
from evaluation.prequential import prequential

#Normalizers
# from newmain import NewMain 
# from maxMin_Normalizer import maxMin_Normalizer
# from NewmaxMin import *


'''
incremental 
'''
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

'''
normalizer do sklearner
'''
from sklearn.preprocessing import Normalizer


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

#to balance training dataset
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN 



def read_data(file):
	df = pd.read_csv(file)
	data = df.values
	X=data[:,5:-2]
	y=data[:, -1]
	for i in range(len(y)):
		if y[i]!=0:
			y[i]=1
	label=np.unique(y).tolist()
	le = preprocessing.LabelEncoder()
	le.fit(label)
	y = le.transform(y)
	return X.astype(float),y



n_train = 2000
X, y = read_data(sys.argv[1])
window = [100,200,500,1000,2000]

output=open('test-drift.txt','w')

clfs = [
		MultinomialNB(),
		#DetectorClassifier(Adwin(), PageHinkley(), np.unique(y)),
		#GaussianNB(),
		DetectorClassifier(MultinomialNB(), PageHinkley(), np.unique(y)),
		DetectorClassifier(MultinomialNB(), Adwin(), np.unique(y))
	]
clfs_label = ["MultinomialNB",  "Page-Hinkley", "ADWIN"]

#plt.title("Accuracy (exact match)")
plt.xlabel("Instances")
plt.ylabel("Accuracy")

for w in window:

	for i in range(len(clfs)):
		print("\n{} :".format(clfs_label[i]))
		with np.errstate(divide='ignore', invalid='ignore'):
			y_pre, time = prequential(X, y, clfs[i], n_train)
		if clfs[i].__class__.__name__ == "DetectorClassifier":
			print("Drift detection: {}".format(clfs[i].change_detected))
			output.write(str(clfs[i].change_detected)+'\n')
		estimator = (y[n_train:] == y_pre) * 1

		acc_run = np.convolve(estimator, np.ones((w,)) / w, 'same')
		print("Mean acc within the window {}: {}".format(w, np.mean(acc_run)))
		output.write(str(np.mean(acc_run))+'\n')
		plt.plot(acc_run, "-", label=clfs_label[i])


plt.legend(loc='lower right')
plt.ylim([0, 1])
plt.show()
output.close()