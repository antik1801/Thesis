import sys
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pickle
import math
from sklearn import metrics
import seaborn as sns
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import style

data_set=pd.read_csv('dataset/data.csv')

data_set
# data_set.keys()
print(data_set.shape)


data_set.isnull().sum()


print(data_set.describe())


data_set.isnull().sum()


len(data_set)


with sns.axes_style('darkgrid'):
 sns.displot(data_set['label'], bins=5, color='green')
 plt.title("texts label");


 with sns.axes_style('darkgrid'):sns.displot(data_set['word'], bins=5, color='green')
 plt.title("words");


 X = data_set[['length','word']]
 y = data_set['label']


 X
 y


 from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


from sklearn import svm 
model_svm = svm.SVC()
model_svm.fit(X_train, y_train) 
y_prediction_svm = model_svm.predict(X_test) 

score_svm = metrics.accuracy_score(y_prediction_svm, y_test).round(4)
print("----------------------------------")
print('The accuracy of the SVM is: {}'.format(score_svm))
print("----------------------------------")

score = set()
score.add(('SVM', score_svm))

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=3) 
model_knn.fit(X_train, y_train) 
y_prediction_knn = model_knn.predict(X_test) 



score_knn = metrics.accuracy_score(y_prediction_knn, y_test).round(4)
print("----------------------------------")
print('The accuracy of the KNN is: {}'.format(score_knn))
print("----------------------------------")
score.add(('KNN', score_knn))



from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, y_train) 
y_prediction_nb = model_nb.predict(X_test) 

score_nb = metrics.accuracy_score(y_prediction_nb, y_test).round(4)
print("---------------------------------")
print('The accuracy of the NB is: {}'.format(score_nb))
print("---------------------------------")
score.add(('NB', score_nb))


from sklearn.linear_model import LogisticRegression 
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train) 
y_prediction_lr = model_lr.predict(X_test) 
score_lr = metrics.accuracy_score(y_prediction_lr, y_test).round(4)
print("---------------------------------")
print('The accuracy of the LR is: {}'.format(score_lr))
print("---------------------------------")
score.add(('LR', score_lr))