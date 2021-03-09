Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@hesamjalalian 
Learn Git and GitHub without any code!
Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.


hesamjalalian
/
Hesam
Private
2
00
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Settings
Hesam/preprocessing.py /
@hesamjalalian
hesamjalalian Create preprocessing.py
Latest commit e1e22f4 4 days ago
 History
 1 contributor
48 lines (40 sloc)  1.4 KB
  
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import svm
from joblib import dump, load

dataset = loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Wine.mat")
data=dataset['dataset']
#print(dataset)

X = data[:, 0:-1]
y = data[:, -1]

# I tried here to split once and I am going to update it with 20 repetition
train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print(X_train)
#print(X_train)

# normalization
ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.transform(X_test)

#print(X_test_ms)

# handling the missing data and replace missing values with mean of all the other values
# I do not apply this function for test_set since it should be unseen data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train_ms[:, 1:])
X_train_ms[:, 1:] = imputer.transform(X_train_ms[:, 1:])
print(X_train_ms)

clf = svm.SVC()
dump(clf, 'X_train_ms.joblib')
clf = load('X_train_ms.joblib')
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
