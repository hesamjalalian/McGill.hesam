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

#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Adult.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Banana.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Blood.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Breast.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\CTG.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Ecoli.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Faults.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\German.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\GLASS.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Haberman.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Heart.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\ILPD.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Ionosphere.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Laryngeal1.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Laryngeal3.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Lithuanian.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Liver.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Magic.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Mammographic.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Monk2.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Phoneme.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Pima.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Segmentation.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Sonar.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Thyroid.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Vehicle.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Vertebral.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\WDVG1.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Weaning.mat")
#dataset=loadmat(r"C:\Users\Hesam\27.2.2021\datasets\Wine.mat")
#data=dataset['dataset']
#print(dataset)

dataDir = "C:/Users/Hesam/27.2.2021/datasets/"
mats = []
for file in os.listdir(dataDir):
    mats.append( sio.loadmat( dataDir+file ) )

print(mats)
#def load_dataset():
#    for i in range(1, 30):
#        array=["Adult", "Banana", "Blood", "Breast", "CTG", "Ecoli", "Faults", "German", "Glass", "Haberman", "Heart", "ILPD", "Ionosphere", "Laryngeal1", "Laryngeal3", "Lithuanian", "Liver", "Magic", "Mammographic", "Monk2", "Phoneme", "Pima", "Segmentation", "Sonar", "Thyroid", "Vehicle", "Vertebral", "WDVG1", "Weaning", "Wine"]
#        data = sio.loadmat(array[i])
#    return data
#data=dataset['dataset']

#print(load_dataset())

#def load_dataset(dataset):
#    mat = sio.loadmat(dataset)
#    return mat

#dataset=load_dataset("Adult")
#dataset=load_dataset("Breast")
#dataset=load_dataset("CTG")
#dataset=load_dataset("Ecoli")
#dataset=load_dataset("Faults")
#dataset=load_dataset('German')
#dataset=load_dataset("Glass")
#dataset=load_dataset("Haberman")
#dataset=load_dataset("Heart")
#dataset=load_dataset("ILPD")
#dataset=load_dataset("Ionosphere")
#dataset=load_dataset("Laryngeal1")
#dataset=load_dataset("Laryngeal3")
#dataset=load_dataset("Lithuanian")
#dataset=load_dataset("Liver")
#dataset=load_dataset("Magic")
#dataset=load_dataset("Mammographic")
#dataset=load_dataset("Monk2")
#dataset=load_dataset("Phoneme")
#dataset=load_dataset("Pima")
#dataset=load_dataset("Segmentation")
#dataset=load_dataset("Sonar")
#dataset=load_dataset("Thyroid")
#dataset=load_dataset("Vehicle")
#dataset=load_dataset("Vertebral")
#dataset=load_dataset("WDVG1")
#dataset=load_dataset("Weaning")
#dataset=load_dataset("Wine")
#data=dataset['dataset']
#print(dataset)
#print(load_dataset("Wine"))

X = mats[:, 0:-1]
Y = mats[:, -1]

# splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25,random_state=1)

# There are 4 main important steps for the preprocessing of data: 1.Splitting of the data set in Training and Validation sets/2. Missing values/ 3.Normalization of data set
# min_max = MinMaxScaler()

# Standardising data with StandardScaler
#ss = StandardScaler()
#X_train_ss = ss.fit_transform(X_train)
#X_test_ss = ss.transform(X_test)
#print(X_train_ss)

# normalized_X = preprocessing.normalize(X)
#X_normalized = preprocessing.normalize(X, norm='l2')
#print(X_normalized)

# normalization
ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.transform(X_test)

print(X_test_ms)
