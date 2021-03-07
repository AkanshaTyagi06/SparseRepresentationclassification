#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:08:29 2021

@author: akansha
"""
import os
import librosa
import numpy as np
import glob
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit

dataset_path="/home/akansha/Desktop/others/SRC/Environmental-Sound-Classification-master/dataset/"


''' returns the mfcc '''

def get_features(file_name):

    if file_name: 
        X, sample_rate = sf.read(file_name, dtype='float32')

    # mfcc (mel-frequency cepstrum)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    #print("shape of mfcc",mfccs.shape)
    #print("sample rate",sample_rate)
    mfccs_scaled = np.mean(mfccs.T,axis=0)
    return mfccs_scaled


sub_dirs = os.listdir(dataset_path)
#print("directories length",len(sub_dirs))
#print("directories names \n",sub_dirs)
sub_dirs.sort()
#print("sorted directories names \n",sub_dirs)
features_list = []


for label, sub_dir in enumerate(sub_dirs):
    print("label",label)
    print("sub_dir",sub_dir)
    for file_name in glob.glob(os.path.join(dataset_path,sub_dir,"*.ogg")):
            #print("Extracting file ", file_name)
            mfccs = get_features(file_name)
            features_list.append([mfccs,label])
features_df = pd.DataFrame(features_list,columns = ['feature','class_label'])
#print(features_df.head())    


X = np.array(features_df.feature.tolist())
y = np.array(features_df.class_label.tolist())
print("X shape",X.shape)
print("y shape",y.shape)


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     
u1, c1 = np.unique(y_train, return_counts=True)
print ("train labels",np.asarray((u1, c1)).T)

u2, c2 = np.unique(y_test, return_counts=True)
print ("test labels",np.asarray((u2, c2)).T)
#mfccs = get_features("/home/akansha/Desktop/others/SRC/Environmental-Sound-Classification-master/dataset/001 - Dog bark/1-30226-A.ogg")
#print(mfccs.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

def delta(x,i,classs):
    '''
    fonction indicatrice de la classe i
    '''
    n,m = len(x),len(classs)
    
    if (n != m):
        print('vectors of differents sizes, cannot operate delta')
        
    tmp = i*np.ones(n)-classs

    for k in range(n):
        if tmp[k]==0:
            tmp[k]=1
        else:
            tmp[k]=0 
            
    return tmp*x

#The residu function returns the class which minimizes the reconstruction error following the norm 2
def residu(y,A,x,class_x):
    #print("class_x",class_x)
    '''
    renvoie les residus pour chaque classe.
    '''
    k = int(np.max(class_x)+1)
    #print()
    r = np.zeros(k)
    
    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))
        
    return r

# ** Utility **: find the class of an element of the test set for the ESC50 dataset
def find_class(i):
    return int(i)/10

from sklearn.linear_model import Lasso

pred=[]
true=[]
for i in range(120):
    print("i",i)
    clf = Lasso(alpha=0.02,max_iter=50000)
    y = X_test[i,:]
    print("xtrain shape",X_train.shape)
    print("y shape",y.shape)
    clf.fit(X_train.T,y)
    x = clf.coef_ 
    print(len(x))
    #plt.show()
    pred_class = np.argmin(residu(X_test[i,:],X_train.T,x,y_train))
    pred.append(pred_class)
    #print("Real class: ", y_test[i])
    true.append(y_test[i])
    print("Predicted class: ", pred_class)
    
from sklearn.metrics import accuracy_score
acc= accuracy_score(true, pred)
print("accuracy",acc)
  