#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 16:20:59 2021

@author: akansha
"""

import numpy as np
from PIL import Image
import os, sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time
import random

random.seed(2017)

#start=time.time()
# Indicator function of the i-th class.
# For a given vector it returns just the index corresponding to the i-th class
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


# residu function returns the class which minimizes the reconstruction error following the norm 2
def residu(y,A,x,class_x):
    '''
    renvoie les residus pour chaque classe.
    '''
    k = np.max(class_x)+1
    r = np.zeros(k)
    
    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))
        
    return r

# find the class of an element of the test set for the Yale Database
def find_class(i):
    return int(i)/12


# read all the images given a path to a database
def read_images(path, sz=None,sz0=168,sz1=192): 
    '''
    Chargement des données
    Si spécifiée, réduction de dimensions incluse
    '''
    c=0
    X,y = [], []
    for dirname , dirnames , filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname , subdirname) 
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path , filename)) 
                    im = im.convert("L")
                    # resize to given size (if given) and check that it's the good size
                    if ((im.size[0] == sz0) & (im.size[1]==sz1)):
                        if (sz is not None):
                            im = im.resize(sz, Image.NEAREST)     
                        X.append(np.asarray(im, dtype=np.uint8)) 
                        y.append(c)
                except IOError:
                    pass
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
            
    print("Images chargees !")
    return [X,y]

mode_reduc_features = 'randomfaces'
size_features = (12,10) 


path_images = '/home/akansha/Desktop/john-wright/CroppedYale' # Path to the dataset

n_components = size_features[0]*size_features[1]
print("Selected feature reduction: ", mode_reduc_features)

if ((mode_reduc_features == 'randomfaces') & (size_features != None)):
    X, y = read_images(path_images, sz=size_features)
elif ((mode_reduc_features != 'reduced_fs_dimension') & (mode_reduc_features != None)):
    X, y = read_images(path_images, sz=None)
else :
    X, y = read_images(path_images, sz=(12,10))
    
    
X_train, X_test = [], []
ytrain, ytest = [], []
indices_train, indices_test = [], []

for i in range(len(X)):
    if i%64==0:
        #compute the indices of the elements to be placed in the test, they are different for each image.
        test1 = np.random.choice(28,5,replace=False)
        test2 = 29 + np.random.choice(35,7,replace=False)
    
   # We create X_test and X_train
    if ((i%64 in test1) or (i%64 in test2)):
        X_test.append(X[i])
        ytest.append(y[i])
        indices_test.append(i)
    else:
        X_train.append(X[i])
        ytrain.append(y[i])
        indices_train.append(i)
        
X_toconcat_train = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_train]
X_toconcat_test = [np.reshape(e,(X_train[0].shape[0]*X_train[0].shape[1],1)) for e in X_test]

# Then concatenation to have a unique matrix
Xtrain = np.concatenate(X_toconcat_train,axis=1) 
Xtest = np.concatenate(X_toconcat_test,axis=1) 

from features_reduction import *

# Normalisation
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.fit_transform(Xtest)


if (mode_reduc_features == 'eigenfaces'):
    Xtrain, Xtest = eigenfaces(Xtrain.T,Xtest.T,n_components=n_components)
    Xtrain, Xtest = Xtrain.T, Xtest.T
elif (mode_reduc_features == 'fisherfaces'):
    Xtrain, Xtest = fisherfaces(Xtrain.T,ytrain,Xtest.T,n_components=n_components)
    Xtrain, Xtest = Xtrain.T, Xtest.T
elif (mode_reduc_features == 'randomfaces'):
    Xtrain, Xtest = randomfaces(Xtrain.T, Xtest.T, n_components=n_components)
    Xtrain, Xtest = Xtrain.T, Xtest.T
    
print("# train images",Xtrain.shape) 
print("# test images",Xtest.shape)   
from sklearn.linear_model import Lasso

#alpha_vec = np.logspace(2,4,20)
#alpha_vec = alpha_vec/(Xtest.shape[0])


pred=[]
true=[]
for i in range(457):
    print("i",i)
    clf = Lasso(alpha=0.02) 
    y = Xtest[:,i]
    clf.fit(Xtrain,y)
    x = clf.coef_
    pred_class = np.argmin(residu(Xtest[:,i],Xtrain,x,ytrain))
    pred.append(pred_class)
    print("Real class: ", ytest[i])
    true.append(ytest[i])
    print("Predicted class: ", pred_class)


from sklearn.metrics import accuracy_score
acc= accuracy_score(true, pred)
print("accuracy",acc)
#print("time taken",time.time() - start)


