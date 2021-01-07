#vectorize the matrix and see sparse vector
#dimensionality reduction and see sparse vector

'''
ESC-50 dataset
50 classes
fold1

total vectors=400
train=280
test=120

mfcc coefficients- 20*216
if vectorization - 4320
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np

mfcc_df1 = pd.read_json('/home/akansha/Desktop/john-wright/A/JsonFiles/fold1.json')
df1 = mfcc_df1.sort_values(by=['label'])
mat1=pd.DataFrame(df1).to_numpy()


y1=mat1[:,0]
x1=mat1[:,1]

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.30, random_state=42)

print("y1_train",len(y_train1))


num_class=50

#Indicator function of the i-th class.
#For a given vector it returns just the index corresponding to the i-th class

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
    k = np.max(class_x)+1
    r = np.zeros(k)
    
    for i in range(0,k):
        r[i] = np.linalg.norm(y - np.dot(A,delta(x,i,class_x)))
        
    return r

# ** Utility **: find the class of an element of the test set for the ESC50 dataset
def find_class(i):
    return int(i)/num_class


    
#print(label_y1_train)
m1_train=np.zeros((5600,216)) 
k=0
for i in range(len(x_train1)):
    temp=np.asarray(x_train1[i])
    for j in range(len(temp)):
        m1_train[k]=temp[j]
        k+=1
    #print("s",len(temp))  
#m1_train_norm=np.linalg.norm(m1_train) 

m1_train_norm = normalize(m1_train)

'''
code to convert mfcc matrix to array and store as row for train data
'''
train_flat=np.zeros((280,4320))
st=0
en=20
for i in range(280):
    #print("st",st)
    #print("en",en)
    temp=np.zeros((20,216))
    temp = m1_train_norm[st:en,:]
    temp_flat=temp.flatten()
    #print("a",len(temp_flat))
    train_flat[i]=temp_flat
    st=st+20
    en=en+20
    

        
m1_test=np.zeros((2400,216)) 
k1=0
for i in range(len(x_test1)):
    temp1=np.asarray(x_test1[i])
    for j in range(len(temp1)):
        m1_test[k1]=temp[j]
        k1+=1

m1_test_norm = normalize(m1_test)

'''
code to convert mfcc matrix to array and store as row for train data
'''

test_flat=np.zeros((120,4320))
st=0
en=20
for i in range(120):
    #print("st",st)
    #print("en",en)
    temp=np.zeros((20,216))
    temp = m1_test_norm[st:en,:]
    temp_flat=temp.flatten()
    #print("a",len(temp_flat))
    test_flat[i]=temp_flat
    st=st+20
    en=en+20

        
print("train-flat '''shape",train_flat.shape)
print("test-flat shape",test_flat.shape)


from sklearn.linear_model import Lasso

for i in range(115,116):
    print("i",i)
    clf = Lasso(alpha=0.02,max_iter=50000)
    y = test_flat[i,:]
    #print("xtrain shape",m1_train.shape)
    #print("y shape",y.shape)
    clf.fit(np.transpose(train_flat),y)
    x = clf.coef_ 
    print(len(x))
    plt.plot(x)
    plt.show()
    
  
    
    



