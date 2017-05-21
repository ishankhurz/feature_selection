# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:28:11 2017

@author: admin
"""

import itertools
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors

#columns =['Gene Set Found','Error Estimate','Testset Error Estimate ']
#index=range(11)
#df = pd.DataFrame(index=index, columns=columns)
#df = df.fillna(0) 

#Read relevant data as data frame

training = pd.read_table('C:\Training_Data.txt')
n=training.columns
testing  = pd.read_table('C:\Testing_Data.txt')
feature_raw = np.array(training.iloc[:,0:71])
output_vector = np.array(training.iloc[:,-1])
feature_test = np.array(testing.iloc[:,0:71])
output_test = np.array(testing.iloc[:,-1])
feature_size = 3

#function for creating all possible subsets of required size
def findsubsets(S,m):
    return set(itertools.combinations(S, m))
feature_space = findsubsets(range(71),feature_size)
feature_space = np.array(list(feature_space))
b=[]

#find the optimal feature
for ip in feature_space:
    x = feature_raw[:,ip].reshape((feature_raw.shape[0],feature_size))
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1).fit(x,output_vector)
    #clf = neighbors.KNeighborsClassifier(3).fit(x,output_vector)
    a=clf.score(x, output_vector)
    b.append(1-a)
b = np.array(b)
error_estimate = min(b)
index = np.argmin(b)

ip_optimal = feature_raw[:,feature_space[index]].reshape((feature_raw.shape[0],feature_size))
#clf_optimal = neighbors.KNeighborsClassifier(3).fit(ip_optimal,output_vector)
clf_optimal = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1).fit(ip_optimal,output_vector)

x = feature_test[:,feature_space[index]].reshape((feature_test.shape[0],feature_size))
output_pred = clf_optimal.predict(x)
acc = float((output_test == output_pred).sum()) / output_pred.shape[0]
testset_error = 1-acc 
#gene3 = n[feature_space[index,0]] + ', ' + n[feature_space[index,1]] + ', ' + n[feature_space[index,2]]
#gene2 = n[feature_space[index,0]] + ', ' + n[feature_space[index,1]]
#gene1 = n[feature_space[index,0]]
