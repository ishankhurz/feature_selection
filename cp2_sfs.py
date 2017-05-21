# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:00:02 2017

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

training = pd.read_table('C:\Training_Data.txt')
n=training.columns
testing  = pd.read_table('C:\Testing_Data.txt')
feature_raw = np.array(training.iloc[:,0:71])
output_vector = np.array(training.iloc[:,-1])
feature_test = np.array(testing.iloc[:,0:71])
output_test = np.array(testing.iloc[:,-1])

#lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1)
knn = KNeighborsClassifier(n_neighbors=3)
sfs1 = SFS(knn, k_features=1, forward=True, floating=False, verbose=2,scoring='accuracy',cv=0)
sfs1 = sfs1.fit(feature_raw, output_vector)
error_estimate = 1-sfs1.k_score_
feature_train = feature_raw[:,sfs1.k_feature_idx_]
knn.fit(feature_train,output_vector)
feature_testing = feature_test[:,sfs1.k_feature_idx_]
output_pred = knn.predict(feature_testing)
acc = float((output_test == output_pred).sum()) / output_pred.shape[0]
testset_error = 1-acc
#gene8 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]] + ', ' + n[sfslda.k_feature_idx_[3]] + ', ' + n[sfslda.k_feature_idx_[4]] + ', ' + n[sfslda.k_feature_idx_[5]] + ', ' + n[sfslda.k_feature_idx_[6]] + ', ' + n[sfslda.k_feature_idx_[7]]
#gene7 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]] + ', ' + n[sfslda.k_feature_idx_[3]] + ', ' + n[sfslda.k_feature_idx_[4]] + ', ' + n[sfslda.k_feature_idx_[5]] + ', ' + n[sfslda.k_feature_idx_[6]]
#gene6 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]] + ', ' + n[sfslda.k_feature_idx_[3]] + ', ' + n[sfslda.k_feature_idx_[4]] + ', ' + n[sfslda.k_feature_idx_[5]]
#gene5 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]] + ', ' + n[sfslda.k_feature_idx_[3]] + ', ' + n[sfslda.k_feature_idx_[4]]
#gene4 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]] + ', ' + n[sfslda.k_feature_idx_[3]]
#gene3 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]] + ', ' + n[sfslda.k_feature_idx_[2]]
#gene2 = n[sfslda.k_feature_idx_[0]] + ', ' + n[sfslda.k_feature_idx_[1]]
#gene1 = n[sfs1.k_feature_idx_]
