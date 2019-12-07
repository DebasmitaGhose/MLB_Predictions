#!/usr/bin/env python

import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import collections
from sklearn.impute import SimpleImputer

X = []

with open('../../Data/temp_mlb_predictions.csv') as csvfile:
    mlb_row = csv.reader(csvfile, delimiter=',')
    for row in mlb_row:
        X.append(row)

X = X[1:] # remove the first row as it is the header     
X_arr = np.array(X) # convert to a numpy array

y = X_arr[:,1] # 2nd column is the label
y = y.astype(int) # convert the label from string to int
X_arr = X_arr[:,6:] # remove first 6 features as they are not useful for ML

X_arr[X_arr == 'NA'] = -100 # convert NA to nan

X_arr = X_arr.astype(float)

X_arr[X_arr == -100] = np.nan # convert NA to nan


#Remove the columns from the feature set if the percentange of nan values are more than 50%
X_select = []
select_col = []
threshold = 50
print("Column IDs with percentage of missing values")
for i in range(X_arr.shape[1]):
    col_i = X_arr[:,i]
    num_nan_in_col = col_i[np.isnan(col_i)].shape[0]
    col_size = float(col_i.shape[0]) 
    num_nan = num_nan_in_col / col_size   
    percent_nan = num_nan*100    
    print(i, percent_nan)
    if percent_nan < threshold:
        select_col.append(i)
        X_select.append(col_i)

X_select = np.array(X_select)
X_select = np.transpose(X_select)


#Replace missing values with mean of the column
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_select)
X_without_nan = imp.transform(X_select)

##### SANITY CHECK --> to check the replacement of missing values #########
#print(np.where(np.isnan(X_select[:,10])==True))
#print(X_select[55])
#print(X_without_nan[55])








                  