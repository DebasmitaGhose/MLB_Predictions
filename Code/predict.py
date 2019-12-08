#!/usr/bin/env python

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import csv


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


# Replace missing values with mean of the column
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_select)
X_without_nan = imp.transform(X_select)

##### SANITY CHECK --> to check the replacement of missing values #########
#print(np.where(np.isnan(X_select[:,10])==True))
#print(X_select[55])
#print(X_without_nan[55])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_without_nan, y, test_size=0.2, random_state=42)

print('Number of training examples',X_train.shape)
print('Number of test samples',X_test.shape)

# Logistic Regressor Classifier
print("LOGISTIC REGRESSION")
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
Y_predict = clf.predict(X_test)
print("Predictions", Y_predict)
print("Classification Accuracy = ", clf.score(X_test, y_test))
print("Precision, Recall, F-Score:", precision_recall_fscore_support(y_test, Y_predict, average='weighted'))

# Support Vector Machine(SVM) Classifier
print("SUPPORT VECTOR MACHINE")
clf = SVC(gamma='auto').fit(X_train, y_train)
Y_predict = clf.predict(X_test)
print("Predictions", Y_predict)
print("Classification Accuracy = ", clf.score(X_test, y_test))
print("Precision, Recall, F-Score:", precision_recall_fscore_support(y_test, Y_predict, average='weighted'))








                  