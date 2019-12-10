#!/usr/bin/env python

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import csv


X_train = []
X_test = []

########### TRAIN ###############

with open('../../Data/training_data.csv') as csvfile:
    mlb_row = csv.reader(csvfile, delimiter=',')
    for row in mlb_row:
        X_train.append(row)

########## TEST ###################

with open('../../Data/test_data.csv') as csvfile:
    mlb_row = csv.reader(csvfile, delimiter=',')
    for row in mlb_row:
        X_test.append(row)

########## TRAIN ##################

X_train = X_train[1:] # remove the first row as it is the header     
X_arr_train = np.array(X_train) # convert to a numpy array

X_test = X_test[1:] # remove the first row as it is the header     
X_arr_test = np.array(X_test) # convert to a numpy array

y_train = X_arr_train[:,1] # 2nd column is the label
y_train = y_train.astype(int) # convert the label from string to int
X_arr_train = X_arr_train[:,6:] # remove first 6 features as they are not useful for ML

X_arr_train[X_arr_train == 'NA'] = -100 # convert NA to nan

X_arr_train = X_arr_train.astype(float)

X_arr_train[X_arr_train == -100] = np.nan # convert NA to nan

########### TEST ####################

y_test = X_arr_test[:,1] # 2nd column is the label
y_test = y_test.astype(int) # convert the label from string to int
X_arr_test = X_arr_test[:,6:] # remove first 6 features as they are not useful for ML

X_arr_test[X_arr_test == 'NA'] = -100 # convert NA to nan

X_arr_test = X_arr_test.astype(float)

X_arr_test[X_arr_test == -100] = np.nan # convert NA to nan

########## Remove the columns from the feature set if the percentange of nan values are more than 50%
X_select_train = []
X_select_test = []
select_col_train = []
select_col_test = []
threshold = 50
print("Column IDs with percentage of missing values")
for i in range(X_arr_train.shape[1]):
    col_i_train = X_arr_train[:,i]
    col_i_test = X_arr_test[:,i]
    num_nan_in_col = col_i_train[np.isnan(col_i_train)].shape[0]
    col_size = float(col_i_train.shape[0]) 
    num_nan = num_nan_in_col / col_size   
    percent_nan = num_nan*100    
    print(i, percent_nan)
    if percent_nan < threshold:
        #print(X_train[0][i])
        select_col_train.append(i)
        select_col_test.append(i)
        X_select_train.append(col_i_train)
        X_select_test.append(col_i_test)

X_select_train = np.array(X_select_train)
X_select_train= np.transpose(X_select_train)

X_select_test = np.array(X_select_test)
X_select_test = np.transpose(X_select_test)


############ TRAINING ########################
# Replace missing values with mean of the column
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_select_train)
X_without_nan_train = imp.transform(X_select_train)
print("Size of the Training set:", X_without_nan_train.shape)

########### TESTING ############################
# Replace missing values with mean of the column
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_select_test)
X_without_nan_test = imp.transform(X_select_test)
print("Size of the Testing set:", X_without_nan_test.shape)


X_train = X_without_nan_train
X_test = X_without_nan_test

##### SANITY CHECK --> to check the replacement of missing values #########
#print(np.where(np.isnan(X_select[:,10])==True))
#print(X_select[55])
#print(X_without_nan[55])

# Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X_without_nan, y, test_size=0.2, random_state=42)

#print('Number of training examples',X_train.shape)
#print('Number of test samples',X_test.shape)

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

#print(Y_predict.type)
filename = 'predictions.csv'

Y_predict_list = Y_predict.tolist()
print(len(Y_predict_list))

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    for i in zip(Y_predict_list):
		csvwriter.writerow((i))

    # writing the data rows 
    #csvwriter.writerows(Y_predict)







                  