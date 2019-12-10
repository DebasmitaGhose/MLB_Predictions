#!/usr/bin/env python

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import numpy as np
import csv
import itertools
from scipy import stats
import copy


def plot_confusion_matrix(cm, classes=[0,1],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
 
    print (cm)
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

X_train = []
X_test = []

########### TRAIN ###############

with open('Data/training_data.csv') as csvfile:
    mlb_row = csv.reader(csvfile, delimiter=',')
    for row in mlb_row:
        X_train.append(row)

########## TEST ###################

with open('Data/test_data.csv') as csvfile:
    mlb_row = csv.reader(csvfile, delimiter=',')
    for row in mlb_row:
        X_test.append(row)

######### TRAIN ##################



X_train = X_train[1:] # remove the first row as it is the header     
X_arr_train = np.array(X_train) # convert to a numpy array

X_test = X_test[1:] # remove the first row as it is the header     
X_arr_test = np.array(X_test) # convert to a numpy array

y_train = X_arr_train[:,1] # 2nd column is the label
y_train = y_train.astype(int) # convert the label from string to int
y_train_old = copy.deepcopy(y_train)

X_arr_train = X_arr_train[:,6:] # remove first 6 features as they are not useful for ML

X_arr_train[X_arr_train == 'NA'] = -100 # convert NA to nan

X_arr_train = X_arr_train.astype(float)

X_arr_train[X_arr_train == -100] = np.nan # convert NA to nan

########### TEST ####################

y_test = X_arr_test[:,1] # 2nd column is the label
y_test = y_test.astype(int) # convert the label from string to int

X_player_data = X_arr_test[:,:6] # keep these first six columns for sanity checks
X_arr_test = X_arr_test[:,6:] # remove first 6 features as they are not useful for ML

X_arr_test[X_arr_test == 'NA'] = -100 # convert NA to nan

X_arr_test = X_arr_test.astype(float)

X_arr_test[X_arr_test == -100] = np.nan # convert NA to nan

########### Remove the columns from the feature set if the percentange of nan values are more than 50%
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
    # print(i, percent_nan)
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


############# TRAINING ########################
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


########### OVERSAMPLING ##############
ros = RandomOverSampler(random_state=None, sampling_strategy='minority')
X_train, y_train = ros.fit_resample(X_without_nan_train, y_train_old)
print(ros)

# What is the split after oversampling?
yes = y_train[y_train == 1]
no = y_train[y_train == 0]
print(len(yes), " 1's and ", len(no), " 0's")
print(y_train.shape,'after over_sampling')

X_test = X_without_nan_test


print('Number of training examples',X_train.shape)
print('Number of test samples',X_test.shape)

# Logistic Regressor Cimport itertoolslassifier
print("LOGISTIC REGRESSION")
clf = LogisticRegression(random_state=None).fit(X_train, y_train)
Y_predict_LR = clf.predict(X_test)
Y_predict_LR_probs = clf.predict_proba(X_test)
print("Predictions", Y_predict_LR)
print("Classification Accuracy = ", clf.score(X_test, y_test))
print("Precision, Recall, F-Score:", precision_recall_fscore_support(y_test, Y_predict_LR, average='weighted'))
print("Predicted Probabilities", Y_predict_LR_probs)

# Checking overall distribution of predicted probabilities
print("Min: ", np.min(Y_predict_LR_probs[:,1]))
print("Q1: ", np.quantile(Y_predict_LR_probs[:,1], 0.25))
print("Median: ", np.median(Y_predict_LR_probs[:,1]))
print("Mean: ", np.mean(Y_predict_LR_probs[:,1]))
print("Q3: ", np.quantile(Y_predict_LR_probs[:,1], 0.75))
print("Max: ", np.max(Y_predict_LR_probs[:,1]))


# Do this orocess above a few times and then take the average
preds = []
for i in range(0, 100):
    ros = RandomOverSampler(random_state=i, sampling_strategy='minority')
    X_train, y_train = ros.fit_resample(X_without_nan_train, y_train_old)
    clf = LogisticRegression(random_state=None).fit(X_train, y_train)
    Y_predict_LR_probs = clf.predict_proba(X_test)
    preds.append(Y_predict_LR_probs)
    
overall_preds = np.mean(preds, axis = 0)

print(Y_predict_LR_probs.shape)
print(overall_preds.shape)

# Checking final distribution of predicted probabilities
print("Min: ", np.min(overall_preds[:,1]))
print("Q1: ", np.quantile(overall_preds[:,1], 0.25))
print("Median: ", np.median(overall_preds[:,1]))
print("Mean: ", np.mean(overall_preds[:,1]))
print("Q3: ", np.quantile(overall_preds[:,1], 0.75))
print("Max: ", np.max(overall_preds[:,1]))

print(X_player_data.shape)
print(overall_preds.shape)
print(X_player_data[0])
print(overall_preds[0, 1])

# Get demographic data of the 
final_data = []
for i in range(0, X_player_data.shape[0]):
    row = X_player_data[i].tolist()
    
    # We don't need this value
    del row[1]
    
    # Append to row and then final data
    row.append(overall_preds[i,1])
    final_data.append(row)

print(len(final_data))

filename = 'predictions.csv'

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    print(["playerID", "debutYear", "birthYear", "namefirst", "namelast", "hof_prob"])
    for i in final_data:
        csvwriter.writerow(i)

print("Done successfully!")
           