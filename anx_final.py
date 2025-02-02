# -*- coding: utf-8 -*-
"""mw_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1etMRl5M8G_1UWyh1BJXV9-gkCmzD7uF9
"""

# Commented out IPython magic to ensure Python compatibility.
# Imports
import sys
import os
import numpy as np
import pandas as pd
# import keras
# from keras import optimizers
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation
# from tensorflow.keras.layers import BatchNormalization
# from keras.callbacks import ModelCheckpoint
# from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from numpy import argmax
import seaborn as sns

from imp import reload

import matplotlib.pyplot as plt
# %matplotlib inline
import csv
import scipy as sp
import getFeatures as gf
import mne

from pathlib import Path
# %matplotlib inline
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn import datasets, neighbors, linear_model, tree
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#1-KNN
from sklearn.neighbors import KNeighborsClassifier
#2-RandomForest
from sklearn.ensemble import RandomForestClassifier
#3-SVM
from sklearn import datasets, svm
#4-DecisionTree
from sklearn import tree
#5-LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#6-GaussianNB
from sklearn.naive_bayes import GaussianNB
#7-MLP
from sklearn.neural_network import MLPClassifier

def csv_to_mne_raw(csv_file_path):
  #read the csv file into a pandas dataframe
  df = pd.read_csv(csv_file_path)

  #get the starting 10 columns corresponding to the 8 channels and the index data
  df1 = df.iloc[:,0:10]

  #also keep the timestamp, Cognitive Workload and Anxiety column
  df1['Timestamp'] = pd.to_datetime(df[' Timestamp (Formatted)'])
  df1['Cognitive Workload'] = df['Cognitive_Wokload']
  df1['Anxiety'] = df['Anxiety']

  #reindex the columns
  df1 = df1.reindex(columns=[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',
                            ' EXG Channel 3', ' EXG Channel 4', ' EXG Channel 5',
                            ' EXG Channel 6', ' EXG Channel 7', 'Changed Sample Index',
                            'Sample Index','Cognitive Workload','Anxiety','Timestamp'])


  #drop the sample index column
  df1 = df1.drop(['Sample Index'], axis=1)

  #rename the channels
  df1 = df1.rename(columns={' EXG Channel 0': 'eeg1', ' EXG Channel 1': 'eeg2',
                            ' EXG Channel 2': 'eeg3', ' EXG Channel 3': 'eeg4',
                              ' EXG Channel 4': 'eeg5', ' EXG Channel 5': 'eeg6',
                              ' EXG Channel 6': 'eeg7', ' EXG Channel 7': 'eeg8',
                              'Changed Sample Index':'Index'})


  #convert dataframe to numpy array
  data = df1[['eeg1','eeg2','eeg3','eeg4','eeg5','eeg6','eeg7','eeg8']].to_numpy()
  # data = df1.to_numpy()

  #convert the data to float32
  data = data.astype('float32')

  # Scaling Data with the scale factor S = (4500000)/4/(2**23-1)
  data = data*4500000/4/(2**23-1)

  #get transpose of the data
  data = data.T

  # Create some metadata
  n_channels = 8
  sampling_freq = 250  # in Hertz
  ch_types = ['eeg'] * n_channels
  ch_names = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8']
  info = mne.create_info( ch_names=ch_names, sfreq=sampling_freq,ch_types=ch_types)
  # print(info)
  raw = mne.io.RawArray(data, info)
  return raw,df1.iloc[0]['Anxiety']

features = ['Coeffiecient of Variation','Mean of Vertex to Vertex Slope','Variance of Vertex to Vertex Slope',
         'Hjorth_Activity','Hjorth_Mobility','Hjorth_Complexity',
         'Kurtosis','2nd Difference Mean','2nd Difference Max',
         'Skewness','1st Difference Mean','1st Difference Max',
         'FFT Delta MaxPower','FFT Theta MaxPower','FFT Alpha MaxPower','FFT Beta MaxPower','Delta/Theta','Delta/Alpha','Theta/Alpha','(Delta+Theta)/Alpha',
         '1Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '2Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '3Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '4Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '5Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '6Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '7Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         '8Wavelet Approximate Mean','Wavelet Approximate Std Deviation','Wavelet Approximate Energy','Wavelet Detailed Mean','Wavelet Detailed Std Deviation','Wavelet Detailed Energy','Wavelet Approximate Entropy','Wavelet Detailed Entropy',
         'AR1','AR2','AR3','AR4','AR5','AR6','AR7','AR8','AR9','AR10','AR11','AR12','AR13','AR14','AR15','AR16','AR17','AR18',
         'AR19','AR20','AR21','AR22','AR23','AR24']

def createFeatures(files,data_dir,flag):

    #convert files to a list

    if(flag==0):
        features_csv_file = 'features_train_anx.csv'
    else:
        features_csv_file = 'features_test_anx.csv'
    features_csv_file_path = data_dir/features_csv_file
    #delete the file if it exists
    if os.path.exists(features_csv_file_path):
        os.remove(features_csv_file_path)


    with open(features_csv_file_path, "a",newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(features)
        for counter in range(len(files)):
            data_path = files[counter]
            print(data_path)
            # read the csv file and get a mne raw object
            data,label = csv_to_mne_raw(data_path)
            # get the data in ndarray
            raw_data = data.get_data()

            #apply a band pass filter of 1-40Hz
            filtered_data = mne.filter.filter_data(raw_data,sfreq=250,l_freq=1,h_freq=40)

            # sigbufs = [l for l in filtered_data]
            sigbufs = [l for l in raw_data]
            sigbufs = np.array(sigbufs)
            sigbufs = sigbufs.transpose()
            sigbufs = sigbufs.astype(float)

            print(sigbufs.shape)
            # continue

            #get the features
            for i in np.arange(0,185):
                temp_features = []
                epoch = sigbufs[i*250:(i+1)*250,:]
                if (len(epoch)==0 or len(epoch[0]) == 0):
                    break

                #Coeffeicient of Variation
                temp_features.append(gf.coeff_var(epoch))

                #Mean of Vertex to Vertex Slope
                temp_features.append(gf.slope_mean(epoch))

                #Variance of Vertex to Vertex Slope
                temp_features.append(gf.slope_var(epoch))

                #Hjorth Parameters
                feature_list = gf.hjorth(epoch)
                for feat in feature_list:
                    temp_features.append(feat)

                #Kurtosis
                temp_features.append(gf.kurtosis(epoch))

                #Second Difference Mean
                temp_features.append(gf.secDiffMean(epoch))

                #Second Difference Max
                temp_features.append(gf.secDiffMax(epoch))

                #Skewness
                temp_features.append(gf.skewness(epoch))

                #First Difference Mean
                temp_features.append(gf.first_diff_mean(epoch))

                #First Difference Max
                temp_features.append(gf.first_diff_max(epoch))

                # print(type(epoch))
                epoch_t = epoch.transpose()
                #FFT Max Power - Delta, Theta, Alpha & Beta Band!
                # feature_list  =  maxPwelch(epoch,128)
                feature_list  =  gf.maxPwelch(epoch_t,250)
                for feat in feature_list:
                    temp_features.append(feat)
                #FFT Frequency Ratios
                temp_features.append(feature_list[0]/feature_list[1])
                temp_features.append(feature_list[0]/feature_list[2])
                temp_features.append(feature_list[1]/feature_list[3])
                temp_features.append((feature_list[0] + feature_list[1])/feature_list[2])

                #Wavelet Fetures!      lineterminator
                feature_list = gf.wavelet_features(epoch,8)
                for feat in feature_list:
                    temp_features.append(feat)

                #Autoregressive model Coefficients
                feature_list = gf.autogressiveModelParametersBurg(epoch)
                for feat in feature_list:
                    temp_features.append(feat.real)

                temp_features.append(label)
                writer.writerow(temp_features)

    if(flag==0):
        r = csv.reader(open('features_train_anx.csv'))
    else:
        r = csv.reader(open('features_test_anx.csv'))

    lines = [l for l in r]
    for i in range(len(lines[1])-1):
        columns = []
        for j in range(1,len(lines)):
            columns.append(float(lines[j][i]))
        mean = np.mean(columns,axis = 0)
        #print('\nMean = ',mean)
        std_dev  = np.std(columns,axis = 0)
        #print('\nSTD Deviation = ',std_dev)
        for j in range(1,len(lines)):
            lines[j][i] = (float(lines[j][i])-mean)/std_dev

    if(flag==0):
        norm_file = 'Normalizedfeatures_train_anx.csv'
    else:
        norm_file = 'Normalizedfeatures_test_anx.csv'
    writer = csv.writer(open(norm_file, 'w'))
    writer.writerows(lines)
    print('Done!')

files = []
data_dir = Path.cwd()
user_data_path = data_dir/'User_data'
for user in os.listdir(user_data_path):
    for session in os.listdir(user_data_path/user):
        for file in os.listdir(user_data_path/user/session):
            if file.endswith('.csv') and not file.endswith('_HEP.csv'):
                path = user_data_path/user/session/file
                path = str(path)
                files.append(path)

def KNN(X_train, X_test, y_train, y_test):
    # KNN
    model = neighbors.KNeighborsClassifier(n_neighbors = 3)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None),y_test_pred

def RandomForest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None), y_test_pred

def SVM(X_train, X_test, y_train, y_test):
    def evaluate_on_test_data(model=None):
        predictions = model.predict(X_test)
        correct_classifications = 0
        for i in range(len(y_test)):
            if predictions[i] == y_test[i]:
                correct_classifications += 1
        accuracy = correct_classifications/len(y_test)
        return accuracy,predictions

    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)
    acc,pred = evaluate_on_test_data(model)
    return acc,pred

def MLP(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(25,), random_state=1)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None),y_test_pred

#5-DecisionTree
def DecisionTree(X_train, X_test, y_train, y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None),y_test_pred

def LDA(X_train, X_test, y_train, y_test):
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None) #shrinkage='auto'
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return metrics.accuracy_score(y_test, y_test_pred, normalize=True, sample_weight=None),y_test_pred

def result_preprocesing(flag):
    if flag == 0:
        file = 'Normalizedfeatures_train_anx.csv'
    else:
        file = 'Normalizedfeatures_test_anx.csv'

    # df = pd.read_csv(file)
    # df.drop('Mean of Vertex to Vertex Slope', axis = 1, inplace = True)
    # df.drop('Coeffiecient of Variation', axis = 1, inplace = True)
    # df.to_csv(file)
    f = open(file)
    attributes=f.readline()
    X = []
    y = []
    for line in f:
        line = line.rstrip().split(',')
        print(line)
        l = [float(i) for i in line]
        X.append(l[:-1])
        y.append(l[-1])

    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

X_train = []
y_train = []
X_test = []
y_test = []
accuracy_dict = {}
for i,item in enumerate(files):
    #create a new list not having the ith item
    train = files[:i] + files[i+1:]
    test = files[i]
    createFeatures(train,data_dir,0)
    X_train, y_train = result_preprocesing(0)
    createFeatures([test],data_dir,1)
    X_test, y_test = result_preprocesing(1)
    accuracy_dict['KNN'] = KNN(X_train, X_test, y_train, y_test)
    accuracy_dict['RandomForest'] = RandomForest(X_train, X_test, y_train, y_test)
    accuracy_dict['SVM'] = SVM(X_train, X_test, y_train, y_test)
    accuracy_dict['MLP'] = MLP(X_train, X_test, y_train, y_test)
    accuracy_dict['DecisionTree'] = DecisionTree(X_train, X_test, y_train, y_test)
    accuracy_dict['LDA'] = LDA(X_train, X_test, y_train, y_test)
    max_accuracy = 0
    for key, value in accuracy_dict.items():
        if value[0] > max_accuracy:
            max_accuracy = value[0]
            max_key = key
            max_pred = value[1]

    path = Path(test).stem
    path = str(path).split('session')
    user = path[0][0:-1]
    session = path[1][1:2]
    session = 'session_'+str(session)
    file_path = Path.cwd()/'User_data'/user/session
    for file in os.listdir(file_path):
        if file.endswith('_HEP.csv'):
            df = pd.read_csv(file_path/file)
            #add a new column 'MW' to the dataframe
            df['ANX'] = max_pred
            df.to_csv(file_path/file)