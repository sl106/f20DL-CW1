#K means Clustering
import sys

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.utils import *
import os
import random
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

#Read the datsets
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import metrics

import utils #importing method from secondary file

data = pd.read_csv('x_train_gr_smpl.csv') #reading samples
labels = pd.read_csv('y_train_smpl.csv') #reading labels

#Shuffle the data
data=shuffle(data, random_state = 87)
labels=shuffle(labels, random_state = 87)

#add the labels
data.insert(2304, "y", labels, allow_duplicates = False)

#normalize the data by row
data=sklearn.preprocessing.normalize(data, norm='l1', axis=1, copy=True, return_norm=True)[0]

#define the train size
trainSize = int(len(data) * 0.7)

#split the data and labels between train and tes
x_train, x_test = np.asarray(data[:trainSize]), np.asarray(data[trainSize:])
y_train, y_test = np.asarray(labels[:trainSize]), np.asarray(labels[trainSize:])

#set up kmeans with 10 clusters as we want 10 ideal groups
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train) #fitting the data
training_labels = kmeans.labels_ #resulting labels
switches_to_make = utils.find_closest_digit_to_centroids(kmeans, x_train, y_train)  # Obtaining the most probable labels (digits) for each region

# prepare the data
utils.treat_data(switches_to_make, training_labels, y_train)

# get predictions for the test data
predictions = kmeans.predict(x_test)

#prepare the resulting data
utils.treat_data(switches_to_make, predictions, y_test)

#label every cluster with the closest ground truth labels and map them for changing the labels
predicted_labels_for_regions=switches_to_make
print(predicted_labels_for_regions)

y_predicted_labelled=[]

#record predicted labels, not indexes of the clusters
for i in range(0,len(predictions)):
    print("Clauster index :",predictions[i]," Predicted label for the region : ",predicted_labels_for_regions[predictions[i]]," Real label : ",y_test[i][-1] )
    y_predicted_labelled.append(int(predicted_labels_for_regions[predictions[i]][0])) #change the cluster index with the one decided previously that matched that cluster

#format the data to meet metrics.accuracy_score requirements
y_predicted_labelled=np.asarray(y_predicted_labelled)
y_test=y_test.reshape(1,y_test.shape[0])
y_predicted_labelled=y_predicted_labelled.reshape(1,y_predicted_labelled.shape[0])
y_test=y_test.astype(np.float)
y_test=y_test[0]
y_predicted_labelled=y_predicted_labelled.astype(np.float)
y_predicted_labelled=y_predicted_labelled[0]

print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_labelled)) #accuracy printing


