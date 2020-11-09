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

import utils


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    result=sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return result

data = pd.read_csv('x_train_gr_smpl.csv')
labels = pd.read_csv('y_train_smpl.csv')

#Shuffle the data
data=shuffle(data, random_state = 10000)
labels=shuffle(labels, random_state = 10000)

#data.insert(2304, "y", labels, allow_duplicates = False)
#data=sklearn.preprocessing.normalize(data, norm='l1', axis=1, copy=True, return_norm=True)[0]

trainSize = int(len(data) * 0.7)

x_train, x_test = np.asarray(data[:trainSize]), np.asarray(data[trainSize:])
y_train, y_test = np.asarray(labels[:trainSize]), np.asarray(labels[trainSize:])

#kmeans = KMeans(n_clusters=10).fit(data_train)
#centroids = kmeans.cluster_centers_
#pred_results = kmeans.predict(data_test)
#data_test=np.asarray(data_test)

#true_results=data_test[:,-1]

kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train)
training_labels = kmeans.labels_
switches_to_make = utils.find_closest_digit_to_centroids(kmeans, x_train, y_train)  # Obtaining the most probable labels (digits) for each region

utils.treat_data(switches_to_make, training_labels, y_train)

predictions = kmeans.predict(x_test)

utils.treat_data(switches_to_make, predictions, y_test)
predicted_labels_for_regions=switches_to_make
print(predicted_labels_for_regions)
y_predicted_labelled=[]
for i in range(0,len(predictions)):
    print("Clauster index :",predictions[i]," Predicted label for the region : ",predicted_labels_for_regions[predictions[i]]," Real label : ",y_test[i][-1] )
    y_predicted_labelled.append(int(predicted_labels_for_regions[predictions[i]][0]))

y_predicted_labelled=np.asarray(y_predicted_labelled)
y_test=y_test.reshape(1,y_test.shape[0])
y_predicted_labelled=y_predicted_labelled.reshape(1,y_predicted_labelled.shape[0])
#print(metrics.confusion_matrix(y_test, predictions_labelled))
y_test=y_test.astype(np.float)
y_test=y_test[0]
y_predicted_labelled=y_predicted_labelled.astype(np.float)
y_predicted_labelled=y_predicted_labelled[0]
print(y_test)
print(y_predicted_labelled)
print("Accuracy:", metrics.accuracy_score(y_test, y_predicted_labelled))
#for i in range(0,len(results)):
#    print(results[i])

#print('kmeans: {}'.format(silhouette_score(data, kmeans.labels_,metric='euclidean')))

#pca = PCA(n_components=3)
#pca_result = pca.fit_transform(data.values)
#data['pca-one'] = pca_result[:,0]
#data['pca-two'] = pca_result[:,1]
#data['pca-three'] = pca_result[:,2]

#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#rndperm = np.random.permutation(data.shape[0])

#plt.figure(figsize=(16,10))
#sns.scatterplot(
#    x="pca-one", y="pca-two",
#    hue="y",
#    palette=sns.color_palette("hls", 10),
#    data=data.loc[rndperm,:],
#    legend="full",
#    alpha=0.3
#)
#plt.show()

#df=data
#ax = plt.figure(figsize=(16,10)).gca(projection='3d')
#ax.scatter(
#    xs=df.loc[rndperm,:]["pca-one"],
#    ys=df.loc[rndperm,:]["pca-two"],
#    zs=df.loc[rndperm,:]["pca-three"],
#    c=df.loc[rndperm,:]["y"],
#    cmap='tab10'
#)
#ax.set_xlabel('pca-one')
#ax.set_ylabel('pca-two')
#ax.set_zlabel('pca-three')
#plt.show()
