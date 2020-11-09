
import sys
assert sys.version_info >= (3, 5)
import sklearn
import numpy as np
import os
import tarfile
import urllib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
#import X
from sklearn.utils import shuffle

X = pd.read_csv("resource/x_train_gr_smpl.csv")
print(X.head())

#import Y
Y = pd.read_csv("resource/y_train_smpl_0.csv")

#insert Y to last column of X
X.insert(2304, "class", Y, allow_duplicates = False)
X = shuffle(X,random_state=0)
# print(X.head())
# print(X.shape)

# #do correlation matrix
corr_matrix = X.corr()
tmp = corr_matrix["class"].sort_values(ascending=False).head(11)

print(tmp.keys())


new10 = X[tmp.keys().values.tolist()].copy()
print(new10.head())
# print(new10.shape())

def correlations(path):
    X = pd.read_csv("resource/x_train_gr_smpl.csv")
    y = pd.read_csv(path)
    X.insert(2304, "class", y, allow_duplicates=False)
    shuffle(X, random_state = 0)
    corr_matrix = X.corr()
    return corr_matrix["class"].sort_values(ascending=False).head(11)


def run():
    for i in range(0, 10):
        print("Top 10 most correlated pixels of sign " + str(i) + ":")
        path = "resource/y_train_smpl_" + str(i) + ".csv"
        correlations(path)


# run()
