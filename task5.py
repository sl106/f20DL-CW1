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


keys5 = []
keys10 = []
keys20 = []

def correlations(path):
    X = pd.read_csv("resource/x_train_gr_smpl.csv")
    y = pd.read_csv(path)
    X.insert(2304, "class", y, allow_duplicates=False)
    shuffle(X, random_state = 0)
    corr_matrix = X.corr()
    return corr_matrix["class"].sort_values(ascending=False).head(21)



def makeNewSet(klist):
    tmp = klist.keys().values.tolist()
    tmp.remove("class")
    keys5.extend(tmp[:5])
    keys10.extend(tmp[:10])
    keys20.extend(tmp)


def removeDupliate(klist):
    tmp = list(set(klist))
    return tmp


def run():
    for i in range(0, 10):
        # print("Top 10 most correlated pixels of sign " + str(i) + ":")
        path = "resource/y_train_smpl_" + str(i) + ".csv"
        makeNewSet(correlations(path))


def newCVS():
    print("running correlations")
    run()
    print("removing duplicates")
    nkeys5 = removeDupliate(keys5)
    nkeys10 = removeDupliate(keys10)
    nkeys20 = removeDupliate(keys20)
    X = pd.read_csv("resource/x_train_gr_smpl.csv")
    print("creating new dataframe")
    new5 = X[nkeys5].copy()
    new5 = new5.reindex(sorted(new5.columns), axis=1)
    new10 = X[nkeys10].copy()
    new10 = new10.reindex(sorted(new10.columns), axis=1)
    new20 = X[nkeys20].copy()
    new20 = new20.reindex(sorted(new20.columns), axis=1)
    print("exporting to csv files")
    new5.to_csv(r'resource/new51.csv', index = False, header=True)
    new10.to_csv(r'resource/new101.csv', index=False, header=True)
    new20.to_csv(r'resource/new201.csv', index=False, header=True)
    print("done!")

# newCVS()

