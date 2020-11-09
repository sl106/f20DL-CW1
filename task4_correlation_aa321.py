
import math
import pandas as pd
from collections import Counter
import numpy as np

# some models for classification and regression
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# model selection tools
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

# metrics
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

#Read the datsets
dataset = pd.read_csv('x_train_gr_smpl.csv')

#Shuffle the data
shuffle(dataset, random_state = 30)

#label for which the pixels are found
x = pd.read_csv('y_train_smpl_5.csv')
shuffle(x,random_state = 30)

def correlation():

    dataset.insert(2304, "y", x, allow_duplicates = False)
    target_col_name = 'y'
    feature_target_corr = {}
    for col in dataset:
        if target_col_name != col:
            feature_target_corr[col + '_' + target_col_name] = pearsonr(dataset[col], dataset[target_col_name])[0]
    #print("Feature-Target Correlations")
    #print(feature_target_corr)

    maxvals(feature_target_corr, 10)

def maxvals(d,n):
    k = Counter(d)

    # Finding n highest values
    max = k.most_common(n)

    print("n highest values:")
    print("Pixel: Correlation")

    for i in max:
        print(i[0], " :", i[1], " ")

correlation()
