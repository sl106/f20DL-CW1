
import sys

assert sys.version_info >= (3, 5)
import pandas as pd
from sklearn.utils import shuffle


# function to run correlation
def correlations(path):
    X = pd.read_csv("resource/x_train_gr_smpl.csv")
    y = pd.read_csv(path)
    X.insert(2304, "class", y, allow_duplicates=False)
    shuffle(X, random_state=0)
    corr_matrix = X.corr()
    return corr_matrix["class"].sort_values(ascending=False).head(11)


# function to iterate through each signs
def run():
    for i in range(0, 10):
        print("Top 10 most correlated pixels of sign " + str(i) + ":")
        path = "resource/y_train_smpl_" + str(i) + ".csv"
        print(correlations(path))


run()
