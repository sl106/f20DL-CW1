# Python ≥3.5 is required
import sys

from sklearn.model_selection import train_test_split

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
from numpy import loadtxt
import seaborn as sns;

sns.set()

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures


path = r"resource/x_train_gr_smpl.csv"
datapath = open(path, 'r')
X = loadtxt(datapath, delimiter=",")

path = r"resource/y_train_smpl.csv"
datapath = open(path, 'r')
y = loadtxt(datapath, delimiter=",")

# Prepare data
print(X.shape)
print(y.shape)

# #randomise the data
from scipy.sparse import coo_matrix

X_sparse = coo_matrix(X)

from sklearn.utils import shuffle

X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)

# split data to training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=17)

# run NB - BernoulliNB
BernNB = BernoulliNB(binarize=True)
BernNB.fit(X_train, y_train)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))

# run NB - Multinomial
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
y_pred = MultiNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))

GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
y_pred = GausNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_expect, y_pred))


