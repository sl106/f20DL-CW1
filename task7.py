import sys

assert sys.version_info >= (3, 5)
import pandas as pd
from sklearn.utils import shuffle

X = pd.read_csv("DataSets/new200.csv")
y = pd.read_csv("resource/y_train_smpl.csv")
X.insert(139, "class", y, allow_duplicates=False)


X["class"] = X["class"].map({0: "s0", 1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5", 6: "s6", 7: "s7", 8: "s8", 9: "s9"})
print(X.head())
X.to_csv(r'resource/new200-c.csv', index = False, header=True)