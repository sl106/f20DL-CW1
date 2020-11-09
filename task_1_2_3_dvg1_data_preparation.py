import csv
import math
import numpy as np
from csv import reader
import sklearn
import sklearn.preprocessing
assert sklearn.__version__ >= "0.20"
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

from decimal import *

learningRate=0.1
epochs=10000


def readData():
    dataset = list()
    outputs = list()
    with open('y_train_smpl.csv', 'r') as fileY: #iterate trough all the rows of class values
        csv_readerY = reader(fileY)
        for row in csv_readerY:
            if not row:
                continue
            outputs.append(int(row[0])) #store the class values
        print(len(outputs))
    with open('x_train_gr_smpl.csv', 'r') as fileX: #iterate trough all the rows of atributes values
        csv_readerX = reader(fileX)
        cnt=0
        for row in csv_readerX:
            if not row:
                continue
            rowAsInteger = list(map(float, row)) #convert the row values to float from strings
            rowAsInteger.append(outputs[cnt]) #add the class value at the end of the instance
            cnt=cnt+1;
            dataset.append(rowAsInteger) #add the row value
        print(len(outputs))
    dataset.pop(0)
    return dataset

def BiLisToBiArray(dataset): #converting a list of lists to a bidimensional array
    array=[]
    for instance in dataset:
        array.append(np.asarray(instance))
    return np.asarray(array)

dataset=readData() #read data
np.random.shuffle(dataset) #randomize the instances
datasetAsArray=BiLisToBiArray(dataset) #convert the list into a bidimensional array
listClassValues=datasetAsArray[:, -1] #get all the class values, now we have it in the same order
listAttributesValues=datasetAsArray[:, :-2] #get all the attributes values
print("---- previous to normalization")
print(listAttributesValues[0])
print("---- after normalization")
listAttributesValues=sklearn.preprocessing.normalize(listAttributesValues, norm='l2', axis=1, copy=True, return_norm=True)[0] #normalize the attributes values, [0] as it creates a list with the array inside
print(listAttributesValues[0])
print("Rows = ",len(listAttributesValues),", Columns = ",len(listAttributesValues[0]))
print(listClassValues.shape[0])

#instatiate gaussian bayesian network
clf = GaussianNB()

#define test size
testSize=int(len(listAttributesValues)*0.7)

#separate the train and test sets
Attributes_train, Attributes_test, Class_train, Class_test = listAttributesValues[:testSize], listAttributesValues[testSize:], listClassValues[:testSize], listClassValues[testSize:]

#fit the data and the classes to train the network
clf.fit(Attributes_train, Class_train)

#get the results of the training
y_train_pred = cross_val_predict(clf, Attributes_test, Class_test, cv=10)
print(metrics.confusion_matrix(Class_test, y_train_pred)) #print the matrix
predicted = clf.predict(Attributes_test) #get the predictions for the test


print("Accuracy:", metrics.accuracy_score(Class_test, predicted)) #print accuracy


