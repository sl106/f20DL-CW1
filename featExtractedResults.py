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
from correlationDVG1 import getNTops
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

def BiLisToBiArray(dataset):
    array=[]
    for instance in dataset:
        array.append(np.asarray(instance))
    return np.asarray(array)

dataset=np.array(readData()) #read data
np.random.shuffle(dataset)
listClassValues=dataset[:, -1]
feat_selected=getNTops(15,False)
temp=[]
temp.append(dataset[:,0])
temp=np.asarray(temp)
temp = temp.reshape(temp.shape[1], 1)
for pixel in feat_selected:
    temp=np.asarray(temp)
    column=dataset[:,int(pixel)]
    column = column.reshape(column.shape[0], 1)
    temp=np.hstack((temp,column))
dataset=np.array(temp)
 #randomize the instances
#datasetAsArray=BiLisToBiArray(dataset) #convert the list into a bidimensional array
 #get all the class values, now we have it in the same order
listAttributesValues=dataset[:, :-1] #get all the attributes values
print("---- previous to normalization")
print(listClassValues.shape)
print("---- after normalization")
listAttributesValues=sklearn.preprocessing.normalize(listAttributesValues, norm='l2', axis=1, copy=True, return_norm=True)[0] #normalize the attributes values, [0] as it creates a list with the array inside
print(listClassValues.shape)
#listClassValues=listClassValues.reshape(listClassValues.shape[0], 1) #reshaping it to be a column instead of a row (adding a dimension)
#print(listClassValues.shape[0])
#datasetPrepared=np.hstack((listAttributesValues,listClassValues)) #adding the (unnormalized) column of class values
#print("---- adding class value")
#print("Rows = ",len(datasetPrepared),", Columns = ",len(datasetPrepared[0]))
#clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42) #define the stochastic gradiant descent classifier
#clf.fit(listAttributesValues, listClassValues) #train of the classifier
clf = GaussianNB()
testSize=int(len(listAttributesValues)*0.7)

Attributes_train, Attributes_test, Class_train, Class_test = listAttributesValues[:testSize], listAttributesValues[testSize:], listClassValues[:testSize], listClassValues[testSize:]
clf.fit(Attributes_train, Class_train)

y_train_pred = cross_val_predict(clf, Attributes_train, Class_train, cv=10)
print(metrics.confusion_matrix(Class_train, y_train_pred))
predicted = clf.predict(Attributes_test)

print("Accuracy:", metrics.accuracy_score(Class_test, predicted))




