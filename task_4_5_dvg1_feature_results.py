import csv
import math
import numpy as np
from csv import reader
import sklearn
import sklearn.preprocessing
assert sklearn.__version__ >= "0.20"
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

def readData(number): #method to read the data, the parameter makes reference to the specific dataset to be readed, used to proof normalization improvements
    dataset = list() #initialize the list for x and y
    outputs = list()
    with open('y_train_smpl_'+str(number)+'.csv', 'r') as fileY: #iterate trough all the rows of class values
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

def readData(): #method to read the data, this method is similar to the top one only changing the dataset to be picked
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
    dataset.pop(0) #substracting the first value as it is unrelated
    return dataset

def BiLisToBiArray(dataset): # transforming a list of lists to a bi-dimensional array
    array=[]
    for instance in dataset:
        array.append(np.asarray(instance)) #adding the list as array to the array
    return np.asarray(array)

def read_feat(n): #reading the files of the top N features
    list=[]
    with open('top'+str(n)+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            for pixel in row:
                list.append(int(pixel)) # putting all the feaures together as an array
        print(list)
    return list

def Gaussian(NTOPS,numero): #main method of the code, obtaining the dataset, fitting the model and testing accuracy
    feat_selected=np.asarray(read_feat(NTOPS))
    print(len(feat_selected))

    dataset=np.array(readData(numero)) #read the data
    np.random.shuffle(dataset)
    listClassValues=dataset[:, -1]
    #feat_selected=getNTops(10,False)
    temp=[]
    temp.append(dataset[:,0])
    temp=np.asarray(temp)
    temp = temp.reshape(temp.shape[1], 1)
    for pixel in feat_selected:
        temp=np.asarray(temp)
        column=dataset[:, int(pixel)]
        column = column.reshape(column.shape[0], 1)
        temp=np.hstack((temp,column))
    dataset=np.array(temp)

    listAttributesValues=dataset[:, :-1] #get all the attributes values
    print("---- previous to normalization")
    print(listClassValues.shape)
    print("---- after normalization")
    listAttributesValues=sklearn.preprocessing.normalize(listAttributesValues, norm='l2', axis=1, copy=True, return_norm=True)[0] #normalize the attributes values, [0] as it creates a list with the array inside
    print(listClassValues.shape)

    #instantiate the target model
    clf = GaussianNB()
    #clf = MultinomialNB()
    #clf = BernoulliNB()

    #set the training size
    trainSize=int(len(listAttributesValues)*0.7)

    #split train and test
    Attributes_train, Attributes_test, Class_train, Class_test = listAttributesValues[:trainSize], listAttributesValues[trainSize:], listClassValues[:trainSize], listClassValues[trainSize:]

    #fit the data to the model
    clf.fit(Attributes_train, Class_train)

    #get the results for the training classification
    y_train_pred = cross_val_predict(clf, Attributes_train, Class_train, cv=10)
    print(metrics.confusion_matrix(Class_train, y_train_pred)) #print the confusion, matrix
    predicted = clf.predict(Attributes_test) #predict the results to the test
    print(type(Class_test[0]))
    print(predicted)
    print("Accuracy:", metrics.accuracy_score(Class_test, predicted)) #print the accuracy obtained by the model

np.random.seed(42) #specifying the seed to allow comparison of results between different runs

for i in range (0,10): #running the top 10 features (best one) through all the different datasets to prove the advantages of normalization
    Gaussian(10,i)

#Gaussian(5) #five top features run
#Gaussian(10) #ten top features run
#Gaussian(20) #twenty top features run



