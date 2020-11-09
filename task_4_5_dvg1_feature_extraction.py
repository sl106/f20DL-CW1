import csv

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def getNTops(N,printBool): #method to get the N top features from the datasets
    listG=[]  #initialize the global list
    dataX = pd.read_csv("x_train_gr_smpl.csv")  # get value data
    for i in range(0,10): #iterate trought all the dataset
        Y = pd.read_csv("y_train_smpl_"+str(i)+".csv") #get output data
        #adding y as a column at the end of X (better method than the one i implemented)
        X=dataX.copy() #Copy the data of the rows to a new object to avoid reading it multiple times
        X.insert(2304, "class", Y, allow_duplicates = False) #insert the labels of the current y_train
        X = shuffle(X,random_state=0) # better method for shuffling as the other one seems to only works for lists

        #get correlation matrix
        cmatrix = X.corr()
        #we sort the results, the ones with most correlation first, head(11) as class is also taken in account, ["class"] as we
        #want to select the most correlated ones with the "class" column
        results = cmatrix["class"].sort_values(ascending=False).head(N+1)
        if(printBool): #priting or not is one of the parameters
            printTops(i,results) #print the top results for the correlation
        results_att=results.keys() # we want the attribute not the correlation value
        results_att=results_att[1:] #remove class attribute
        for pixel in results_att:
            if pixel not in listG: #avoid duplicates
                listG.append(pixel) #add top feature of this label to general top features
                print(listG)
    return listG #return global list

def printTops(index,list): #print features method
    print("Top 10 most correlated pixels of sign " + str(index) + ":")
    print(list)

def saveToCSV(n,tops): #saving the selected features to csv, so every single time we dont need to obtain it again
    with open('top'+str(n)+'.csv', 'w', newline='') as csvfile:
        pixelwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        pixelwriter.writerow(tops) #write top features to csv

