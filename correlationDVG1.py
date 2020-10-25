import pandas as pd
import numpy as np
from sklearn.utils import shuffle
def getNTops(N,printBool):
    listG=[]
    for i in range(0,1): #iterate trought all the dataset
        X = pd.read_csv("x_train_gr_smpl.csv") #get value data
        Y = pd.read_csv("y_train_smpl_"+str(i)+".csv") #get output data
        #adding y as a column at the end of X (better method than the one i implemented)
        X.insert(2304, "class", Y, allow_duplicates = False)
        X = shuffle(X,random_state=0) # better method as the other one seems to only works for lists

        #get correlation matrix
        cmatrix = X.corr()
        #we sort the results, the ones with most correlation first, head(11) as class is also taken in account, ["class"] as we
        #want to select the most correlated ones with the "class" column
        results = cmatrix["class"].sort_values(ascending=False).head(11)
        if(printBool):
            printTops(i,results) #print the top results for the correlation
        results_att=results.keys() # we want the attribute not the correlation value
        results_att=results_att[1:] #remove class attribute
        print(results_att)
        for pixel in results_att:
            if pixel not in listG: #avoid duplicates
                listG.append(pixel)
                print(listG)
    return listG

def printTops(index,list):
    print("Top 10 most correlated pixels of sign " + str(index) + ":")
    print(list)

print(getNTops(10,True))
#print(results.keys())


#print(X[results.keys().values.tolist()])
