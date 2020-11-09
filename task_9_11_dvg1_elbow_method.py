#K means Clustering
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Read the datsets
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

data = pd.read_csv('x_train_gr_smpl.csv') #read sample data
labels = pd.read_csv('y_train_smpl.csv') #read labels data

#Shuffle the data
shuffle(data, random_state = 30)  #shuffle the data
shuffle(labels, random_state = 30)

#data.insert(2304, "y", labels, allow_duplicates = False) #inserting the labels values

data = sklearn.preprocessing.normalize(data, norm='l1', axis=1, copy=True, return_norm=True)[
  0]  # normalize the attributes values, [0] as it creates a list with the array inside

#mms = MinMaxScaler() # min max scaler algorithm initialization, similar to the above normalization but improves the performance
#mms.fit(data) #fit the data into the min max scaler
#data_transformed = mms.transform(data) # prepare the data for inputing it to the algorithm
#for k in K:
#    km = KMeans(n_clusters=k) #kmeans for the iterative number of clusters
#    km = km.fit(data_transformed) #fit the data to the algorithm
#    #print('kmeans: {}'.format(silhouette_score(data, km.labels_, metric='euclidean'))) #print the resulting value of the score
#    print(km.inertia_)
#    Sum_of_squared_distances.append(km.inertia_) # joining this distance to the global one

model = KMeans() #another method use for the other computation of the optimal number of clusters
visualizer = KElbowVisualizer( #elbow method visualization
    model, k=(2,15), metric='calinski_harabasz', timings=False #caliski metric elbow method computation from 2 to 15 clusters, as 1 was not possible
    #model, k=(1,12), metric='distortion', timings=False #distortion metric elbow method computation
)

visualizer.fit(data)        # Fit the data to the visualizer
visualizer.show()      #show the graph
'''
pca = PCA(n_components=2) #number of dimentions 
pca_result = pca.fit_transform(data.values)
data['pca-one'] = pca_result[:,0] 
data['pca-two'] = pca_result[:,1]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

rndperm = np.random.permutation(data.shape[0])

plt.figure(figsize=(16,10)) #graph configurations 
sns.scatterplot( #plot configurations
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=data.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
plt.show() #show graph
'''