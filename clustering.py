#K means Clustering
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#Read the datsets
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

data = pd.read_csv('x_train_gr_smpl.csv')
labels = pd.read_csv('y_train_smpl_1.csv')

#Shuffle the data
shuffle(data, random_state = 30)
shuffle(labels, random_state = 30)

data.insert(2304, "y", labels, allow_duplicates = False)

kmeans = KMeans(n_clusters=10).fit(data)
centroids = kmeans.cluster_centers_
print(centroids)

print('kmeans: {}'.format(silhouette_score(data, kmeans.labels_,metric='euclidean')))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data.values)
data['pca-one'] = pca_result[:,0]
data['pca-two'] = pca_result[:,1]
data['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

rndperm = np.random.permutation(data.shape[0])

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=data.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
