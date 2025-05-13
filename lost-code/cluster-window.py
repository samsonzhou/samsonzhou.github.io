import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt

#https://realpython.com/k-means-clustering-python/
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

#DATASET EXTRACTION
file = open("Skin_NonSkin.txt", "r")

dataset_str=[]
for line in file.readlines():
    line_str=line.split("\t")
    thisLine = []
    for term in line_str:
        thisLine.append(int(term))
    dataset_str.append(thisLine)
dataset = np.array(dataset_str)
dataset_normalized = preprocessing.scale(dataset)

###CLUSTERING ALGORITHM###
#number of repetitions
n_reps=50
#n_reps=1
#k
#n_clusters=3
#number of independent kmeans++ instances
n_init=5
#n_init=1
#number of kmeans++ iterations
max_iter=10
#max_iter=1
#random seed
random_state=42
#dimension
#n_features=4
n_features=2
#coreset sizes
#exp_num=6
exp_num=9
k_list=[2,3,4,5,6,7,8,9,10]
#coreset_list=[5,10,15,20,25,30]
#coreset_list=[3,4,5,6,7,8,9,10,11,12]
#coreset_list=[3]

####POINT GENERATION###
#num of points in each cluster
#n_samples=200 for overall 200 points, uniform
#n_samples = [100,100]
n_samples = [100000,100000]
#list of centers
#centers =[[-10,10,0,0],[10,-10,0,0]]
centers =[[-10,10],[10,-10]]
#std of each cluster
cluster_std=1

uni_means=[]
imp_means=[]
hist_means=[]
true_means=[]
uni_max=[]
imp_max=[]
hist_max=[]
true_max=[]
uni_min=[]
imp_min=[]
hist_min=[]
true_min=[]

flag = True

for exp in range(exp_num):

    uni_scores=[]
    imp_scores=[]
    hist_scores=[]
    true_scores=[]
    #coreset_size=coreset_list[exp]
    n_clusters=k_list[exp]
    coreset_size=25
    for t_rep in range(n_reps):

        #features, true_labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=1)
        #nextpoint = (make_blobs(n_samples=1,n_features=n_features,centers=[[500,500,0,0]],cluster_std=1))[0]
        features, true_labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=2.75)
        nextpoint = (make_blobs(n_samples=1,n_features=2,centers=[[100000,100000]],cluster_std=2.75))[0]
        features = np.vstack((features, nextpoint))
        #features = np.vstack((dataset_normalized, features))

        #true_labels = np.hstack((true_labels,[2]))

        uni_features = []
        uni_positions = random.sample(range(0, len(features)), coreset_size)
        for i in uni_positions:
            uni_features.append(features[i])
        #last element is far
        imp_features = []
        imp_features.append(features[len(features)-1])
        imp_positions = random.sample(range(0, len(features)-1), coreset_size-1)
        for i in imp_positions:
            imp_features.append(features[i])
        #histogram
        hist_features = []
        #extrapoints = (make_blobs(n_samples=2,n_features=2,centers=[[-10,10,0,0],[-10,-10,0,0]],cluster_std=1,random_state=42))[0]
        extrapoints = (make_blobs(n_samples=2,n_features=2,centers=[[-100000,100000],[-100000,-100000]],cluster_std=2.75,random_state=42))[0]
        hist_features.append(features[len(features)-1])
        hist_features.append(extrapoints[0])
        hist_features.append(extrapoints[1])
        hist_positions = random.sample(range(0, len(features)-1), coreset_size-3)
        for i in hist_positions:
            hist_features.append(features[i])

        #init="k-means++" for kmeans++ and "random" otherwise
        kmeans = KMeans(init="k-means++",n_clusters=n_clusters,n_init=n_init,max_iter=max_iter,random_state=42)
        kmeans.fit(uni_features)
        uni_centers=kmeans.cluster_centers_
        uni_score=abs(kmeans.score(features,uni_centers))
        kmeans.fit(hist_features)
        hist_centers=kmeans.cluster_centers_
        hist_score=abs(kmeans.score(features,hist_centers))
        kmeans.fit(imp_features)
        imp_centers=kmeans.cluster_centers_
        imp_score=abs(kmeans.score(features,imp_centers))
        kmeans.fit(features)
        true_centers=kmeans.cluster_centers_
        true_score=abs(kmeans.score(features,true_centers))

        flag = flag and ((len(features)-1) not in uni_positions)
        uni_scores.append(uni_score)
        imp_scores.append(imp_score)
        hist_scores.append(hist_score)
        true_scores.append(true_score)

    uni_means.append(np.mean(uni_scores))
    imp_means.append(np.mean(imp_scores))
    hist_means.append(np.mean(hist_scores))
    true_means.append(np.mean(true_scores))
    uni_max.append(np.max(uni_scores))
    imp_max.append(np.max(imp_scores))
    hist_max.append(np.max(hist_scores))
    true_max.append(np.max(true_scores))
    uni_min.append(np.min(uni_scores))
    imp_min.append(np.min(imp_scores))
    hist_min.append(np.min(hist_scores))
    true_min.append(np.min(true_scores))

##FLAG = DID UNIFORM SAMPLING FIND OUTLIER?
print(flag)

##PRINT STATISTICS FOR EACH ALGORITHM
##print(uni_min)
##print(uni_means)
##print(uni_max)
##print(hist_min)
##print(hist_means)
##print(hist_max)
##print(imp_min)
##print(imp_means)
##print(imp_max)
##print(true_min)
##print(true_means)
##print(true_max)

#print(min_scores,imp_scores,true_scores)
#print(kmeans.inertia_)
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)

##LINEPLOT OF ERRORS
##plt.fill_between(coreset_list, uni_min, uni_max, color='blue', alpha=0.1, label='uniform range')
##plt.plot(coreset_list, uni_means, label="avg. uni. errs.")
##plt.fill_between(coreset_list, hist_min, hist_max, color='red', alpha=0.1, label='histogram range')
##plt.plot(coreset_list, hist_means, label="avg. hist. errs")
##plt.fill_between(coreset_list, imp_min, imp_max, color='green', alpha=0.1, label='importance range')
##plt.plot(coreset_list, imp_means, label="avg. imp. errs")
##plt.fill_between(coreset_list, true_min, true_max, color='orange', alpha=0.1, label='ofline range')
##plt.plot(coreset_list, true_means, label="avg. offl. errs")
#plt.fill_between(k_list, uni_min, uni_max, color='blue', alpha=0.1, label='uniform range')
plt.plot(k_list, uni_means, label="avg. uni. errs.")
#plt.fill_between(k_list, hist_min, hist_max, color='red', alpha=0.1, label='histogram range')
plt.plot(k_list, hist_means, label="avg. hist. errs")
#plt.fill_between(k_list, imp_min, imp_max, color='green', alpha=0.1, label='importance range')
plt.plot(k_list, imp_means, label="avg. imp. errs")
#plt.fill_between(k_list, true_min, true_max, color='orange', alpha=0.1, label='ofline range')
plt.plot(k_list, true_means, label="avg. offl. errs")
# Display the plot
plt.yscale('log',base=2)
#plt.legend(loc="center right")
plt.legend(loc="upper right")
plt.ylabel('Clustering cost')
#plt.xlabel('Number of samples')
plt.xlabel('Number of clusters')
plt.show()

####SCATTERPLOT OF DATASET
##plt.scatter(features[:,0], features[:,1])
##plt.xscale('log',base=200)
##plt.yscale('log',base=200)
##plt.show()
