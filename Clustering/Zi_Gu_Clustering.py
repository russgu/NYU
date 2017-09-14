
# coding: utf-8

# In[22]:

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl
import random 


#1) transform the feature vectors
grain_text=open('dataCereal-grains-pasta.txt');
grain_features=grain_text.read().split("\n");

fat_text=open('dataFats-oils.txt');
fat_features=fat_text.read().split("\n");

shellfish_text=open('dataFinfish-shellfish.txt');
shellfish_features=shellfish_text.read().split("\n");

veggie_text=open('dataVegetables.txt');
veggie_features=veggie_text.read().split("\n");


def edit_text(features):
    labels=[];
    for i in range(len(features)):
        features[i]=features[i][:-1];
        features[i]=features[i].split("^");
        labels.append(features[i][0]);
        del features[i][0];
        for j in range(len(features[i])):
            features[i][j]=float(features[i][j]);
    return labels;


grain_labels=edit_text(grain_features);
fat_labels=edit_text(fat_features);
fish_labels=edit_text(shellfish_features);
veggie_labels=edit_text(veggie_features);

all_features=[];

def transform(feature_vector):
    for j in range(150):
        maximum=-10**10;
        minimum=10**10;
        for i in range(len(feature_vector)):
            if feature_vector[i][j]>maximum:
                maximum=feature_vector[i][j];
            if feature_vector[i][j]<minimum:
                minimum=feature_vector[i][j];
        for i in range(len(feature_vector)):
            feature_vector[i][j]=(feature_vector[i][j]-minimum)/(maximum-minimum);  
                    
transform(grain_features);
transform(fat_features);
transform(shellfish_features);
transform(veggie_features);

for g in grain_features:
    all_features.append(g);
for s in shellfish_features:
    all_features.append(s);
for f in fat_features:
    all_features.append(f);
for v in veggie_features:
    all_features.append(v);
   


#2)Apply K means Clustering with K=4
def K_Means(features):
    X=np.array(features);
    kmeans=KMeans(n_clusters=4);
    kmeans.fit(X);
    label=kmeans.labels_;
    return label



#3)RandIndex
def randIndex(truth, predicted):
    """
    The function is to measure similarity between two label assignments
    truth: ground truth labels for the dataset (1 x 1496)
    predicted: predicted labels (1 x 1496)
    """
    if len(truth) != len(predicted):
        print "different sizes of the label assignments";
        return -1;
    elif (len(truth) == 1):
        return 1;
    sizeLabel = len(truth);
    agree_same = 0;
    disagree_same = 0;
    count = 0;
    for i in range(sizeLabel-1):
        for j in range(i+1,sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1;
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same +=1;
            count += 1;
    return (agree_same+disagree_same)/float(count)



truth=[];
for i in grain_features:
    truth.append(1);
for i in fat_features:
    truth.append(0);
for i in shellfish_features:
    truth.append(2);
for i in veggie_features:
    truth.append(3);
    
    
truth_permutated=random.sample(truth, len(truth));
predicted=K_Means(all_features); 
base_randindex=randIndex(truth,truth_permutated);
randindex=randIndex(truth,predicted);
print ("The baseline is:", base_randindex);
print ("Randindex for kmeans predicted is: ", randindex);

#4)Run KMeans 20 times
def Initialize_Centroids(features):
    for i in range(20):
        X=np.array(features);
        kmeans=KMeans(n_init=1,init='random');
        kmeans.fit(X);
        J=kmeans.inertia_;
        randindex=randIndex(truth,kmeans.labels_);
Initialize_Centroids(all_features);    
   
    
#5)dendograms

grain=random.sample(grain_features,30);
veggie=random.sample(veggie_features,30);
fat=random.sample(fat_features,30);
shellfish=random.sample(shellfish_features,30);


list=[];
for g in grain:
    list.append(g);
for f in fat:
    list.append(f);
for s in shellfish:
    list.append(s);
for v in veggie:
    list.append(v);
    
fig = pl.figure();
data = np.array(list);
datalable = (['grain'] * 30) + (['fat'] * 30) + (['shellfish'] * 30) + (['veggie'] * 30);
hClsMat = sch.linkage(data, method='complete'); # Complete clustering
sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45);
fig.show()
resultingClusters = sch.fcluster(hClsMat,t = 3, criterion = 'distance');
print resultingClusters


#6)Cut dendogram
fig2 = pl.figure();
data_all = np.array(all_features);
datalabel= (['grains'] * len(grain_features)) + (['fat'] * len(fat_features)) + (['shellfish'] * len(shellfish_features)) + (['veggie'] * len(veggie_features));
hClsMat = sch.linkage(data_all, method='complete'); # Complete clustering
sch.dendrogram(hClsMat, labels= datalabel, leaf_rotation = 45);
fig2.show()
resultingClusters = sch.fcluster(hClsMat,t=3.8,criterion='distance');
print resultingClusters


#7)sub-clusters

X=np.array(grain_features);
kmeans_5=KMeans(n_clusters=5);
kmeans_10=KMeans(n_clusters=10);
kmeans_25=KMeans(n_clusters=25);
kmeans_50=KMeans(n_clusters=50);
kmeans_75=KMeans(n_clusters=75);
kmeans_5.fit(X);
kmeans_10.fit(X);
kmeans_25.fit(X);
kmeans_50.fit(X);
kmeans_75.fit(X);

dic_5={};

for label in kmeans_5.labels_:
    if label in dic_5:
        dic_5[label]+=1;
    else:
        dic_5[label]=1;

print "K=5";
print dic_5;
for key in dic_5:
    print ("label is: ",key);
    for i in range(len(kmeans_5.labels_)):
        if key==kmeans_5.labels_[i]:
            print grain_labels[i];
    print ("---------------");

    
dic_10={};
for label in kmeans_10.labels_:
    if label in dic_10:
        dic_10[label]+=1;
    else:
        dic_10[label]=1;

print "K=10";
print dic_10;
for key in dic_10:
    print ("label is: ",key);
    for i in range(len(kmeans_10.labels_)):
        if key==kmeans_10.labels_[i]:
            print grain_labels[i];
    print ("---------------");
    

dic_25={};
for label in kmeans_25.labels_:
    if label in dic_25:
        dic_25[label]+=1;
    else:
        dic_25[label]=1;

print "K=25";
print dic_25;
for key in dic_25:
    print ("label is: ",key);
    for i in range(len(kmeans_25.labels_)):
        if key==kmeans_25.labels_[i]:
            print grain_labels[i];
    print ("---------------");


dic_50={};
for label in kmeans_50.labels_:
    if label in dic_50:
        dic_50[label]+=1;
    else:
        dic_50[label]=1;

print "K=50";
print dic_50;
for key in dic_50:
    print ("label is: ",key);
    for i in range(len(kmeans_50.labels_)):
        if key==kmeans_50.labels_[i]:
            print grain_labels[i];
    print ("---------------");
        


dic_75={};
for label in kmeans_75.labels_:
    if label in dic_75:
        dic_75[label]+=1;
    else:
        dic_75[label]=1;

print "K=75";
print dic_75;
for key in dic_75:
    print ("label is: ",key);
    for i in range(len(kmeans_75.labels_)):
        if key==kmeans_75.labels_[i]:
            print grain_labels[i];
    print ("---------------");





