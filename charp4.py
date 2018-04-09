# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:56:22 2018

@author: Lenny
"""

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

centers=[[-2,2],[2,2],[0,4]]
X,Y=make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.6)
plt.figure(figsize=(16,10),dpi=144)
c=np.array(centers)
plt.scatter(X[:,0],X[:,1],c=Y,s=100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s=100,marker='^',c='orange')
k=5
clf=KNeighborsClassifier(n_neighbors=k)
clf.fit(X,Y)
X_sample=np.array([[0,2],])
Y_sample=clf.predict(X_sample)
neighbors=clf.kneighbors(X_sample,return_distance=False)
plt.scatter(X_sample[:,0],X_sample[:,1],marker='x',c=Y_sample,s=100,cmap='cool')
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[:,0]],[X[i][1],X_sample[:,1]],'k--',linewidth=0.6)