# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:56:22 2018

@author: Lenny
"""

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split,ShuffleSplit,learning_curve,KFold,cross_val_score
import pandas as pd
from sklearn.feature_selection import SelectKBest

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
    
data=pd.read_csv(r'code/datasets/pima-indians-diabetes/diabetes.csv')
data.head()
data.shape
data.groupby(['Outcome']).size()
X=data.iloc[:,0:8]
Y=data.iloc[:,8]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
models=[]
models.append(('KNN',KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN with weights',KNeighborsClassifier(n_neighbors=2,weights='distance')))
models.append(('Radius Neighbors',RadiusNeighborsClassifier(n_neighbors=2,radius=500)))
results=[]
for name,model in models:
    model.fit(X_train,Y_train)
    results.append((name,model.score(X_test,Y_test)))
for i in range(len(results)):
    print('name:{};score:{}'.format(results[i][0],results[i][1]))
    
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)
train_score=knn.score(X_train,Y_train)
test_score=knn.score(X_test,Y_test)
print('train score:{};test score:{}'.format(train_score,test_score))

knn=KNeighborsClassifier(n_neighbors=2)
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
plt.figure(figsize=(10,6),dpi=200)
plot_learning_curve(knn,'Learn Curve for KNN Diabetes',X,Y,ylim=(0.0,1.01),cv=cv)

selector=SelectKBest(k=2)
X_new=selector.fit_transform(X,Y)
X_new[0:5]
results=[]
for name,model in models:
    kfold=KFold(n_splits=10)
    cv_result=cross_val_score(model,X_new,Y,cv=kfold)
    results.append((name,cv_result))
for i in range(len(results)):
    print('name:{};cross val score:{}'.format(results[i][0],results[i][1].mean()))
plt.figure(figsize=(10,6),dpi=200)
plt.ylabel('BMI')
plt.xlabel('Glucose')
plt.scatter(X_new[Y==0][:,0],X_new[Y==0][:,1],c='r',s=20,marker='o')
plt.scatter(X_new[Y==1][:,0],X_new[Y==1][:,1],c='g',s=20,marker='^')
