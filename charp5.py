# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:23:52 2018

@author: Lenny
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib.figure import SubplotParams
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,ShuffleSplit
import time

def polynomial_model(degree=1):
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression()
    pipeline=Pipeline([('polynomial_features',polynomial_features),('linear_regression',linear_regression)])
    return pipeline
def plot_learning_curve(plt, estimator, title, X, y, ylim=None, cv=None,
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
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

n_dots=200
X=np.linspace(-2*np.pi,2*np.pi,n_dots)
Y=np.sin(X)+0.2*np.random.rand(n_dots)-0.1
X=X.reshape(-1,1)
Y=Y.reshape(-1,1)
degrees=[2,3,5,10]
results=[]
for d in degrees:
    model=polynomial_model(degree=d)
    model.fit(X,Y)
    train_score=model.score(X,Y)
    mse=mean_squared_error(Y,model.predict(X))
    results.append({'model':model,'degree':d,'score':train_score,'mse':mse})
for r in results:
    print('degree:{};train score:{};mean squared error:{}'.format(r['degree'],r['score'],r['mse']))

plt.figure(figsize=(12,6),dpi=200,subplotpars=SubplotParams(hspace=0.3))
for i,r in enumerate(results):
    fig=plt.subplot(2,2,i+1)
    plt.xlim(-8,8)
    plt.title('LinearRegression degree={}'.format(r['degree']))
    plt.scatter(X,Y,s=5,c='b',alpha=0.5)
    plt.plot(X,r['model'].predict(X),'r-')

boston=load_boston()
X=boston.data
Y=boston.target
X.shape
Y.shape
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
model=LinearRegression(normalize=True)
start=time.clock()
model.fit(X_train,Y_train)
train_score=model.score(X_train,Y_train)
cv_score=model.score(X_test,Y_test)
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))

model=polynomial_model(degree=2)
start=time.clock()
model.fit(X_train,Y_train)
train_score=model.score(X_train,Y_train)
cv_score=model.score(X_test,Y_test)
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))

cv=ShuffleSplit(n_splits=10,test_size=.2,random_state=0)
title='Learning Curves (degree={0})'
degrees=[1,2,3]
start=time.clock()
plt.figure(figsize=(18,4),dpi=200)
for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(plt,polynomial_model(degrees[i]),title.format(degrees[i]),X,Y,ylim=(0.01,1.01),cv=cv)
print('elaspe:{0:.6f}'.format(time.clock()-start))
    