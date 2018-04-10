# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:06:21 2018

@author: Lenny
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,ShuffleSplit
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import time
import numpy as np

def polynomial_model(degree=1,**kwarg):
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    logistic_regression=LogisticRegression(**kwarg)
    pipeline=Pipeline([('polynomial_features',polynomial_features),('logistic_regression',logistic_regression)])
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

cancer=load_breast_cancer()
X=cancer.data
Y=cancer.target
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=LogisticRegression()
model.fit(X_train,Y_train)
train_score=model.score(X_train,Y_train)
test_score=model.score(X_test,Y_test)
print('train score:{train_score:.6f};test score:{test_score:.6f}'.format(train_score=train_score,test_score=test_score))
Y_pred=model.predict(X_test)
print('matchs:{0}/{1}'.format(np.equal(Y_pred,Y_test).shape[0],Y_test.shape[0]))
Y_pred_proba=model.predict_proba(X_test)
print('sample of predict probability:{0}'.format(Y_pred_proba[0]))
Y_pred_proba_0=Y_pred_proba[:,0]>0.1
result=Y_pred_proba[Y_pred_proba_0]
Y_pred_proba_1=result[:,1]>0.1
print(result[Y_pred_proba_1])

model=polynomial_model(degree=2,penalty='l1')
start=time.clock()
model.fit(X_train,Y_train)
train_score=model.score(X_train,Y_train)
cv_score=model.score(X_test,Y_test)
print('elaspe:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}'.format(time.clock()-start,train_score,cv_score))

logistic_regression=model.named_steps['logistic_regression']
print('model parameters shape:{0};count of non-zero element:{1}'.format(logistic_regression.coef_.shape,np.count_nonzero(logistic_regression.coef_)))

cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
title='Learning Curve (degree={0},penalty={1})'
degrees=[1,2]
penalty='l1'
start=time.clock()
plt.figure(figsize=(12,4),dpi=144)
for i in range(len(degrees)):
    plt.subplot(1,len(degrees),i+1)
    plot_learning_curve(plt
                        ,polynomial_model(degree=degrees[i],penalty=penalty)
                        ,title.format(degrees[i],penalty)
                        ,X
                        ,Y
                        ,ylim=(0.8,1.01)
                        ,cv=cv
                        )
print('elaspe:{0:.6f}'.format(time.clock()-start))

penalty='l2'
start=time.clock()
plt.figure(figsize=(12,4),dpi=144)
for i in range(len(degrees)):
    plt.subplot(1,len(degrees),i+1)
    plot_learning_curve(plt
                        ,polynomial_model(degree=degrees[i],penalty=penalty)
                        ,title.format(degrees[i],penalty)
                        ,X
                        ,Y
                        ,ylim=(0.8,1.01)
                        ,cv=cv
                        )
print('elaspe:{0:.6f}'.format(time.clock()-start))