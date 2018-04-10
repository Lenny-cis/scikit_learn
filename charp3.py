# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 22:16:22 2018

@author: Lenny
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

n_dots=20
x=np.linspace(0,1,n_dots)
y=np.sqrt(x)+0.2*np.random.rand(n_dots)-0.1

def plot_polynomial_fit(x,y,order):
    p=np.poly1d(np.polyfit(x,y,order))
    t=np.linspace(0,1,200)
    plt.plot(x,y,'ro',t,p(t),'-',t,np.sqrt(t),'r--')
    return p
plt.figure(figsize=(18,4),dpi=200)
titles=['Under Fitting', 'Fitting', 'Over Fitting']
models=[None,None,None]
for index,order in enumerate([1,3,10]):
    plt.subplot(1,3,index+1)
    models[index]=plot_polynomial_fit(x,y,order)
    plt.title(titles[index],fontsize=20)

coeffs_1d=[0.2,0.6]
plt.figure(figsize=(9,6),dpi=200)
t=np.linspace(0,1,200)
plt.plot(x,y,'ro',t,models[0](t),'-',t,np.poly1d(coeffs_1d)(t),'r-')
plt.annotate(
        r'L1:$y={1}+{0}x$'.format(coeffs_1d[0],coeffs_1d[1])
        ,xy=(0.8,np.poly1d(coeffs_1d)(0.8)),xycoords='data'
        ,xytext=(-90,-50),textcoords='offset points',fontsize=16,
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.annotate(
        r'L2:$y={1}+{0}x$'.format(models[0].coeffs[0],models[0].coeffs[1])
        ,xy=(0.3,models[0](0.3)),xycoords='data'
        ,xytext=(-90,-50),textcoords='offset points',fontsize=16,
        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

n_dots=200
X=np.linspace(0,1,n_dots)
Y=np.sqrt(X)+0.2*np.random.rand(n_dots)-0.1
X=X.reshape(-1,1)
Y=Y.reshape(-1,1)
def polynomial_model(degree=1):
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression=LinearRegression()
    pipeline=Pipeline([('polynomial_features',polynomial_features),('linear_regression',linear_regression)])
    return pipeline
def plot_learning_curve(estimator,title,X,Y,ylim=None,cv=None,n_job=1,train_sizes=np.linspace(.1,1,5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('Score')
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,Y,cv=cv,n_jobs=n_job,train_sizes=train_sizes)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Train score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='Cross-validation score')
    plt.legend(loc='best')
    return plt
cv=ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
titles=['Learning Curves(Under Fitting)','Learning Curves','Learning Curves(Over Fitting)']
degrees=[1,3,10]
plt.figure(figsize=(18,4),dpi=200)
for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(polynomial_model(degrees[i]),titles[i],X,Y,ylim=(0.75,1.01),cv=cv)
    