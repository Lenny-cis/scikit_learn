# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:00:39 2018

@author: Lenny
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn import datasets

x=np.linspace(-np.pi,np.pi,200)
C,S=np.cos(x),np.sin(x)
plt.plot(x,C,color='blue',linewidth=2.0,linestyle='-')
plt.plot(x,S,color='red',linewidth=2.0,linestyle='-')
plt.xlim(x.min()*1.1,x.max()*1.1)
plt.ylim(C.min()*1.1,C.max()*1.1)
plt.xticks((-np.pi,-np.pi/2,np.pi/2,np.pi),(r'$-\pi$',r'$-\pi/2$',r'$+\pi/2$',r'$+\pi$'))
plt.yticks([-1,-0.5,0,0.5,1])
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.legend(loc='upper left')
t=2*np.pi/3
plt.plot([t,t],[0,np.cos(t)],color='blue',linewidth=1.5,linestyle='--')
plt.scatter([t,],[np.cos(t),],50,color='blue')
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$'
             ,xy=(t,np.cos(t)),xycoords='data'
             ,xytext=(-90,-50),textcoords='offset points'
             ,fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white',edgecolor='None',alpha=0.65))
plt.show()

X=np.linspace(-np.pi,np.pi,200,endpoint=True)
C,S=np.cos(X),np.sin(X)
plt.figure(num='sin',figsize=(16,4))
plt.plot(X,S)
plt.figure(num='cos',figsize=(16,4))
plt.plot(X,C)
plt.figure(num='sin')
plt.plot(X,C)
print(plt.figure(num='sin').number)
print(plt.figure(num='cos').number)

plt.figure(figsize=(18,4))
plt.subplot(2,2,1)
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'subplot(2,2,1)',ha='center',va='center',size=20,alpha=0.5)
plt.subplot(2,2,2)
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'subplot(2,2,2)',ha='center',va='center',size=20,alpha=0.5)
plt.subplot(2,2,3)
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'subplot(2,2,3)',ha='center',va='center',size=20,alpha=0.5)
plt.subplot(2,2,4)
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'subplot(2,2,4)',ha='center',va='center',size=20,alpha=0.5)

plt.figure(figsize=(18,4))
G=gridspec.GridSpec(3,3)
axes_1=plt.subplot(G[0,:])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'Axes 1',ha='center',va='center',size=24,alpha=0.5)
axes_2=plt.subplot(G[1:,0])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'Axes 2',ha='center',va='center',size=24,alpha=0.5)
axes_3=plt.subplot(G[1:,-1])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'Axes 3',ha='center',va='center',size=24,alpha=0.5)
axes_4=plt.subplot(G[1,-2])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'Axes 4',ha='center',va='center',size=24,alpha=0.5)
axes_5=plt.subplot(G[-1,-2])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'Axes 5',ha='center',va='center',size=24,alpha=0.5)
plt.tight_layout()

plt.figure(figsize=(18,4))
plt.axes([0.1,0.1,0.8,0.8])
plt.xticks(())
plt.yticks(())
plt.text(0.2,0.5,'axes([0.1,0.1,0.8,0.8])',ha='center',va='center',size=20,alpha=0.5)
plt.axes([0.5,0.5,0.3,0.3])
plt.xticks(())
plt.yticks(())
plt.text(0.5,0.5,'axes([0.5,0.5,0.3,0.3])',ha='center',va='center',size=20,alpha=0.5)

def tickline():
    plt.xlim(0,10),plt.ylim(-1,1),plt.yticks([])
    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    for lable in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(16)
    ax.plot(np.arange(11),np.zeros(11))
    return ax
locators=[
        'plt.NullLocator()'
        ,'plt.MultipleLocator(base=1.0)'
        ,'plt.FixedLocator(locs=[0,2,8,9,10])'
        ,'plt.IndexLocator(base=3,offset=-1)'
        ,'plt.LinearLocator(numticks=5)'
        ,'plt.LogLocator(base=2,subs=[1.0])'
        ,'plt.MaxNLocator(nbins=3,steps=[1,3,5,7,9,10])'
        ,'plt.AutoLocator()'
        ]
n_locators=len(locators)
size=1024,60*n_locators
dpi=72.0
figsize=size[0]/float(dpi),size[1]/float(dpi)
fig=plt.figure(figsize=figsize,dpi=dpi)
fig.patch.set_alpha(0)
for i,locator in enumerate(locators):
    plt.subplot(n_locators,1,i+1)
    ax=tickline()
    ax.xaxis.set_major_locator(eval(locator))
    plt.text(5,0.3,locator[3:],ha='center',va='center',size=16)
plt.subplots_adjust(bottom=0.1,top=0.99,left=0.1,right=0.99)

n=1024
X=np.random.normal(0,1,n)
Y=np.random.normal(0,1,n)
T=np.arctan2(Y,X)
plt.subplot(1,2,1)
plt.scatter(X,Y,s=75,c=T,alpha=0.5)
plt.xlim(-1.5,1.5)
plt.xticks(())
plt.ylim(-1.5,1.5)
plt.yticks(())

n=256
X=np.linspace(-np.pi,np.pi,n,endpoint=True)
Y=np.sin(2*X)
plt.subplot(1,2,2)
plt.plot(X,Y+1,color='blue',alpha=1.00)
plt.fill_between(X,1,Y+1,color='blue',alpha=0.25)
plt.plot(X,Y-1,color='blue',alpha=1.00)
plt.fill_between(X,-1,Y-1,(Y-1)>-1,color='blue',alpha=0.25)
plt.fill_between(X,-1,Y-1,(Y-1)<-1,color='red',alpha=0.25)
plt.xlim(-np.pi,np.pi)
plt.xticks(())
plt.ylim(-2.5,2.5)
plt.yticks(())

n=12
X=np.arange(n)
Y1=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2=(1-X/float(n))*np.random.uniform(0.5,1.0,n)
plt.subplot(1,2,1)
plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
for x,y in zip(X,Y1):
    plt.text(x+0.4,y+0.05,'%.2f'%y,ha='center',va='bottom')
for x,y in zip(X,Y2):
    plt.text(x+0.4,-y-0.05,'%.2f'%y,ha='center',va='top')
plt.xlim(-.5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())

def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n=256
x=np.linspace(-3,3,n)
y=np.linspace(-3,3,n)
X,Y=np.meshgrid(x,y)
plt.subplot(1,2,2)
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)
C=plt.contour(X,Y,f(X,Y),8,color='black',linewidth=0.5)
plt.clabel(C,inline=1,fontsize=10)
plt.xticks(())
plt.yticks(())

plt.subplot(1,2,1)
n=10
x=np.linspace(-3,3,4*n)
y=np.linspace(-3,3,3*n)
X,Y=np.meshgrid(x,y)
plt.imshow(f(X,Y),cmap='hot',origin='low')
plt.colorbar(shrink=0.83)
plt.xticks(())
plt.yticks(())

plt.subplot(1,2,2)
n=20
Z=np.ones(n)
Z[-1]*=2
plt.pie(Z,explode=Z*0.05,colors=['%f'%(i/float(n)) for i in range(n)])
plt.axis('equal')
plt.xticks(())
plt.yticks(())

ax=plt.subplot(1,2,1)
ax.set_xlim(0,4)
ax.set_ylim(0,3)
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.grid(which='major',axis='x',linewidth=0.75,linestyle='-',color='0.75')
ax.grid(which='minor',axis='x',linewidth=0.25,linestyle='-',color='0.75')
ax.grid(which='major',axis='y',linewidth=0.75,linestyle='-',color='0.75')
ax.grid(which='minor',axis='y',linewidth=0.25,linestyle='-',color='0.75')
ax.set_xticklabels([])
ax.set_yticklabels([])

ax=plt.subplot(1,2,2,polar=True)
N=20
theta=np.arange(0.0,2*np.pi,2*np.pi/N)
radii=10*np.random.rand(N)
width=np.pi/4*np.random.rand(N)
bars=plt.bar(theta,radii,width=width,bottom=0.0)
for r,bar in zip(radii,bars):
    bar.set_facecolor(plt.cm.jet(r/10.))
    bar.set_alpha(0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])

digits=datasets.load_diabetes()

images_and_labels=list(zip(digits.images,digits.target))