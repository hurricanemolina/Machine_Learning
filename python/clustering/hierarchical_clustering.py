#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 18:35:07 2018

Maria J. Molina
Ph.D. Candidate
Central Michigan University

"""

#########################################################################################
#########################################################################################
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#########################################################################################
#########################################################################################
#########################################################################################


dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values


#########################################################################################
#########################################################################################
#########################################################################################


#Using the dendrogram to find optimal number of clusters

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#########################################################################################
#########################################################################################
#########################################################################################


#Fitting hierarchical clustering to the mall dataset.

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage='ward') 
#ward : minimize variance in each cluster

y_hc = hc.fit_predict(X)


#########################################################################################
#########################################################################################
#########################################################################################


#Visualising the clusters

color = ['r','b','g','c','m']
customer = ['Careful', 'Standard', 'Target', 'Careless', 'Sensible']

for i in range(5):
    plt.scatter(X[y_hc==i, 0], X[y_hc==i, 1], s=50, c=color[i], label=customer[i])

plt.title('Clusters of Clients')
plt.ylabel('Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.show()


#########################################################################################
#########################################################################################
#########################################################################################
