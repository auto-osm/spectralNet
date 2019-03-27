#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize			# version : 0.17
import matplotlib.pyplot as plt




def cluster_plot(X, allocation):
	cmap = ['b', 'g', 'r', 'c', 'm', 'y','k']

	labels = np.unique(allocation)
	n = labels.shape[0]

	#plt.subplot(121)
	for i, j in enumerate(labels):
		subX = X[allocation == labels[i]]
		plt.plot(subX[:,0], subX[:,1], cmap[i] + '.')


	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Clustering Results')
	plt.show()

if __name__== "__main__":
	n = 100
	x1 = np.random.randn(n,2) + np.array([4,4])
	x2 = np.random.randn(n,2) + np.array([-4,-4])
	X = np.vstack((x1,x2))
	
	Y = np.concatenate([np.zeros(n), np.ones(n)])
	
	cluster_plot(X, Y)
