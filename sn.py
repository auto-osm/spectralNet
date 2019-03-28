#!/usr/bin/env python

import sys
import numpy as np

sys.path.append('./src')
from spectralNet import *
from sklearn import preprocessing
from plot_clusters import *

#	Uncomment one of these datasets
#data_path = 'dataset/inner_rings.csv'; k = 3
#data_path = 'dataset/spiral_arm.csv'; k = 3
#data_path = 'dataset/smiley.csv'; k = 3
data_path = 'dataset/moon.csv'; k = 2
#data_path = 'dataset/four_lines.csv'; k = 4
#data_path = 'dataset/noisy_two_clusters.csv'; k = 3


X = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
X = preprocessing.scale(X)

sn = spectralNet(X, k)
allocation = sn.run(solver='specNet')		#specNet, or eig
cluster_plot(X, allocation)

