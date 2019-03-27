#!/usr/bin/env python

import sys
import numpy as np

sys.path.append('./src')
from spectralNet import *
from sklearn import preprocessing
from plot_clusters import *


data_path = 'dataset/inner_rings.csv'; k = 3
X = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
X = preprocessing.scale(X)

sn = spectralNet(X, k)
allocation = sn.run()
cluster_plot(X, allocation)

