#!/usr/bin/env python

import sys
sys.path.append('./src')
from spectralNet import *
import numpy as np
from sklearn import preprocessing


data_path = 'dataset/inner_rings.csv'
X = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
X = preprocessing.scale(X)

sn = spectralNet(X)





