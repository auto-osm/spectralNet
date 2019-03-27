#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import numpy as np

class DManager(Dataset):
	def __init__(self, X, dataType):
		self.dtype = np.float64				#np.float32
		self.array_format = 'numpy'			# numpy, pytorch

		self.X = X
		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		

		self.X_Var = torch.tensor(self.X)
		self.X_Var = Variable(self.X_Var.type(dataType), requires_grad=False)

		print('\t\tData of size %dx%d was loaded ....'%(self.N, self.d))

	def __getitem__(self, index):
		return self.X[index], index


	def __len__(self):
		try: return self.X.shape[0]
		except: return 0

