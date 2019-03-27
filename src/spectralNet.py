
import sys
sys.path.append('./src')
sys.path.append('./src/models')
sys.path.append('./src/data_loader')
sys.path.append('./src/optimizer')
sys.path.append('./src/helper')

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize			# version : 0.17
from MLP_autoencoder import *
from torch.autograd import Variable
from MLP_autoencoder import *
from DManager import *
from basic_optimizer import *

import sklearn.metrics
import numpy as np
import numpy.matlib
import torch

np.set_printoptions(precision=4)
np.set_printoptions(threshold=3000)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

class spectralNet():
	def __init__(self, X, k):
		self.X = X

		db = {}
		db['add_decoder'] = False
		db['learning_rate'] = 0.001
		db['net_input_size'] = X.shape[1]
		db['mlp_width'] = 45	#X.shape[1]
		db['net_depth'] = 3
		db['k'] = k
		db['cuda'] = False
		db['batch_size'] = 10

		if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
		else: db['dataType'] = torch.FloatTensor				


		db['Kx'] = self.STSC_σ(X)
		K = torch.tensor(db['Kx'])
		self.L = Variable(K.type(db['dataType']), requires_grad=False)


		self.mlp = MLP_autoencoder(db)
		self.mlp.set_Laplacian(self.L)

		self.db = db
		self.setup_data_loader(X)

	def setup_data_loader(self, X):
		db = self.db

		db['data'] = DManager(X, db['dataType'])
		db['data_loader'] = DataLoader(dataset=db['data'], batch_size=db['batch_size'], shuffle=True)


	def L_to_U(self, L, k):
		eigenValues,eigenVectors = np.linalg.eigh(L)
	
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_lambda = eigenValues[n1:n2]
		U_normalized = normalize(U, norm='l2', axis=1)
		
		return [U, U_normalized]


	def obtain_eigen_vectors(self):
		db = self.db
		m_sqrt = np.sqrt(db['data'].X.shape[0])		
		basic_optimizer(self.mlp, self.db, 'data_loader')
		Y = self.mlp.get_orthogonal_out(db['data'].X_Var)/m_sqrt	# Y^TY = I
		Y = Y.data.numpy()
		return Y

	def run(self):
		db = self.db

		#[U, U_normalized] = self.L_to_U(db['Kx'], db['k'])
		#allocation = KMeans(db['k'], n_init=20).fit_predict(U_normalized)
		

		Y = self.obtain_eigen_vectors()
		Y = normalize(Y, norm='l2', axis=1)
		allocation = KMeans(db['k'], n_init=20).fit_predict(Y)
		return allocation

	def STSC_σ(self, X):
		n = X.shape[0]
		if n < 50:
			num_of_samples = n
		else:
			num_of_samples = 50
		
	
		unique_X = np.unique(X, axis=0)
		neigh = NearestNeighbors(num_of_samples)
		neigh.fit(unique_X)
		
		[dis, idx] = neigh.kneighbors(X, num_of_samples, return_distance=True)
		dis_inv = 1/dis[:,1:]
		idx = idx[:,1:]
		
		total_dis = np.sum(dis_inv, axis=1)
		total_dis = np.reshape(total_dis,(n, 1))
		total_dis = np.matlib.repmat(total_dis, 1, num_of_samples-1)
		dis_ratios = dis_inv/total_dis
	
		result_store_dictionary = {}
		σ_list = np.zeros((n,1))
		
		for i in range(n):
			if str(X[i,:]) in result_store_dictionary:
				σ = result_store_dictionary[str(X[i,:])] 
				σ_list[i] = σ
				continue
	
			dr = dis_ratios[i,:]
	
			Δ = unique_X[idx[i,:],:] - X[i,:]
			Δ2 = Δ*Δ
			d = np.sum(Δ2,axis=1)
			σ = np.sqrt(np.sum(dr*d))
			σ_list[i] = σ#*10
	
			result_store_dictionary[str(X[i,:])] = σ
	
		σ2 = σ_list.dot(σ_list.T)
		D = sklearn.metrics.pairwise.pairwise_distances(X, metric='euclidean', n_jobs=1)
	
		K = np.exp(-(D*D)/(σ2))
		np.fill_diagonal(K,0)
	
		D_inv = 1.0/np.sqrt(np.sum(K, axis=1))
		Dv = np.outer(D_inv, D_inv)
		DKD = Dv*K
	
		return DKD


