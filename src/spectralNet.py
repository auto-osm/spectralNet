
from sklearn.cluster import KMeans
import sklearn.metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy.matlib


class spectralNet():
	def __init__(self, X):
		self.X = X
		self.K = self.STSC_σ(X)


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


