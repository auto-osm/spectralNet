

from sp_net import *
from basic_optimizer import *

class sn_eig_solver():
	def __init__(self, db):
		self.spn = sp_net(db)
		self.db = db

	def obtain_eigen_vectors(self):
		db = self.db
		self.spn.set_Laplacian(db['L'])
		m_sqrt = np.sqrt(db['data'].X.shape[0])		

		basic_optimizer(self.spn, db, 'data_loader')
		Y = self.spn.get_orthogonal_out(db['data'].X_Var)/m_sqrt	# Y^TY = I
		Y = Y.data.numpy()
		return Y

