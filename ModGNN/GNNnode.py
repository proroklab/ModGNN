import torch.nn as nn
import torch
import types

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Two options for implementing the GNN Node:
# • Extend GNN_Node and implement the finput, fcom, ... submodules. Remember to register parameters
# • GNN(finput=finput_gen, fcom=fcom_gn, ...), where finput_gen = (finput, finput_params)

class GNNnode(nn.Module):

	def __init__(self, K=1, **kwargs):
		super().__init__()
		for fname, fgen in kwargs.items():
			if fname in ['finput','fcom','fpre','fmid','ffinal']:
				self.add_submodule(fname, fgen)
		self.data = {k: [] for k in range(K+1)}

	def finput(self, X, *layer_outputs):
		# X: (batch * (K+1) * N) x D_obs
		# OUT: (batch * (K+1) * N) x D_input
		return X
	def fcom(self, A, X):
		# A: batch x N x N
		# X: batch x N x D_input
		# OUT: batch x N x N x D_input
		return (A[:,:,:,None] * X[:,None,:,:]) - (A[:,:,:,None] * X[:,:,None,:])
	def fpre(self, X):
		# X: (batch * N * N) x K+1 x D_input
		# OUT: (batch * N * N) x K+1 x D_pre
		return X
	def fmid(self, X):
		# X: (batch * N) x K+1 x D_pre
		# OUT: (batch * N) x K+1 x D_mid
		return X
	def ffinal(self, X):
		# X: (batch * N) x D_mid
		# OUT: (batch x N) x D_final
		return X

	def add_submodule(self, fname, fgen):
		# fgen is a tuple of (submodule, param_list)
		func, param_list = fgen
		setattr(self, fname, func)
		self.register_params(param_list, name=fname)
		
	def register_params(self, params, name=""):
		# params can be a module, list, dict, or generator
		if params is None:
			return
		if isinstance(params, types.GeneratorType):
			params = list(params)
		if isinstance(params, dict):
			params = list(params.values())
		if not isinstance(params, list):
			params = [params]
		for i, layer in enumerate(params):
			if isinstance(layer,torch.nn.parameter.Parameter):
				self.register_parameter("%s_param_%d" % (name, i), layer)
			else:
				self.add_module("%s_module_%d" % (name, i), layer)

	def forward(self, A, X):
		# A: batch x N x N (most recent adjacency matrix)
		# X: batch x N x N x K+1 x D (last K+1 time steps of the joint state)
		batch, N, _, K, _ = X.shape; K -= 1
		I = torch.eye(N, device=device).repeat(batch,1,1,1)
		Ak = torch.cat([I, A.unsqueeze(1).expand(-1,-1,-1,K)], dim=3).unsqueeze(4) # Ak: batch x N x N x K+1 x 1
		# fpre
		fpre_out = self.fpre(X.view(batch*N*N, K+1, -1)).view(batch, N, N, K+1, -1) # fpre_out: batch x N x N x K+1 x D_pre
		Z1 = (Ak * fpre_out).sum(dim=3) # Z1: batch x N x K+1 x Dpre
		# fmid
		fmid_out = self.fmid(Z1.view(batch*N, K+1, -1)).view(batch, N, K+1, -1) # fmid_out: batch x N x K+1 x D_mid
		Z2 = fmid_out.sum(dim=1) # Z2: batch x N x D_post
		# ffinal
		output = self.ffinal(Z2.view(batch*N, -1)).view(batch, N, -1) # output: batch x N x D_final
		return output


	def input(self, x):
		# x : batch x D or D
		if len(x.shape) == 1:
			x = x.unsqueeze(0)
		self.data[0] = [self.finput(x)]


	def receive(self, data):
		pass


	def send(self):
		pass


	def compute(self):
		pass






