import torch.nn as nn
import torch
import types

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Two options for implementing the GNN Node:
# • Extend GNN_Node and implement the finput, fcom, ... submodules. Remember to register parameters
# • GNN(finput=finput_gen, fcom=fcom_gen, ...), where finput_gen = (finput, finput_params)

class GNNnode(nn.Module):

	def __init__(self, K=1, **kwargs):
		super().__init__()
		for fname, fgen in kwargs.items():
			if fname in ['finput','fcom','fpre','fmid','ffinal']:
				self.add_submodule(fname, fgen)
		self.K = K
		self.Y = None
		self.y = None
		self.x = None


	def finput(self, X, *layer_outputs):
		# X: (batch * N * (K+1)) x D_obs
		# OUT: (batch * N * (K+1)) x D_input
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
		I = torch.eye(N, device=device).unsqueeze(0).unsqueeze(3).repeat(batch,1,1,1)
		Ak = torch.cat([I, A.unsqueeze(3).expand(-1,-1,-1,K)], dim=3).unsqueeze(4) # Ak: batch x N x N x K+1 x 1
		# fpre
		fpre_out = self.fpre(X.view(batch*N*N, K+1, -1)).view(batch, N, N, K+1, -1) # fpre_out: batch x N x N x K+1 x D_pre
		Z1 = (Ak * fpre_out).sum(dim=2) # Z1: batch x N x K+1 x Dpre
		# fmid
		fmid_out = self.fmid(Z1.view(batch*N, K+1, -1)).view(batch, N, K+1, -1) # fmid_out: batch x N x K+1 x D_mid
		Z2 = fmid_out.sum(dim=2) # Z2: batch x N x D_post
		# ffinal
		output = self.ffinal(Z2.view(batch*N, -1)).view(batch, N, -1) # output: batch x N x D_final
		return output


	def input(self, obs):
		# x : D_obs
		self.x = self.finput(obs)
		self.Dinput = self.x.shape[0]
		if self.Y is None:
			self.Y = torch.zeros(1, self.K+1, self.Dinput)
		self.Y[0,0,:] = self.x


	def receive(self, incoming):
		# incoming: list of data of shape (K x Dinput) from each neighbouring agent
		Ni = len(incoming)
		X = torch.stack([self.y] + incoming, dim=1) # X: K x Ni+1 x Dinput
		A0 = torch.zeros(Ni+1, Ni+1)
		A0[0,1:] = 1
		A = A0.repeat(self.K,1,1) # A: K x N+1 x N+1
		Xc = self.fcom(A, X)[:,0,:,:] # Xc: K x Ni x Dinput
		self.Y = torch.zeros(Ni+1, self.K+1, self.Dinput) # Y: Ni+1 x K+1 x Dinput
		self.Y[0,0,:] = self.x
		self.Y[1:,1:,:] = Xc[:,1:,:].permute(1,0,2)


	def send(self):
		# aggregates each neighbourhood of stored data from k=0 to K=K-1
		self.y = self.Y[:,:-1,:].sum(dim=0)
		return self.y


	def compute(self):
		X = self.Y
		Ni, _, _ = X.shape; Ni -= 1
		mask = torch.cat([torch.zeros(Ni+1, 1, 1), torch.ones(Ni+1, self.K, 1)], dim=1)
		mask[0,0,:] = 1
		mask[0,1:,:] = 0
		# fpre
		fpre_out = self.fpre(X) # fpre_out: Ni+1 x K+1 x D_pre
		Z1 = (mask * fpre_out).sum(dim=0) # Z1: K+1 x Dpre
		# fmid
		fmid_out = self.fmid(Z1) # fmid_out: K+1 x D_mid
		Z2 = fmid_out.sum(dim=0) # Z2: D_post
		# ffinal
		output = self.ffinal(Z2) # output: D_final
		return output






