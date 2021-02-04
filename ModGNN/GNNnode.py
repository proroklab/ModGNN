import torch.nn as nn
import torch
import types

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Two options for implementing the GNN Node:
# • Extend GNN_Node and implement the finput, fcom, ... submodules. Remember to register parameters
# • GNN(finput=finput_gen, fcom=fcom_gn, ...), where finput_gen = (finput, finput_params)

class GNNnode(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		for fname, fgen in kwargs.items(): # example: GNN_Node(finput=finput_gen)
			if fname in ['finput','fcom','fpre','fmid','ffinal']:
				self.add_submodule(fname, fgen)

	def finput(self, X, *layer_outputs):
		# X: batch x K+1 x N x D_obs
		# OUT: batch x K+1 x N x D_input
		return X
	def fcom(self, A, X):
		# A: batch x N x N
		# X: batch x N x D_input
		# OUT: batch x N x N x D_input
		return (A[:,:,:,None] * X[:,None,:,:]) - (A[:,:,:,None] * X[:,:,None,:])
	def fpre(self, X):
		# X: batch x K+1 x N x N x D_input
		# OUT: batch x K+1 x N x N x D_pre
		return X
	def fmid(self, X):
		# X: batch x K+1 x N x D_pre
		# OUT: batch x K+1 x N x D_mid
		return X
	def ffinal(self, X):
		# X: batch x N x D_mid
		# OUT: batch x N x D_final
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
		# A: batch x N x N
		# X: batch x K+1 x N x N x D
		batch, K, N, _, _ = X.shape; K -= 1
		I = torch.eye(N, device=device).repeat(batch,1,1,1).double()
		Ak = torch.cat([I, A.unsqueeze(1).expand(-1,K,-1,-1)], dim=1).unsqueeze(4) # Ak: batch x K+1 x N x N x 1
		# fpre
		fpre_out = self.fpre(X) # fpre_out: batch x K+1 x N x N x D_pre
		Z1 = (Ak * fpre_out).sum(dim=3) # Z1: batch x K+1 x N x Dpre
		# fmid
		fmid_out = self.fmid(Z1) # fmid_out: batch x K+1 x N x D_mid
		Z2 = fmid_out.sum(dim=1) # Z2: batch x N x D_post
		# ffinal
		output = self.ffinal(Z2) # output: batch x N x D_final
		return output



