from ModGNN import GNN, GNNnode
import torch
from torch import nn


def main():
	K=1
	# Implementation Method 1: extend abstract GNNnode class
	gnn1_layer1 = GNNnodeCustom(K=K)
	gnn1 = GNN(K=K, layers=[gnn1_layer1])
	# Implementation Method 2: function, parameters tuple
	gnn2_layer1 = GNNnode(K=K, finput=finput_gen(), fcom=fcom_gen(), fpre=fpre_gen(), fmid=fmid_gen(K=K), ffinal=ffinal_gen())
	gnn2 = GNN(K=K, layers=[gnn2_layer1])
	# show submodules
	print("Impmentation Method 1:")
	print(gnn1)
	print("Impmentation Method 2:")
	print(gnn2)

## Implementation Method 1

class GNNnodeCustom(GNNnode):

	def __init__(self, K=1, **kwargs):
		# Run GNNnode constructor
		super().__init__(K=K, **kwargs)
		# Define networks
		self.finput_net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,4), nn.ReLU())
		self.fpre_net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6,8), nn.ReLU())
		self.fmid_net = [nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,6), nn.ReLU()) for k in range(K+1)]
		self.ffinal_net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,3))
		# Register parameters
		self.register_params(self.fmid_net, "fmid") # modules defined in lists/dicts must be registered (added to the GNNnode's list of params) in order to be learnable
		
	def finput(self, X, *layer_outputs):
		# X: the joint state with shape [(batch*N*(K+1)) x Dobs]
		# layer_outputs: a list of outputs from previous layers in the GNN, each with shape [batch x N x Dobs]
		# output: the encoded observation to be communicated, with shape [batch*N*(K+1) x Dinput]
		return self.finput_net(X)

	def fcom(self, A, X):
		# A: the adjacency matrix with shape [batch x N x N]
		# X: the joint state with shape [batch x N x Dinput]
		# output: the incoming data (dim=2) for each agent (dim=1), with shape [batch x N x N x Dinput]
		return A[:,:,:,None] * X[:,None,:,:]

	def fpre(self, X):
		# X: the joint state with shape [(batch*N*N) x (K+1) x Dinput]
		# output: the processed inputs of shape [(batch*N*N) x (K+1) x Dpre]
		batchxNxN, K, Dinput = X.shape; K-=1
		Xr = X.view(batchxNxN * (K+1), Dinput)
		output = self.fpre_net(Xr) # applies the same fpre to all neighbourhoods
		return output.view(batchxNxN, K+1, -1)

	def fmid(self, X):
		# X: the joint state with shape [(batch*N) x (K+1) x Dpre]
		# output: the processed aggregated neighbourhoods of shape [(batch*N) x (K+1) x Dmid]
		return torch.stack([self.fmid_net[k](X[:,k,:]) for k in range(X.shape[1])], dim=1) # applies a different fmid to each neighbourhood

	def ffinal(self, X):
		# X: the joint state with shape [(batch*N) x Dmid]
		# output: the processed aggregated neighbourhoods of shape [(batch*N) x Dout]
		return self.ffinal_net(X)


## Implementation Method 2

def finput_gen(in_dim=6, out_dim=4):
	net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim), nn.ReLU())
	def finput(X, *layer_outputs):
		return net(X)
	return finput, net # can either pass a module (as in this case), or a generator/list of parameters (obtained by net.parameters())

def fcom_gen():
	def fcom(A, X):
		return A[:,:,:,None] * X[:,None,:,:]
	return fcom, None

def fpre_gen(in_dim=4, out_dim=8):
	net = nn.Sequential(nn.Linear(in_dim, int(0.5*in_dim+0.5*out_dim)), nn.ReLU(), nn.Linear(int(0.5*in_dim+0.5*out_dim), out_dim), nn.ReLU())
	def fpre(X):
		batchxNxN, K, Dinput = X.shape; K-=1
		Xr = X.view(batchxNxN * (K+1), Dinput)
		output = net(Xr)
		return output.view(batchxNxN, K+1, -1)
	return fpre, net # alternatively, instead of the list of modules you could pass itertools.chain(*[module.parameters() for module in net])

def fmid_gen(K=1, in_dim=8, out_dim=6):
	net = [nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim,out_dim), nn.ReLU()) for k in range(K+1)]
	def fmid(X):
		return torch.stack([net[k](X[:,k,:]) for k in range(X.shape[1])], dim=1)
	return fmid, net

def ffinal_gen(in_dim=6, out_dim=3):
	net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim), nn.ReLU())
	def ffinal(X):
		return net(X)
	return ffinal, net



if __name__ == '__main__':
	main()