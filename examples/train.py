from ModGNN import GNN, GNNnode
import torch
from torch import nn
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj


def main():
	# GNN
	K=2
	layer1 = GNNnodeCustom(K=K)
	gnn = GNN(K=K, layers=[layer1])
	# Optimiser
	optimiser = torch.optim.Adam(gnn.parameters(), lr=1e-3)
	loss_fn = nn.CrossEntropyLoss()
	# Dataset
	dataset = GNNBenchmarkDataset(root='/Users/ryko/Desktop/Work/PhD/datasets', name='PATTERN', split='train')
	dataloader = DataLoader(dataset, batch_size=1)
	# Train
	for idx, sample in enumerate(dataloader):
		A = to_dense_adj(sample.edge_index)
		X = to_dense_batch(sample.x)[0]
		Ytruth = sample.y
		optimiser.zero_grad()
		Y = gnn(A, X)
		loss = loss_fn(Y.view(Y.shape[0]*Y.shape[1],Y.shape[2]), Ytruth)
		loss.backward()
		optimiser.step()
		print(loss)



class GNNnodeCustom(GNNnode):

	def __init__(self, K=1, **kwargs):
		# Run GNNnode constructor
		super().__init__(K=K, **kwargs)
		# Define networks
		self.finput_net = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6,4), nn.ReLU())
		self.fpre_net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6,8), nn.ReLU())
		self.fmid_net = [nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,6), nn.ReLU()) for k in range(K+1)]
		self.ffinal_net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,2))
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


if __name__ == '__main__':
	main()