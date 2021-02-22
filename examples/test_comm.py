from ModGNN import GNN, GNNnode
import torch


def main():
	# init
	K = 3
	N = 5
	DIM = 3
	layer1 = GNNnode(K=K)
	gnn = GNN(K=K, layers=[layer1])
	# test data
	batch = 8
	X = torch.rand(batch, K+1, N, DIM)
	A = random_adjacency(N, 0.5).unsqueeze(0).unsqueeze(1).expand(batch,K+1,-1,-1)
	# for GNN, get post communication data Z: batch x N x K+1 x D
	y = gnn.forward(A, X)
	Z_gnn = gnn.Z[0] # gnn Z is originally in form batch x N x N x K+1 x D (incoming data from neighours is not aggregated)
	Z_gnn = Z_gnn.sum(dim=2) # aggregate neighbour data in dim=2 to match GCN
	# for GCN, get the same post communication data Z: batch x N x K+1 x D
	Z_gcn = GCN(A, X)
	# compare
	equal = Z_gnn == Z_gcn
	diff = torch.abs(Z_gnn - Z_gcn)
	print("Maximum deviation: ", torch.max(diff))
	try:
		assert torch.all(diff < 1e-5)
	except:
		print("Difference found in GCN and GNN communication")


def random_adjacency(N, fill=0.6):
	B = (torch.rand(N,N) < torch.sqrt(torch.tensor(fill))).float()
	A = B * B.T
	torch.diagonal(A).fill_(0)
	return A


def GCN(A, X):
	batch, K, N, DIM = X.shape; K-=1
	Z = torch.zeros(batch, N, K+1, DIM)
	Z[:,:,0,:] = X[:,0,:,:]
	for k in range(1,K+1):
		S = torch.eye(N).unsqueeze(0).expand(batch,-1,-1)
		for t in range(k):
			S = torch.bmm(S, laplacian(A[:,t,:,:]))
		Z[:,:,k,:] = torch.bmm(S, X[:,k,:,:])
	return Z # returns the aggregated K-hop neighbourhoods for each agent


def laplacian(A):
	if len(A.shape) == 3:
		degree = torch.sum(A, dim=-1)[:,:,None] * torch.eye(A.shape[-1])[None,:,:]
	elif len(A.shape) == 2:
		degree = torch.sum(A, dim=-1) * torch.eye(A.shape[-1])
	laplacian = A - degree
	return laplacian


if __name__ == '__main__':
	main()