from ModGNN import GNN, GNNnode
import torch


def main():
	# init
	K = 1
	N = 5
	DIM = 3
	layer1 = GNNnode(K=K)
	gnn = GNN(K=K, layers=[layer1])
	# test data
	batch = 8
	X = torch.rand(batch, K+1, N, DIM)
	A = random_adjacency(N, 0.5).unsqueeze(0).unsqueeze(1).expand(batch,K+1,-1,-1)
	# evaluate
	y = gnn.forward(A, X)
	print(y.shape)


def random_adjacency(N, fill=0.6):
	B = (torch.rand(N,N) < torch.sqrt(torch.tensor(fill))).float()
	A = B * B.T
	torch.diagonal(A).fill_(0)
	return A


if __name__ == '__main__':
	main()