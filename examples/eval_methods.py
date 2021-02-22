from ModGNN import GNN, GNNnode
import torch


def main():
	# init
	K = 1
	N = 5
	DIM = 3
	# test data
	batch = 8
	X = torch.rand(batch, K+1, N, DIM)
	A = random_adjacency(N, 0.5).unsqueeze(0).unsqueeze(1).expand(batch,K+1,-1,-1)
	# evaluate centralised
	layer1 = GNNnode(K=K)
	gnn = GNN(K=K, layers=[layer1])
	yc = gnn.forward(A, X)
	print(yc.shape)
	# evaluate decentralised
	gnn_nodes = [GNNnode(K=K) for _ in range(N)]
	yd = eval_decentralised(gnn_nodes, A, X)
	print(yd.shape)
	# compare
	try:
		assert torch.all(yc==yd)
	except:
		print("decentralised execution does not match centralised execution")


def random_adjacency(N, fill=0.6):
	B = (torch.rand(N,N) < torch.sqrt(torch.tensor(fill))).float()
	A = B * B.T
	torch.diagonal(A).fill_(0)
	return A


def eval_decentralised(gnn_nodes, A, X):
	# A: batch x K+1 x N x N
	# X: batch x K+1 x N x Dobs
	batch, K, N, Dobs = X.shape; K-=1
	output = []
	for sample in range(batch):
		out = None
		for k in range(K+1):
			Ak = A[sample,K-k,:,:]
			Xk = X[sample,K-k,:,:]
			out = step_decentralised(gnn_nodes, Ak, Xk)
		output.append(out)
	output = torch.stack(output, dim=0)
	return output



def step_decentralised(gnn_nodes, A, X):
	# A: N x N
	# X: N x Dobs
	transmitted = []
	output = []
	# Process observations and send message # TODO: switch order
	for i, node in enumerate(gnn_nodes):
		node.input(X[i,:])
		out = node.compute()
		output.append(out)
	output = torch.stack(output, dim=0) # output: N x Dout
	# Send message
	for i, node in enumerate(gnn_nodes):
		sent_data = node.send()
		transmitted.append(sent_data)
	transmitted = torch.stack(transmitted, dim=0)
	# Receive messages
	for i, node in enumerate(gnn_nodes):
		N, K, Dinput = transmitted.shape
		incoming = list(transmitted.view(N, K*Dinput)[A[i,:].bool(),:].view(-1, K, Dinput))
		node.receive(incoming)
	return output



if __name__ == '__main__':
	main()