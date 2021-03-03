from ModGNN import GNN, GNNnode
import torch
from torch import nn
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

# Train using GNN Benchmark Dataset
# Compare performance to other GNNs in the paper https://arxiv.org/pdf/2003.00982.pdf
# For the simplicity of the example, we train with a single-layer GNN, while the paper tests 4 and 16 layer

def main():
	# GNN
	K=3
	layer1 = GNNnodeCustom(K=K)
	gnn = GNN(K=K, layers=[layer1])
	model_path = "gnn_model.pt"
	load_model(gnn, model_path)
	# Optimiser
	optimiser = torch.optim.Adam(gnn.parameters(), lr=1e-3)
	loss_fn = nn.CrossEntropyLoss()
	# Tensorboard
	tensorboard = SummaryWriter()
	tb = program.TensorBoard()
	tb.configure(argv=[None, '--logdir', tensorboard.log_dir])
	url = tb.launch()
	# Dataset
	dataset = GNNBenchmarkDataset(root='/Users/ryko/Desktop/Work/PhD/datasets', name='PATTERN', split='train')
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
	# Train
	for step, sample in enumerate(dataloader):
		A = to_dense_adj(sample.edge_index)
		X = to_dense_batch(sample.x)[0]
		Ytruth = sample.y
		optimiser.zero_grad()
		Y = gnn(A, X)
		# if step==0:
		# 	tensorboard.add_graph(gnn, (A, X))
		loss = loss_fn(Y.view(Y.shape[0]*Y.shape[1],Y.shape[2]), Ytruth)
		loss.backward()
		optimiser.step()
		tensorboard.add_scalar('loss/train', loss, step)
		pred_class = torch.max(Y[0,:,:],dim=1)[1]
		correct = pred_class == Ytruth
		accuracy = torch.sum(correct) / correct.shape[0]
		tensorboard.add_scalar('accuracy/train', accuracy, step)
		for e in range(Y.shape[2]):
			prob = torch.nn.functional.softmax(Y[0,:,:], dim=1)[:,e]
			label = Ytruth == e
			tensorboard.add_pr_curve("train/Class %d" % e, label, prob, global_step=step)
		if step >= 3000:
			break
	save_model(gnn, model_path)
	# Test
	dataset_test = GNNBenchmarkDataset(root='/Users/ryko/Desktop/Work/PhD/datasets', name='PATTERN', split='test')
	dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
	gnn.eval()
	total_accuracy = [0,0]
	for idx, sample in enumerate(dataloader_test):
		A = to_dense_adj(sample.edge_index)
		X = to_dense_batch(sample.x)[0]
		Ytruth = sample.y
		Y = gnn(A, X)
		loss = loss_fn(Y.view(Y.shape[0]*Y.shape[1],Y.shape[2]), Ytruth)
		tensorboard.add_scalar('loss/test', loss, idx)
		correct = (torch.max(Y[0,:,:],dim=1)[1]) == Ytruth
		accuracy = torch.sum(correct) / correct.shape[0]
		tensorboard.add_scalar('accuracy/test', accuracy, idx)
		total_accuracy[0] += torch.sum(correct)
		total_accuracy[1] += correct.shape[0]
	print("Test Accuracy: %g" % (total_accuracy[0]/total_accuracy[1]))
	tensorboard.close()



class GNNnodeCustom(GNNnode):

	def __init__(self, K=1, **kwargs):
		# Run GNNnode constructor
		super().__init__(K=K, **kwargs)
		# Define networks
		self.finput_net = nn.Sequential(nn.Linear(3, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6,6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6,6), nn.BatchNorm1d(6), nn.ReLU())
		self.fpre_net = [nn.Sequential(nn.Linear(6, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8,12), nn.BatchNorm1d(12), nn.ReLU()) for k in range(K+1)]
		self.fmid_net = [nn.Sequential(nn.Linear(12, 12), nn.BatchNorm1d(12), nn.ReLU(), nn.Linear(12, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8,6), nn.BatchNorm1d(6), nn.ReLU()) for k in range(K+1)]
		self.ffinal_net = nn.Sequential(nn.Linear(6, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6,2))
		# Register parameters
		self.register_params(self.fpre_net, "fpre")
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
		return torch.stack([self.fpre_net[k](X[:,k,:]) for k in range(X.shape[1])], dim=1)

	def fmid(self, X):
		# X: the joint state with shape [(batch*N) x (K+1) x Dpre]
		# output: the processed aggregated neighbourhoods of shape [(batch*N) x (K+1) x Dmid]
		return torch.stack([self.fmid_net[k](X[:,k,:]) for k in range(X.shape[1])], dim=1) # applies a different fmid to each neighbourhood

	def ffinal(self, X):
		# X: the joint state with shape [(batch*N) x Dmid]
		# output: the processed aggregated neighbourhoods of shape [(batch*N) x Dout]
		return self.ffinal_net(X)



def load_model(model, path):
	try:
		fp = open(path,'rb')
	except Exception as exception:
		print("No Model to Load: %s" % exception)
		return
	model_dict = torch.load(fp)
	model.load_state_dict(model_dict)
	print("Loaded Model " + path)
	return model


def save_model(model, path):
	model_dict = model.state_dict()
	with open(path, 'wb') as fp:
		torch.save(model_dict, fp)
	print("Saved Model " + path)


if __name__ == '__main__':
	main()