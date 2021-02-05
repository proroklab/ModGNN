import torch
from ModGNN.GNNnode import GNNnode
import itertools
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GNN(nn.Module):

	def __init__(self, K=1, layers=[None], **kwargs):
		super().__init__()
		self.K = K
		self.layers = len(layers)
		self.network = list(map(lambda layer: GNNnode(**kwargs) if (layer is None) else layer, layers))
		for layer in range(self.layers):
			self.add_module("gnn_node_%d" % layer, self.network[layer])


	def forward(self, A, X):
		# A: batch x (K+1)+(nLayers-1) x N x N
		# X: batch x (K+1)+(nLayers-1) x N x D
		batch, _, N, Dobs = X.shape; K -= 1
		# Multi Layer
		layer_outputs = []
		for layer in range(self.layers):
			# Apply finput
			layer_time = self.layers-1 - layer
			obs = X[:,layer_time:layer_time+self.K+1,:,:].view(batch * (self.K+1) * N, Dobs)
			Xc = self.network[layer].finput(obs, *layer_outputs).view(batch, self.K+1, N, -1)
			Ac = A[:,layer_time:layer_time+self.K,:,:]
			Dinput = Xc.shape[-1]
			# Calculate Y[k], the aggregated information from a k-hop neighbourhood
			Y = {}
			Y[1] = {-t: Xc[:,t,:,:] for t in range(1,self.K+1)}
			for k in range(2,self.K+1):
				Y[k] = {}
				for t in range(1, self.K+1-(k-1)):
					yk = self.network[layer].fcom(Ac[:,k-1,:,:], Y[k-1][-t-1])
					Y[k][-t] = yk.sum(dim=2)
			# Calculate controller input Z
			Z = torch.zeros((batch, N, N, self.K+1, Dinput), device=Xc.device)
			Z[:,:,:,0,:] = torch.diag_embed(Xc[:,0,:,:].permute(0,2,1), dim1=1, dim2=2)
			for k in range(1,self.K+1):
				Z[:,:,:,k,:] = self.network[layer].fcom(Ac[:,0,:,:], Y[k][-1])
			# Apply the controller
			output = self.network[layer].forward(Ac[:,0,:,:], Z)
			layer_outputs.append(output)
		# Return output of last layer
		return layer_outputs[-1]

