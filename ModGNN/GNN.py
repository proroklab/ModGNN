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
		self.Z = [None for _ in range(self.layers)]
		self.layer_outputs = [None for _ in range(self.layers)]


	def forward(self, A, X):
		# A: batch x (K+1)+(nLayers-1) x N x N
		# X: batch x (K+1)+(nLayers-1) x N x D
		batch, _, N, Dobs = X.shape
		# Multi Layer
		# TODO: delayed layers
		for layer in range(self.layers):
			## Apply finput
			layer_time = self.layers-1 - layer
			obs = X[:,layer_time:layer_time+self.K+1,:,:].view(batch * (self.K+1) * N, Dobs)
			Xc = self.network[layer].finput(obs, *self.layer_outputs[:-1]).view(batch, self.K+1, N, -1)
			Ac = A[:,layer_time:layer_time+self.K,:,:]
			Dinput = Xc.shape[-1]
			## Calculate Y[k]
			Y = {} # Y[k][-t] is the aggregated information from a k-hop neighbourhood from t timesteps in the past
			Y[0] = {-t: Xc[:,t,:,:] for t in range(1,self.K+1)} # for Y[0][-t], we use the local information from t timesteps ago
			for k in range(1,self.K): # k from 1 to K-1. for each Y[k][-1], we must calculate K-1 communications (t = 1 .. k)
				Y[k] = {} # this loop calculates Y[1][-1] to Y[K-1][-1] (Y[0][-1] is already calculated)
				for t in range(k): # t from 0 to k-1 (k ranges from 1 to K-1). to calculate Y[1][-1] (1-hop neighbourhood, 1 timestpe old), we use Ac[1] (1 timestep old adjacency) to communicate Y[0][-2] (0-hop neighbourhood, 2 timesteps old)
					yk = self.network[layer].fcom(Ac[:,k-t,:,:], Y[t][t-k-1]) # for k=1 (calculating Y[1][-1]), this gives Ac[1] and Y[0][-2]. for k=K-1 (calculating Y[K-1][-1]) it gives Ac[K-1] and Y[0][-K] when t=0 to Ac[1] and Y[K-2][-2] when t=K-2
					Y[t+1][t-k] = yk.sum(dim=2) # aggregates information from Y[neigbhourhood][timestep] into Y[neighbourhood+1][timestep+1]
			# Calculate controller input Z
			self.Z[layer] = torch.zeros((batch, N, N, self.K+1, Dinput), device=Xc.device) # the incoming data for all agents up to K hops. dim=1 is the selected agent, and dim=2 is the agent the data came from
			self.Z[layer][:,:,:,0,:] = torch.diag_embed(Xc[:,0,:,:].permute(0,2,1), dim1=1, dim2=2) # this adds the K=0 data for all agent i=j, where i corresponds to an agent in dim=1, and j corresponds to an agent in dim=2
			for k in range(1,self.K+1):
				self.Z[layer][:,:,:,k,:] = self.network[layer].fcom(Ac[:,0,:,:], Y[k-1][-1]) # last hop of communication (without aggregation). for k-hop data, the most recent adjacency Ac[0] is used to retrieve Y[k-1][-1] (the [k-1]-hop data from the last time step)
			# Apply the controller
			output = self.network[layer].forward(Ac[:,0,:,:], self.Z[layer]) # compute the output with the GNN node, given the k-hop data Z and the current adjacency Ac[0]
			self.layer_outputs[layer] = output
		# Return output of last layer
		return self.layer_outputs[-1] # output: batch x N x Dout

