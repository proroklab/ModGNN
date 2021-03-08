import torch
from ModGNN.GNNnode import GNNnode
import itertools
import torch.nn as nn


class GNN(nn.Module):
	"""
	A torch.nn.Module that handles the evaluation of a GNN in a centralised setting.
	Each layer of the GNN is defined by a single GNNnode. During training/evaluation, that GNNnode is "copied" for every agent in the system (parameter sharing)
	A tensor of the current and K preceding timesteps of the joint state X and adjacency matrix A is required to produce an outupt.

	Attributes
	----------
	K : [(int), default=1] The number of communication "hops" to use. Each agent will receive data grouped by the number of communications hops it took from its source. There are K+1 neighbourhoods, because the local K=0 data is included.
	layers: [(list), default=[None]] A list of GNNnode objects, each representing a layer of the GNN (usually a one-layer gnn  is sufficient len(layers)=1). If any elements of layers is None, a GNNnode will be constructed automatically with the given kwargs: GNNnode(**kwargs)

	Methods
	-------
	forward(A, X) : calculates the joint output given the last K+1 timesteps of the joint state and adjacency matrix
	"""

	def __init__(self, K=1, layers=[None], **kwargs):
		super().__init__()
		self.K = K
		self.layers = len(layers)
		self.network = list(map(lambda layer: GNNnode(K=K, **kwargs) if (layer is None) else layer, layers))
		for layer in range(self.layers):
			self.add_module("gnn_node_%d" % layer, self.network[layer])
		self.Z = [None for _ in range(self.layers)]
		self.layer_outputs = [None for _ in range(self.layers)]
		params = list(self.parameters())
		self.device = params[0].device if len(params)>0 else torch.device('cpu')


	def forward(self, A, X):
		"""
		Calculates the joint output given the last K+1 timesteps of the joint state and adjacency matrix

		Parameters
		----------
		A : [tensor of size (batch x (K+1)+(nLayers-1) x N x N)] The adjacency matrix. dim=0 is the batch dimension, dim=1 is the temporal dimension where index 0 is the current timestep. In dim=2,3, if agent i can receive data from agent j, then A[•,•,i,j]=1, and otherwise A[•,•,i,j]=0
		X : [tensor of size (batch x (K+1)+(nLayers-1) x N x Dobs)] The joint state. dim=0 is the batch dimension, dim=1 is the temporal dimension where index 0 is the current timestep. In dim=2,3, X[•,•,i,:] is the observation of agent i
		
		Output
		-------
		[tensor of size (batch x N x Dout)] The joint output of the multiagent system at the current timestep.

		"""
		if len(X.shape) == 3:
			A = A.unsqueeze(1).repeat(1,self.K+1,1,1)
			X = X.unsqueeze(1).repeat(1,self.K+1,1,1)
		batch, _, N, Dobs = X.shape  # A: batch x (K+1)+(nLayers-1) x N x N      X: batch x (K+1)+(nLayers-1) x N x D
		# Multi Layer
		for layer in range(self.layers):
			## Apply finput
			layer_time = self.layers-1 - layer
			obs = X[:,layer_time:layer_time+self.K+1,:,:].reshape(batch * (self.K+1) * N, Dobs) # TODO: delayed layers
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


	def to(self, device):
		super().to(device)
		self.device = device
