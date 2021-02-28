import torch.nn as nn
import torch
import types


class GNNnode(nn.Module):
	"""
	A torch.nn.Module that defines the local functions in a single node of a GNN.
	The GNN architecture is defined by implementing the finput, fcom, fpre, fmid, and ffinal submodules.
	This model can be trained or evaluated either in a centralised mode or a decentralised mode.
	In  centralised mode, the incoming transmissions for all agents are stacked into a single tensor, and the outputs of all nodes are evaluated simultaneously. 
	Centralised mode uses parameter sharing, so it applies a "copy" of the GNNnode to each agent in the system. It uses vectorised operations, so it is faster than running decentralised mode for each agent.
	In decentralised mode, the model only computes the output of one node. The same submodule implementations are used. 
	Decentralised mode provides functions 'input(obs)', 'send()', 'receive(incoming)', and 'compute()', which decouples the input/communication/output computation, and requires the user to manually handle communication.

	Attributes
	----------
	K : [(int), optional, default=1] The number of communication "hops" to store. K=0 means only local information. Note that this input only affects decentralised mode--to changed the number of hops for centralised mode, see the GNN class
	finput : [((function, parameters) tuple), optional, default=lambda x: x] A tuple containing the finput submodule and its learnable parameters.
	fcom : [((function, parameters) tuple), optional, default=the laplacian GSO] A tuple containing the fcom submodule and its learnable parameters.
	fpre : [((function, parameters) tuple), optional, default=lambda x: x] A tuple containing the fpre submodule and its learnable parameters.
	fmid : [((function, parameters) tuple), optional, default=lambda x: x] A tuple containing the fmid submodule and its learnable parameters.
	ffinal : [((function, parameters) tuple), optional, default=lambda x: x] A tuple containing the ffinal submodule and its learnable parameters.

	Methods
	-------
	finput(X, *layer_outputs) : A submodule that should be overridden in modules that inherit GNNnode (it is also set if finput is given when GNNnode is instantiated)
	fcom(A, X) : A submodule that should be overridden in modules that inherit GNNnode (it is also set if fcom is given when GNNnode is instantiated)
	fpre(X) : A submodule that should be overridden in modules that inherit GNNnode (it is also set if fpre is given when GNNnode is instantiated)
	fmid(X) : A submodule that should be overridden in modules that inherit GNNnode (it is also set if fmid is given when GNNnode is instantiated)
	ffinal(X) : A submodule that should be overridden in modules that inherit GNNnode (it is also set if ffinal is given when GNNnode is instantiated)
	register_params(params, name) : A method of adding a module, list of modules, or list/generator of parameters to the GNNnode module, allowing those parameters to be learnable.
	input(obs) : For use in decentralised mode, this sets the current observation and calls finput to process the observation.
	send() : For use in decentrlised mode, this returns the (K-1)-hop data that should be transmitted to all neighbours.
	receive(incoming) : For use in decentralised mode, this updates the state with a list of incoming transmissions from the neighbours. Each element in the list is the output from another agent's send() method.
	compute() : For use in decentralised mode, this computes a new output given the current internal state which was set with the input(obs) and receive(incoming) methods.
	clear() : For use in decentralised mode, this resets the stored data in the GNNnode. It might be useful to call this in between episodes.
	"""

	def __init__(self, K=1, **kwargs):
		super().__init__()
		for fname, fgen in kwargs.items():
			if fname in ['finput','fcom','fpre','fmid','ffinal']:
				self.add_submodule(fname, fgen[0], fgen[1])
		self.K = K
		params = list(self.parameters())
		self.device = params[0].device if len(params)>0 else torch.device('cpu')
		self.clear()


	def finput(self, X, *layer_outputs):
		"""
		Processes the input into a compressed representation before communication

		Parameters
		----------
		X : [tensor of size ((batch * N * (K+1)) x Dobs)] The joint state, reshaped so the number of agents and timesteps are included in the batch
		layer_outputs : [list with tensors of size (batch x N x Dout)] The outputs of all of the preceding layers

		Output
		------
		[tensor of size ((batch * N * (K+1)) x D_input)] The input X transformed from dimension Dobs into Dinput
		"""
		return X # X: (batch * N * (K+1)) x D_obs      OUT: (batch * N * (K+1)) x D_input

	def fcom(self, A, X):
		"""
		Dictates how data is transformed as it is communicated by specifying the incoming data at each node. It is applied to each k-hop neighbourhood / timestep separately.
		For example, fcom is applied to the 0-hop data from the last timestep to obtain the 1-hop data at the current timestep.

		Parameters
		----------
		A : [tensor of size (batch x N x N)] The adjacency matrix from a specific timestep
		X : [tensor of size (batch x N x Dinput)] The joint state consisting of all data from a specific k-hop neighbourhood.

		Output
		------
		[tensor of size (batch x N x N x Dinput)] The incoming data at each node. output[•,i,j,:] represents the data sent from agent j to agent i (which might be transformed by agent i upon reception)
		"""
		return (A[:,:,:,None] * X[:,None,:,:]) - (A[:,:,:,None] * X[:,:,None,:])   # A: batch x N x N     X: batch x N x D_input    OUT: batch x N x N x D_input

	def fpre(self, X):
		"""
		Applies a transformation before the data from the neighbours is aggregated by neighbourhood.
		The input is a reshaped version of the tensor that represents the incoming transmissions for all nodes, X' : batch x N x N x K+1 x Dinput, where X'[•,i,j,:,:] represents the K-hop data from agent j received by agent i
		Immediately after fpre is applied, the output is reshaped to batch x N x N x K+1 x Dpre, and a summation is performed on dim=2 (the neighbouring agents)
		The K+1 neigbhourhoods are exposed outside the batch dimension because it is often useful to apply a different network to each neighbourhood.

		Parameters
		----------
		X : [tensor of size ((batch * N * N) x K+1 x Dinput)] A reshaped version of the incoming communications tensor
		
		Output
		------
		[tensor of size (batch * N * N) x K+1 x Dpre] The transformed output
		"""
		return X # X: (batch * N * N) x K+1 x D_input     OUT: (batch * N * N) x K+1 x D_pre

	def fmid(self, X):
		"""
		Applies a transformation before the separate neigbhourhoods are aggregated together
		The input is a reshaped version of the tensor that represents the aggregated data from each neighbourhood, X' : batch x N x K+1 x Dpre, where X'[•,i,k,:] represents the aggregated k-hop neighbourhood of agent i.
		Immediately after fmid is applied, the output is reshaped to batch x N x K+1 x Dmid, and a summation is performed on dim=2 (the neighbourhoods)
		The K+1 neigbhourhoods are exposed outside the batch dimension because it is often useful to apply a different network to each neighbourhood.
		
		Parameters
		----------
		X : [tensor of size ((batch * N) x K+1 x Dpre)] A reshaped version of the tensor of aggregated neigbhourhoods

		Output
		------
		[tensor of size (batch * N) x K+1 x Dmid] The transformed output
		"""
		return X # X: (batch * N) x K+1 x D_pre      OUT: (batch * N) x K+1 x D_mid
		
	def ffinal(self, X):
		"""
		Applies a transformation to yield the final output.
		The input is the latent state after all aggregations have been applied, and just before the final output.

		Parameters
		----------
		X : tensor of size ((batch * N) x Dmid)] A rehsaped version of the computed latent state for each agent just before the final output

		Output
		------
		[tensor of size ((batch * N) x Dout)] The final output
		"""
		return X # X: (batch * N) x D_mid    OUT: (batch * N) x D_out


	def add_submodule(self, fname, func, params):
		# fgen is a tuple of (submodule, param_list)
		setattr(self, fname, func)
		self.register_params(params, name=fname)
		

	def register_params(self, params, name=""):
		"""
		Adds parameters to the GNNnode module so they are learnable (they will show up when you pass gnn.parameters() to the optimiser)
		There is no need to run this if you directly assign a module to a class variable (ex: self.net = nn.Linear(2,3)), as it will be added automatically.
		You only need to run this if the parameters are not added automatically, which happens if they are stored in a list or dict.

		Parameters
		----------
		params : [(list) OR (dict) OR (generator) OR (torch Parameter) OR (torch Module)] A torch parameter or module, or a list/dict/generator of torch parameters or modules
		"""
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
				self.add_module("%s_%d" % (name, i), layer)


	def forward(self, A, X):
		# A: batch x N x N (most recent adjacency matrix)
		# X: batch x N x N x K+1 x D (last K+1 time steps of the joint state)
		batch, N, _, K, _ = X.shape; K -= 1
		I = torch.eye(N, device=self.device).unsqueeze(0).unsqueeze(3).repeat(batch,1,1,1)
		Ak = torch.cat([I, A.unsqueeze(3).expand(-1,-1,-1,K)], dim=3).unsqueeze(4) # Ak: batch x N x N x K+1 x 1
		# fpre
		fpre_out = self.fpre(X.view(batch*N*N, K+1, -1)).view(batch, N, N, K+1, -1) # fpre_out: batch x N x N x K+1 x D_pre
		Z1 = (Ak * fpre_out).sum(dim=2) # Z1: batch x N x K+1 x Dpre
		# fmid
		fmid_out = self.fmid(Z1.view(batch*N, K+1, -1)).view(batch, N, K+1, -1) # fmid_out: batch x N x K+1 x D_mid
		Z2 = fmid_out.sum(dim=2) # Z2: batch x N x D_post
		# ffinal
		output = self.ffinal(Z2.view(batch*N, -1)).view(batch, N, -1) # output: batch x N x D_final
		return output


	def input(self, obs):
		"""
		Sets the current observation.
		For decentralised mode.

		Parameters
		----------
		obs : [tensor of size (Dobs)] the observation for a single agent at a single timestep
		"""
		# x : D_obs
		x = self.finput(obs)
		self.Dinput = x.shape[0]
		if self.Y is None:
			self.Y = torch.zeros(1, self.K+1, self.Dinput)
		self.Y[0,0,:] = x
		self.x = x


	def receive(self, incoming):
		"""
		Sets the incoming data from the connected neighbour.
		The input to receive(incoming) is a list of the outputs of send() for all connected neighbours.
		For decentralised mode.

		Parameters
		----------
		incoming : [list of tensors, each of size (K x Dinput)] The aggregated neighbourhoods up to (K-1)-hops for each neigbhour
		"""
		Ni = len(incoming)
		X = torch.stack([self.y] + incoming, dim=1) # X: K x Ni+1 x Dinput
		A0 = torch.zeros(Ni+1, Ni+1, device=self.device)
		A0[0,1:] = 1
		A = A0.repeat(self.K,1,1) # A: K x N+1 x N+1
		Xc = self.fcom(A, X)[:,0,:,:] # Xc: K x Ni x Dinput
		self.Y = torch.zeros(Ni+1, self.K+1, self.Dinput, device=self.device) # Y: Ni+1 x K+1 x Dinput
		self.Y[0,0,:] = self.x
		self.Y[1:,1:,:] = Xc[:,1:,:].permute(1,0,2)


	def send(self):
		"""
		Gets the outgoing data that will be sent to the connected neighbours.
		For decentralised mode.
		
		Output
		------
		[tensor of size (K x Dinput)] The aggregated neighbourhoods from k=0 to K=K-1
		"""
		self.y = self.Y[:,:-1,:].sum(dim=0)
		return self.y


	def compute(self):
		"""
		Computes the output given the current stored local and incoming data.
		For decentralised mode.

		Output
		------
		[tensor of size (Dout)] The output of the GNN at this agent for the current timestep
		"""
		X = self.Y
		Ni, _, _ = X.shape; Ni -= 1
		mask = torch.cat([torch.zeros(Ni+1, 1, 1), torch.ones(Ni+1, self.K, 1)], dim=1)
		mask[0,0,:] = 1
		mask[0,1:,:] = 0
		# fpre
		fpre_out = self.fpre(X) # fpre_out: Ni+1 x K+1 x D_pre
		Z1 = (mask * fpre_out).sum(dim=0).unsqueeze(0) # Z1: 1 x K+1 x Dpre
		# fmid
		fmid_out = self.fmid(Z1) # fmid_out: 1 x K+1 x D_mid
		Z2 = fmid_out.sum(dim=1) # Z2: 1 x D_post
		# ffinal
		output = self.ffinal(Z2) # output: 1 x D_out
		return output[0,:]


	def clear(self):
		"""
		Clears the stored data that was set by the input(obs) and receive(incoming) methods.
		For decentralised mode.
		"""
		self.Y = None
		self.y = None
		self.x = None


	def to(self, device):
		super().to(device)
		self.device = device





