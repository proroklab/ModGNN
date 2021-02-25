
# ModGNN
A modular graph neural network framework implemented in pytorch. The framework handles all of the GNN's communication and aggregation operations, allowing the user to focus on a simple interface. An entire GNN architecture can be defined by simply implementing a set of submodules. The framework has a very general architecture, so most GNNs from existing literature can be represented as special cases. The framework itself is an instance of a torch.nn.Module, so it may be saved/trained/evaluated just like any other torch module. Once implemented, the GNN may be evaluated either in a centralised setting (which simultaneously evaluates a batch of joint states) or a decentralised setting (which performs operations for a single agent at a single timestep, and requires the user to manually implement communication).

* [Installation](#installation)
* [Example](#example)
* [Documentation](#documentation)
* [Architecture](#architecture)

## Installation

    python3 -m pip install https://github.com/Acciorocketships/ModGNN

## Example
```python
from ModGNN import GNN, GNNnode
import torch
from torch import nn

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


K = 2
N = 100
gnn = GNN(K=K, layers=[GNNnodeCustom(K=K)]) # GNN is another torch.nn.Module which wraps our local operations in GNNnode, allowing centralised evaluation
A = (torch.rand(1, K+1, N, N) > 0.5).float() # A : batch x K+1 x N x N, a random adjacency for K+1 consecutive timesteps
X = torch.randn(1, K+1, N, 6) # X : batch x K+1 x N x Dobs, a random joint state for K+1 consecutive timesteps
Y = gnn(A, X) # Y : batch x N x Dout
```
For more examples, see the ModGNN/examples folder.

## Documentation
The ```ModGNN.GNNnode``` class implements local GNN operations, while the ```ModGNN.GNN``` class acts as a wrapper for GNNnode, allowing centralised evaluation and training. For detailed documentation about these classes and their methods, use python's help function (ex: ```help(ModGNN.GNNnode)```, ```help(ModGNN.GNNnode.input)```).

## Architecture



### Multi-Layer Architecture

![Our framework's entire multi-layer architecture. At each layer, the message aggregation module disseminates the output from the last layer, and then the GNN node uses the data from all of the previous layers to compute an output.](https://github.com/Acciorocketships/ModGNN/blob/main/images/WholeArchitecture.png)

Our framework provides infrastructure for multi-layer GNNs. Each layer consists of a message aggregation module to transmit data, and a GNN node to compute the output. For extra flexibility, the input consists of the outputs of all previous layers, but this extra data can easily be masked out if it is not required. Each layer can have different architectures for analogous submodules, but in some cases it makes sense to use the same architecture with parameter sharing.

### Message Aggregation Module

![Our framework's message aggregation module. First, the raw observation or output from the last layer is transformed by f_input. Then, for each transmission up to K hops, the data from the neighbouring agents is passed through f_com and then aggregated. The input to the GNN node is the set of the data from each $k$-hop neighbourhood up to K.](https://github.com/Acciorocketships/ModGNN/blob/main/images/CommunicationSystem.png)

The first step in our message aggregation module is to compress the raw observation with an $f_\mathrm{input}$ function. This step is not only useful for transforming the observation into the desired shape (for example, a CNN can be applied to image observations to flatten the data before it is transmitted---it also provides an opportunity for the observation to be transformed before all of the aggregation steps. Aggregation is a lossy operation, so it is important to transform the data into a space that preserves the most important information. Qualitatively, the purpose of the $f_\mathrm{input}$ submodule can be viewed as learning which information to communicate.

First, we compute $\mathbf{c}_i^{(l)}(t)$, the compressed input to agent i at layer l. To do this, we apply $f_\mathrm{input}^{(l)}$ from layer l to the set of outputs from all of the previous layers:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathbf{c}_i^{(l)}(t)&space;=&space;f_\mathrm{input}^{(l)}\left(\left\{\mathbf{x}_i^{(m)}(t)&space;\;&space;\middle|&space;\;&space;m&space;\in&space;[0..l-1]&space;\right\}\right)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\mathbf{c}_i^{(l)}(t)&space;=&space;f_\mathrm{input}^{(l)}\left(\left\{\mathbf{x}_i^{(m)}(t)&space;\;&space;\middle|&space;\;&space;m&space;\in&space;[0..l-1]&space;\right\}\right)" title="\mathbf{c}_i^{(l)}(t) = f_\mathrm{input}^{(l)}\left(\left\{\mathbf{x}_i^{(m)}(t) \; \middle| \; m \in [0..l-1] \right\}\right)" /></a>

Next, data is transmitted from each agent to all of its neighbours. The data from previous timesteps are cached, so an agent can obtain k-hop data at time t by requesting (k-1)-hop data from time t-1 from its neighbours. The benefit to this communication scheme is that only one communication is required per timestep. The GCN does not specifically define a message aggregation module because it is formulated in a centralised setting, but the naive method is to perform K successive communication steps.

Every time agent i receives data from its neighbours, the $|\mathcal{N}_i(t)|$ incoming vectors are passed through an $f_\mathrm{com}^{(l)}$ function, and then aggregated together. The $f_\mathrm{com}^{(l)}$ submodule defines the graph shift operator (GSO). For example, if $f_\mathrm{com}^{(l)}$ subtracts the local state from each incoming state, then the resulting GSO is the Laplacian. One can also use $f_\mathrm{com}^{(l)}$ to implement an attention mechanism \cite{GraphAttention} or a coordinate transformation system to shift the observations into the local reference frame.

Let $\mathbf{y}_{ij}^{(l)(k)}(t)$ be the data in layer l from a k-hop neighbourhood received by agent i at time t. We define $\mathcal{Y}_i^{(l)(k)}(t)$ as the set of all transmissions that agent i receives at time t from a k-hop neighbourhood in layer l:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathcal{Y}_i^{(l)(k)}(t)&space;=&space;\left\{&space;\mathbf{y}_{ij}^{(l)(k)}(t)&space;\;&space;\middle|&space;\;&space;j&space;\in&space;\mathcal{N}_i(t)&space;\right\}&space;." target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\mathcal{Y}_i^{(l)(k)}(t)&space;=&space;\left\{&space;\mathbf{y}_{ij}^{(l)(k)}(t)&space;\;&space;\middle|&space;\;&space;j&space;\in&space;\mathcal{N}_i(t)&space;\right\}&space;." title="\mathcal{Y}_i^{(l)(k)}(t) = \left\{ \mathbf{y}_{ij}^{(l)(k)}(t) \; \middle| \; j \in \mathcal{N}_i(t) \right\} ." /></a>

We obtain each $\mathbf{y}_{ij}^{(l)(k)}(t)$ in this set by applying the $f_\mathrm{com}^{(l)}$ function of layer l and a summation to the (k-1)-hop data at each neighbour j, and then communicating the result:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathbf{y}_{ij}^{(l)(k)}(t)&space;=&space;{\sum_{\mathbf{z}&space;\in&space;\mathcal{Y}_j^{(l)(k-1)}(t-1)}}&space;\;&space;f_\mathrm{com}^{(l)}\left(\mathbf{z}\right)&space;." target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\mathbf{y}_{ij}^{(l)(k)}(t)&space;=&space;{\sum_{\mathbf{z}&space;\in&space;\mathcal{Y}_j^{(l)(k-1)}(t-1)}}&space;\;&space;f_\mathrm{com}^{(l)}\left(\mathbf{z}\right)&space;." title="\mathbf{y}_{ij}^{(l)(k)}(t) = {\sum_{\mathbf{z} \in \mathcal{Y}_j^{(l)(k-1)}(t-1)}} \; f_\mathrm{com}^{(l)}\left(\mathbf{z}\right) ." /></a>

As a base case for this recursive definition, the 0-hop data $\mathcal{Y}_i^{(l)(0)}(t)$ is defined as $\mathbf{c}_i^{(l)}(t)$, the output of $f_\mathrm{input}$:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathcal{Y}_i^{(l)(0)}(t)&space;=&space;\left\{&space;\mathbf{c}_i^{(l)}(t)&space;\right\}&space;." target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\mathcal{Y}_i^{(l)(0)}(t)&space;=&space;\left\{&space;\mathbf{c}_i^{(l)}(t)&space;\right\}&space;." title="\mathcal{Y}_i^{(l)(0)}(t) = \left\{ \mathbf{c}_i^{(l)}(t) \right\} ." /></a>

At each timestep, the input to the GNN node of agent i is given by the set of data from all neighbourhoods $\mathcal{Y}_i^{(k)}(t)$ up to the user-defined maximum number of hops K:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathcal{Z}_i^{(l)}(t)&space;=&space;\left\{&space;\mathcal{Y}_i^{(l)(k)}(t)&space;\;&space;\middle|&space;\;&space;k&space;\in&space;[0..K]&space;\right\}&space;." target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\mathcal{Z}_i^{(l)}(t)&space;=&space;\left\{&space;\mathcal{Y}_i^{(l)(k)}(t)&space;\;&space;\middle|&space;\;&space;k&space;\in&space;[0..K]&space;\right\}&space;." title="\mathcal{Z}_i^{(l)}(t) = \left\{ \mathcal{Y}_i^{(l)(k)}(t) \; \middle| \; k \in [0..K] \right\} ." /></a>

### GNN Node Architecture

![Our framework's GNN Node architecture. First, f_pre is applied to the incoming transmissions from each neighbour, and the data from those neighbours is aggregated together (in effect, completing the last aggregation of the message aggregation module). Next, the combined data from each of the K+1 neighbourhoods is passed through f_mid and aggregated together. Lastly, f_final is applied to produce the final output.](https://github.com/Acciorocketships/ModGNN/blob/main/images/GNNnode.png)

The GNN node is comprised of two aggregation steps and three user-defined submodules. The first aggregation step combines the states from the $\mathcal{N}_i(t)$ neighbours of agent i (summing along the same dimension as the aggregation operations in the message aggregation module). The second aggregation step combines data from the K+1 different neighbourhoods. The three user-defined submodules are interspersed throughout the model in the spaces between the aggregation steps. We use $\mathbf{x}^{(l)}_i(t)$ to represent the output of the GNN node

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\mathbf{x}^{(l)}_i(t)&space;=&space;f_\mathrm{final}^{(l)}&space;\left(&space;{\sum_{k=0}^{K}}&space;\left[&space;f_\mathrm{mid}^{(l)(k)}&space;\left(&space;{\sum_{\mathbf{z}&space;\in&space;\mathcal{Y}_i^{(l)(k)}(t)}}&space;f_\mathrm{pre}^{(l)(k)}\left(\mathbf{z}\right)&space;\right)&space;\right]&space;\right)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\mathbf{x}^{(l)}_i(t)&space;=&space;f_\mathrm{final}^{(l)}&space;\left(&space;{\sum_{k=0}^{K}}&space;\left[&space;f_\mathrm{mid}^{(l)(k)}&space;\left(&space;{\sum_{\mathbf{z}&space;\in&space;\mathcal{Y}_i^{(l)(k)}(t)}}&space;f_\mathrm{pre}^{(l)(k)}\left(\mathbf{z}\right)&space;\right)&space;\right]&space;\right)" title="\mathbf{x}^{(l)}_i(t) = f_\mathrm{final}^{(l)} \left( {\sum_{k=0}^{K}} \left[ f_\mathrm{mid}^{(l)(k)} \left( {\sum_{\mathbf{z} \in \mathcal{Y}_i^{(l)(k)}(t)}} f_\mathrm{pre}^{(l)(k)}\left(\mathbf{z}\right) \right) \right] \right)" /></a>

where $f_\mathrm{pre}^{(l)(k)}$, $f_\mathrm{mid}^{(l)(k)}$, and $f_\mathrm{final}^{(l)}$ are all submodules from layer l of the GNN node. Optionally, $f_\mathrm{pre}^{(l)(k)}$ and $f_\mathrm{mid}^{(l)(k)}$ may include K+1 separate networks, each of which is applied to a specific neighbourhood k.
