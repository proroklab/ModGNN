
# ModGNN
A modular graph neural network framework implemented in pytorch. The framework handles all of the GNN's communication and aggregation operations, allowing the user to focus on a simple interface. An entire GNN architecture can be defined by simply implementing a set of submodules. The framework has a very general architecture, so most GNNs from existing literature can be represented as special cases. The framework itself is an instance of a torch.nn.Module, so it may be saved/trained/evaluated just like any other torch module. Once implemented, the GNN may be evaluated either in a centralised setting (which simultaneously evaluates a batch of joint states) or a decentralised setting (which performs operations for a single agent at a single timestep, and requires the user to manually implement communication).

* [Installation](#installation)
* [Example](#example)
* [Documentation](#documentation)
* [Architecture](#architecture)
* [Citation](#citation)

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

### Message Aggregation Module

![Our framework's message aggregation module. First, the raw observation or output from the last layer is transformed by f_input. Then, for each transmission up to K hops, the data from the neighbouring agents is passed through f_com and then aggregated. The input to the GNN node is the set of the data from each $k$-hop neighbourhood up to K.](https://github.com/Acciorocketships/ModGNN/blob/main/images/CommunicationSystem.png)

### GNN Node Architecture

![Our framework's GNN Node architecture. First, f_pre is applied to the incoming transmissions from each neighbour, and the data from those neighbours is aggregated together (in effect, completing the last aggregation of the message aggregation module). Next, the combined data from each of the K+1 neighbourhoods is passed through f_mid and aggregated together. Lastly, f_final is applied to produce the final output.](https://github.com/Acciorocketships/ModGNN/blob/main/images/GNNnode.png)

### Multi-Layer Architecture

![Our framework's entire multi-layer architecture. At each layer, the message aggregation module disseminates the output from the last layer, and then the GNN node uses the data from all of the previous layers to compute an output.](https://github.com/Acciorocketships/ModGNN/blob/main/images/WholeArchitecture.png)

## Citation
If you use ModGNN in your work, please cite:
```
@article{kortvelesy2021modgnn,
    title = {ModGNN: Expert Policy Approximation in Multi-Agent Systems with a Modular Graph Neural Network Architecture},
    author = {Ryan Kortvelesy and Amanda Prorok},
    journal = {International Conference on Robotics and Automation (ICRA)}
    year = {2021},
    archivePrefix = {arXiv},
    eprint = {2103.13446},
    primaryClass = {cs.LG}
}
```
