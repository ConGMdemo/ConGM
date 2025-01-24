import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch.nn import ModuleList
from scipy.stats import norm


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.degree_matrix = None

    def forward(self, x, A):
        if self.degree_matrix is None:
            D = torch.diag(A.sum(1))  # Degree matrix
            self.degree_matrix = D.pow(-0.5)  # Inverse square root of the degree matrix
        x = torch.matmul(self.degree_matrix, torch.matmul(A, x))
        return self.linear(x)

class ConGM(nn.Module):
    def __init__(self,
                 encoder='gcn',
                 input_dim=1433,  # Model configuration
                 layer_num=2,  # Number of layers in the encoder
                 hidden=128):  # Encoder hidden size
        super(ConGM, self).__init__()
        # Load components
        if encoder == 'gcn':
            Encoder = GCN
        elif encoder == 'sage-gcn':
            Encoder = GraphSAGE_GCN
        else:
            raise NotImplementedError(f'{encoder} is not implemented!')
        self.encoder = Encoder(input_dim=input_dim, layer_num=layer_num, hidden=hidden)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def bi_level_negative_sample_selection(self, Hu, Hv, alpha, T):
        # Select negative samples according to formulas (9) and (10)
        Mu, Su = Hu.mean(1), Hu.std(1)
        Mv, Sv = Hv.mean(1), Hv.std(1)

        # Compute the probability P(C = u|S = v)
        P_Cu_Sv = alpha * norm.pdf(Hv, loc=Mu, scale=Su) / (alpha * norm.pdf(Hv, loc=Mu, scale=Su) + (1 - alpha) * norm.pdf(Hv, loc=Mv, scale=Sv))

        # Choose hard negative samples: If P(C = u|S = v) is greater than the threshold T, consider nodes in Hv as hard negative samples in Gu
        hard_negatives = P_Cu_Sv > T
        return hard_negatives

    def graph_matching_loss(self, H1, H2, A1, A2, beta):
        # Construct an edge-centric graph
        B = torch.bmm(H1, H2.transpose(1, 2))
        edge_centric_A = A1.unsqueeze(1).repeat(1, A2.size(1), 1) * A2.unsqueeze(2).repeat(1, 1, A1.size(1))

        # Edge contrastive loss
        pos_edges = (edge_centric_A == 1).nonzero()[:, [2, 1, 0]]  # Reorder indices for positive edges
        neg_edges = (edge_centric_A == 0).nonzero()[:, [2, 1, 0]]  # Reorder indices for negative edges
        anchor_nodes = pos_edges[:, 0]
        edge_features = B[anchor_nodes, range(B.size(1)), pos_edges[:, 2]]

        pos_similarity = (edge_features * B[pos_edges[:, 1], range(B.size(1)), pos_edges[:, 2]]).sum(1) / (H1.size(2) ** 0.5)
        neg_similarity = torch.sum(B[anchor_nodes.unsqueeze(1), neg_edges[:, 2]] * B[neg_edges[:, 0], neg_edges[:, 1]], 2) / (H1.size(2) ** 0.5)

        edge_loss = -torch.log((pos_similarity - neg_similarity.exp()).clamp(min=1e-6) / (1 + neg_similarity.exp()))
        return edge_loss.mean()

    def edge_contrastive_loss(self, Hu, Hv, Au, Av, hard_negatives, alpha, beta):
        # Compute similarity between node pairs
        similarity_matrix = torch.matmul(Hu, Hv.transpose(1, 2))

        # Select positive samples
        I = torch.eye(similarity_matrix.size(1), device=similarity_matrix.device)
        pos_similarity = similarity_matrix * I

        # Select negative samples
        neg_samples_mask = 1 - I
        neg_similarity = similarity_matrix * neg_samples_mask

        # Apply hard negative sample selection
        inter_graph_hard_negatives = hard_negatives.unsqueeze(2).repeat(1, 1, neg_samples_mask.size(2))
        neg_similarity = neg_similarity * (1 - inter_graph_hard_negatives)

        # Calculate the loss
        pos_loss = F.logsigmoid(pos_similarity).mean()
        neg_loss = torch.sum(F.logsigmoid(-neg_similarity), dim=2).mean()

        # Combine positive and negative sample losses
        edge_loss = - (alpha * pos_loss + beta * neg_loss)
        return edge_loss

    def forward(self, view1, view2, A1, A2, beta=0.1):

        H1 = self.encoder(view1)
        H2 = self.encoder(view2)

        # Node contrastive learning loss
        node_similarity = torch.matmul(H1, H2.transpose(1, 2))
        node_loss = -torch.log((node_similarity / torch.clamp(node_similarity.sum(1), min=1e-6)).mean())

        Hu = H1  # Node features from graph G1
        Hv = H2  # Node features from graph G2

        # Construct the adjacency matrix for the edge-centric graph
        Au = Au.repeat(A2.size(0), 1, 1)  # Assuming Au and Av are some form of adjacency matrices
        Av = Av.repeat(Au.size(0), 1, 1)

        # Perform bi-level negative sampling
        hard_negatives = self.bi_level_negative_sample_selection(Hu, Hv, alpha=0.5, T=0.6)

        # Compute the edge contrastive loss
        edge_loss = self.edge_contrastive_loss(Hu, Hv, Au, Av, hard_negatives, alpha=0.5, beta=0.5)
        # Combine losses
        loss = node_loss + beta * edge_loss

        return loss


class GCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim

        self.convs = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden*2)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden*2, hidden*2))
                glorot(self.convs[i].weight)
            self.convs.append(GCNConv(hidden*2, hidden))
            glorot(self.convs[-1].weight)

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, hidden)) 
            glorot(self.convs[-1].weight)

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            x = F.relu(self.convs[i](x, edge_index))
        return x


class GraphSAGE_GCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3, hidden=512):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.layer = layer_num
        self.acts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(self.layer):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden, root_weight=True))
            else:
                self.convs.append(SAGEConv(hidden, hidden, root_weight=True))
            # self.acts.append(torch.nn.PReLU(hidden))
            self.acts.append(torch.nn.ELU())
            self.norms.append(torch.nn.BatchNorm1d(hidden))
            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.layer):
            x = self.acts[i](self.norms[i](self.convs[i](x, edge_index)))
        return x


class Projector(torch.nn.Module):
    def __init__(self, shape=()):
        super(Projector, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = torch.nn.Sequential(
            torch.nn.Linear(shape[0], shape[1]),
            torch.nn.BatchNorm1d(shape[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(shape[1], shape[2])
        )

    def forward(self, x):
        return self.main(x)


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

class LogReg(torch.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = torch.nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
