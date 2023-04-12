import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from simple_classify.gumbel import gumbel_softmax
import math
import numpy as np

class Model(nn.Module):
    def __init__(self, n_nodes, n_parts):

        super(Model, self).__init__()
        self.input_dim = n_nodes
        self.num_parts = n_parts
        # embedđings
        self.params = nn.Parameter(init.xavier_normal_(torch.Tensor(self.input_dim, self.num_parts)), requires_grad=True)

    def forward(self, adj, nodes, temp=10, hard=False, beta=1):
        self.adj = adj
        if nodes is not None:
            adj = adj[np.ix_(nodes, nodes)]
            mask = np.zeros((self.adj.shape[0],))
            mask[nodes]=1
            mask=torch.ByteTensor(mask).cuda()
            self.assign_tensor = gumbel_softmax(self.params[mask], temp=temp, hard=hard, beta=beta)
        else:
            self.assign_tensor = gumbel_softmax(self.params, temp=temp, hard=hard, beta=beta)
        #self.assign_tensor = F.gumbel_softmax(self.params[nodes], tau=temp, hard=hard)
        self.assign_tensor_t = torch.transpose(self.assign_tensor, 0, 1)

        super_adj = self.assign_tensor_t @ adj @ self.assign_tensor # A' = S^T*A*S
        return super_adj

    def sigmoid_cross_entropy_with_logits(self, logits, labels):
        sig_aff = torch.sigmoid(logits)
        loss = torch.sum(labels * -torch.log(sig_aff+1e-20) + (1 - labels) * -torch.log(1 - sig_aff+1e-20))
        return loss

    def loss(self, super_adj, nodes, balance_node=True, lam = 0.7, w2v_lam = 0.01, new=False):
        if not new:
            ncut_loss = torch.sum(torch.tril(super_adj, diagonal=-1) + torch.triu(super_adj, diagonal=1))
        else:
            ncut_loss = self.ncut(super_adj)
        if balance_node:
            balance_loss = torch.sum((torch.sum(self.assign_tensor, dim=0) - self.input_dim//self.num_parts)**2)
        else:
            balance_loss = torch.sum((torch.diagonal(super_adj) - torch.sum(torch.diagonal(super_adj))//self.num_parts)**2)
        
        balance_loss = torch.sqrt(balance_loss)
        loss = lam * ncut_loss + (1-lam) * balance_loss
        # loss = lam*ncut_loss + (1-lam)*torch.sqrt(balance_loss)

        if w2v_lam > 0:
            embedding =  self.params[nodes]/self.params[nodes].norm(p=2, dim=1, keepdim=True)
            embed_pairwise = torch.matmul(embedding, torch.transpose(embedding,0,1))
            embed_pairwise = embed_pairwise.view(-1)
            labels = self.adj[np.ix_(nodes, nodes)].view(-1)
            node2vec_loss = self.sigmoid_cross_entropy_with_logits(embed_pairwise, labels)
            loss = (1-w2v_lam)*loss + w2v_lam*node2vec_loss

        return loss, ncut_loss, balance_loss

    def n2v_loss(self, nodes1, nodes2, neg_nodes, adj):
        vec1 = self.params[nodes1]
        vec2 = self.params[nodes2]
        neg_vec = self.params[neg_nodes]

        vec1 = F.normalize(vec1, dim=1)
        vec2 = F.normalize(vec2, dim=1)
        neg_vec = F.normalize(neg_vec, dim=1)

        true_aff = F.cosine_similarity(vec1, vec2)
        neg_aff = vec1.mm(neg_vec.t())
        true_labels = torch.ones(true_aff.shape)
        if torch.cuda.is_available():
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if torch.cuda.is_available():
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        neg_xent = neg_xent * (1-adj[np.ix_(nodes1, neg_nodes)])
        loss = true_xent.sum() + neg_xent.sum()

        return loss

    def ncut(self, super_adj):
        vol = super_adj.sum(1)
        diag = torch.diagonal(super_adj)
        norm_cut = (vol - diag)/(vol+1e-20)
        lozz = norm_cut.sum()
        return lozz

class HierachyModel(nn.Module):
    def __init__(self,n_nodes,n_parts1, n_parts2):
        super(HierachyModel, self).__init__()
        self.input_dim = n_nodes
        self.num_parts1 = n_parts1
        self.num_parts2 = n_parts2

        self.params1 = nn.Parameter(init.xavier_normal_(torch.Tensor(self.input_dim, self.num_parts1)), requires_grad=True)
        self.params2 = nn.Parameter(init.xavier_normal_(torch.Tensor(self.num_parts1, self.num_parts2)), requires_grad=True)

    def forward(self, adj, temp, hard, beta=1):
        self.assign_tensor1 = gumbel_softmax(self.params1, temp=temp, hard=hard, beta=beta)
        self.assign_tensor1_t = torch.transpose(self.assign_tensor1, 0, 1)
        self.super_adj1 = self.assign_tensor1_t @ adj @ self.assign_tensor1

        self.assign_tensor2 = gumbel_softmax(self.params2, temp=temp, hard=hard, beta=beta)
        self.assign_tensor2_t = torch.transpose(self.assign_tensor2, 0, 1)
        self.super_adj2 = self.assign_tensor2_t @ self.super_adj1 @ self.assign_tensor2
        
    def loss(self, lam=0.7):
        ncut_loss1 = self.ncut(self.super_adj1)
        # ncut_loss = torch.sum(torch.tril(self.super_adj1, diagonal=-1) + torch.triu(self.super_adj1, diagonal=1))
        # balance_loss = torch.sum((torch.sum(self.assign_tensor1, dim=0) - self.input_dim//self.num_parts1)**2)
        # ncut_loss2 = ncut_loss + balance_loss

        # ncut_loss2 = self.ncut(self.super_adj2)
        ncut_loss = torch.sum(torch.tril(self.super_adj2, diagonal=-1) + torch.triu(self.super_adj2, diagonal=1))
        balance_loss = torch.sum((torch.sum(self.assign_tensor2, dim=0) - self.num_parts1//self.num_parts2)**2)
        ncut_loss2 = ncut_loss + balance_loss

        loss = lam*ncut_loss1 + (1-lam)*ncut_loss2
        return loss

    def ncut(self, super_adj):
        vol = super_adj.sum(1)
        diag = torch.diagonal(super_adj)
        norm_cut = (vol - diag)/(vol+1e-20)
        lozz = norm_cut.sum()
        return lozz

class DeepModel(nn.Module):
    def __init__(self, n_nodes, embedding_size, n_parts):
        super(DeepModel, self).__init__()
        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.n_parts = n_parts
        self.encoder = nn.Sequential(
            nn.Linear(self.n_nodes, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # nn.Dropout(dropout),
            nn.Linear(512, self.embedding_size),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_size),
            nn.Linear(self.embedding_size, self.n_parts)
        )

        self.CELoss = nn.CrossEntropyLoss()

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, temp=10, hard=False, beta=1, return_logits=True):
        self.params = self.encoder(X)
        if return_logits:
            self.assign_tensor = gumbel_softmax(self.params, temp=temp, hard=hard, beta=beta)
            return self.assign_tensor
        else:
            self.softmax = self.classifier(self.params)
            self.softmax = gumbel_softmax(self.params,temp=temp, hard=hard, beta=beta )
            return self.softmax

    def unsup_loss(self, assign_tensor, adj):
        super_adj = torch.transpose(assign_tensor, 0, 1) @ adj @ assign_tensor
        vol = super_adj.sum(1)
        diag = torch.diagonal(super_adj)
        norm_cut = (vol - diag)/vol
        lozz = norm_cut.sum()
        return lozz

    def sup_loss(self, assign_tensor, labels):
        return self.CELoss(assign_tensor, labels)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, A, bias=True):
        super(GraphConvolution, self).__init__()
        n = A.shape[0]
        assert A.shape[1] == n, "A must be a square matrix"

        I = torch.eye(n)
        if torch.cuda.is_available():
            I = I.cuda()
        A_ = A + I
        D_ = torch.diag(torch.sum(A_, 0)**(-0.5))
        self.A_hat = torch.matmul(torch.matmul(D_,A_),D_)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(self.A_hat , torch.mm(input, self.weight))
        output = support #torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, A, n_nodes, embedding_size, n_parts):
        super(GCN, self).__init__()
        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.n_parts = n_parts
        self.encoder = nn.Sequential(
            GraphConvolution(self.n_nodes, self.embedding_size, A),
            # nn.Tanh(),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            # GraphConvolution(512, embedding_size, A)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            # nn.BatchNorm1d(self.embedding_size),
            GraphConvolution(self.embedding_size, self.n_nodes, A),
        )

        self.CELoss = nn.CrossEntropyLoss()

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, temp=10, hard=False, beta=1):
        self.params = self.encoder(X)
        self.assign_tensor = gumbel_softmax(self.params, temp=temp, hard=hard, beta=beta)
        self.softmax = self.classifier(self.params)
        self.softmax = F.softmax(self.softmax,-1) #gumbel_softmax(self.softmax,temp=temp, hard=hard, beta=beta )
        return self.assign_tensor, self.softmax

    def unsup_loss(self, assign_tensor, adj):
        super_adj = torch.transpose(assign_tensor, 0, 1) @ adj @ assign_tensor
        vol = super_adj.sum(1)
        diag = torch.diagonal(super_adj)
        norm_cut = (vol - diag)/vol
        lozz = norm_cut.sum()
        return lozz

    def sup_loss(self, assign_tensor, labels):
        return self.CELoss(assign_tensor, labels)


"""


"""
