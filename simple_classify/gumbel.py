import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn 
from torch.nn import init

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature,beta=1.0):
    y = logits + beta*sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temp, hard=False,beta=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temp,beta)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    if torch.cuda.is_available():
        y_hard = y_hard.cuda()
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    if y_hard.max() != y_hard.max(): import pdb;pdb.set_trace()
    if y.max() != y.max(): import pdb;pdb.set_trace()
    assert y_hard.dim() ==2
    if hard:
        return (y_hard - y).detach() + y
    else:
        return y
