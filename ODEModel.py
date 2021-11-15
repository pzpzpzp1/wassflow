import math
import numpy as np
from IPython.display import clear_output
import pdb
import time

import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions import MultivariateNormal
use_cuda = torch.cuda.is_available()
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import gc
import importlib
import Utils
# importlib.reload(Utils)
from Utils import InputMapping, BoundingBox, ImageDataset, SaveTrajectory, Sine
from Utils import SaveTrajectory as st

class ODEfunc(nn.Module):
    """
    Calculates time derivatives.

    torchdiffeq requires this to be a torch.nn.Module.
    """
    
    def __init__(self):
        super(ODEfunc, self).__init__()
        # Define network layers.
#         n_freq = 100; sigmac = 20; # frequencies to sample spacetime in.
#         n_freq = 256; sigmac = 15; # frequencies to sample spacetime in. works for 2 frames
# #         n_freq = 70; sigmac = 3; # frequencies to sample spacetime in.
        
#         Z_DIM = 2; # dimension of vector field.
#         imap = InputMapping(Z_DIM+1, n_freq, sigma=sigmac);
#         self.imap = imap; # save for sigma params
# #         pdb.set_trace()
#         N = 512
#         self.f = nn.Sequential(imap,
#                        nn.Linear(imap.d_out, N),
#                        nn.Tanh(),
#                        nn.Linear(N, N),
#                        nn.Tanh(),
#                        nn.Linear(N, N),
#                        nn.Softplus(),
#                        nn.Linear(N, Z_DIM));
#         self.f = nn.Sequential(nn.Linear(Z_DIM+1, Z_DIM));

#     def get_z_dot(self, t, z):
#         """z_dot is parameterized by a NN: z_dot = NN(t, z(t))"""
#         if t.dim()==0:
#             t = t.expand(z.shape[0],1);
#         else:
#             t = t.reshape(z.shape[0],1);
#         tz = torch.cat((t,z),1);
# #         pdb.set_trace()
#         z_dot = self.f(tz)
# #         z_dot = torch.clamp(self.f(z), max = 1, min=-1) 
#         return z_dot
    
#     def __init__(self, hidden_dims=(64,64)):
#         super(ODEfunc, self).__init__()
#         # Define network layers.
#         n_freq = 15; # frequencies to sample spacetime in.
#         Z_DIM = 2; # dimension of vector field.
#         imap = InputMapping(3, n_freq);
#         dim_list = [imap.d_out] + list(hidden_dims) + [Z_DIM]
#         layers = []
#         layers.append(imap);
#         for i in range(len(dim_list)-1):
#             layers.append(nn.Linear(dim_list[i]+1, dim_list[i+1]))
#         self.layers = nn.ModuleList(layers)

#     def get_z_dot(self, t, z):
#         # pdb.set_trace()
#         """z_dot is parameterized by a NN: z_dot = NN(t, z(t))"""
#         z_dot = z;
#         for l, layer in enumerate(self.layers):
#             # Concatenate t at each layer.
#             tz_cat = torch.cat((t.expand(z.shape[0],1), z_dot), dim=1)
#             z_dot = layer(tz_cat) #add time t into first spot
#             # pdb.set_trace()
#             if l < len(self.layers) - 1 and not l == 0:
#                 #pdb.set_trace()
#                 z_dot = F.softplus(z_dot)
#         return z_dot

    # d z_dot d z. assuming zdot was computed from z. otherwise output is just 0.
#     def getJacobians(self, t, z):
#         batchsize = z.shape[0]

#         with torch.set_grad_enabled(True):            
#             z.requires_grad_(True)
#             t.requires_grad_(True)
#             z_dot = self.get_z_dot(t, z)
            
#             # compute jacobian of velocity field. [N,2,2]
#             # inputs z_dot.sum() because each z_dot only depends on one z. no cross derivatives. this batches the grad.
#             dim = z.shape[1];
#             jacobians = torch.zeros([batchsize,dim,dim], dtype=z.dtype, device=z.device);
#             for i in range(z.shape[1]):
#                  jacobians[:,i,:] = torch.autograd.grad( z_dot[:, i].sum(), z, create_graph=True)[0]
#         return z_dot, jacobians
    
#     def forward(self, state):
#         """
#         Calculate the time derivative of z and divergence.

#         Parameters
#         ----------
#         t : torch.Tensor
#             time
#         state : torch.Tensor
#             Contains z

#         Returns
#         -------
#         z_dot : torch.Tensor
#             Time derivative of z.
#         negative_divergence : torch.Tensor
#             Time derivative of the log determinant of the Jacobian.
#         """
# #         pdb.set_trace()
        
#         zt = state
#         batchsize = zt.shape[0]

#         with torch.set_grad_enabled(True):
#             zt.requires_grad_(True)
#             z_dot = self.f(zt)
            
#         return z_dot

class FfjordModel(torch.nn.Module):
    """Continuous noramlizing flow model."""

    def __init__(self):
        super(FfjordModel, self).__init__()
        # Define network layers.
        n_freq = 256; sigmac = 15; # frequencies to sample spacetime in.
#         n_freq = 50; sigmac = 4; # frequencies to sample spacetime in.
#         n_freq = 70; sigmac = 3; # frequencies to sample spacetime in.
        Z_DIM = 2; # dimension of vector field.
        imap = InputMapping(Z_DIM+1, n_freq, sigma=sigmac);
        self.imap = imap; # save for sigma params
#         pdb.set_trace()
        N = 512
        self.f = nn.Sequential(imap,
                       nn.Linear(imap.d_out, N),
                       nn.Tanh(),
                       nn.Linear(N, N),
                       nn.Tanh(),
                       nn.Linear(N, N),
                       nn.Softplus(),
                       nn.Linear(N, Z_DIM));

    def save_state(self, fn='state.tar'):
        """Save model state."""
        torch.save(self.state_dict(), fn)

    def load_state(self, fn='state.tar'):
        """Load model state."""
        self.load_state_dict(torch.load(fn))
    
    def z_t_to_zt(self, z, t):
        """
        z: N d
        t: T
        zz: (TN) d
        tt: (TN) 1
        zt: (TN) (d+1)
        """
        zz = torch.tile(z,(t.shape[0],1))
        tt = t.repeat_interleave(z.shape[0]).reshape((-1,1))
        zt = torch.cat((zz,tt),dim=1)
        
#         pdb.set_trace()
        return zt
    
    def getGrads(self, zt):
        """
        zt: N (d+1)
        out: N d
        jacs: 
        """
        out = self.forward_zt(zt)
        d = zt.shape[1]-1;
        batchsize = zt.shape[0];
        
        jacobians = torch.zeros([batchsize, d, d+1], dtype=zt.dtype, device=zt.device);
        for i in range(d):
            jacobians[:,i,:] = torch.autograd.grad( out[:, i].sum(), zt, create_graph=True)[0]
        return out, jacobians
        
    def forward_zt(self, zt):
        """
        zt: N (d+1)
        out: N d
        """
        with torch.set_grad_enabled(True):
            zt.requires_grad_(True)
        nnout = self.f(zt)
        return nnout
    
    def forward(self, z, integration_times=None):
        """
        Implementation of Eq. 4.
        We want to integrate both f and the trace term. During training, we
        integrate from t_1 (data distribution) to t_0 (base distibution).
        Parameters
        ----------
        z : torch.Tensor
            Samples.
        integration_times : torch.Tensor
            Which times to evaluate at.
        reverse : bool, optional
            Whether to reverse the integration times.
        Returns
        -------
        z : torch.Tensor
            Updated samples.
        """
#         zz = torch.tile(torch.transpose(z,0,1),(1,integration_times.shape[0]))
#         tt = integration_times.repeat_interleave(z.shape[0]).reshape((1,-1))
#         zt = torch.vstack((zz,tt))
        zt = self.z_t_to_zt(z, integration_times)
#         pdb.set_trace()
        d = z.shape[1]
        T = integration_times.shape[0]
        
        with torch.set_grad_enabled(True):
            zt.requires_grad_(True)
        nnout = self.f(zt)
        outvals = nnout.reshape((T,-1,d))
#         pdb.set_trace()

        return outvals
    
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long,             device=x.device)
    return x[tuple(indices)]
