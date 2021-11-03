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
        n_freq = 70; sigmac = 4; # frequencies to sample spacetime in.
        
        Z_DIM = 2; # dimension of vector field.
        imap = InputMapping(Z_DIM+1, n_freq, sigma=sigmac);
        self.imap = imap; # save for sigma params
#         pdb.set_trace()
        self.f = nn.Sequential(imap,
                       nn.Linear(imap.d_out, 512),
                       nn.Tanh(),
                       nn.Linear(512, 512),
                       nn.Tanh(),
                       nn.Linear(512, 512),
                       nn.Softplus(),
                       nn.Linear(512, Z_DIM));

    def get_z_dot(self, t, z):
        """z_dot is parameterized by a NN: z_dot = NN(t, z(t))"""
        if t.dim()==0:
            t = t.expand(z.shape[0],1);
        else:
            t = t.reshape(z.shape[0],1);
        tz = torch.cat((t,z),1);
#         pdb.set_trace()
        z_dot = self.f(tz)
#         z_dot = torch.clamp(self.f(z), max = 1, min=-1) 
        return z_dot
    
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
    def getJacobians(self, t, z):
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):            
            z.requires_grad_(True)
            t.requires_grad_(True)
            z_dot = self.get_z_dot(t, z)
            
            # compute jacobian of velocity field. [N,2,2]
            # inputs z_dot.sum() because each z_dot only depends on one z. no cross derivatives. this batches the grad.
            dim = z.shape[1];
            jacobians = torch.zeros([batchsize,dim,dim], dtype=z.dtype, device=z.device);
            for i in range(z.shape[1]):
                 jacobians[:,i,:] = torch.autograd.grad( z_dot[:, i].sum(), z, create_graph=True)[0]
        return z_dot, jacobians
    
    def forward(self, t, state):
        """
        Calculate the time derivative of z and divergence.

        Parameters
        ----------
        t : torch.Tensor
            time
        state : torch.Tensor
            Contains z

        Returns
        -------
        z_dot : torch.Tensor
            Time derivative of z.
        negative_divergence : torch.Tensor
            Time derivative of the log determinant of the Jacobian.
        """
        z = state
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            # Calculate the time derivative of z. 
            # This is f(z(t), t; \theta) in Eq. 4.
            z_dot = self.get_z_dot(t, z)
            
        return z_dot

class FfjordModel(torch.nn.Module):
    """Continuous noramlizing flow model."""

    def __init__(self):
        super(FfjordModel, self).__init__()
        self.time_deriv_func = ODEfunc()

    def save_state(self, fn='state.tar'):
        """Save model state."""
        torch.save(self.state_dict(), fn)

    def load_state(self, fn='state.tar'):
        """Load model state."""
        self.load_state_dict(torch.load(fn))


    def forward(self, z, integration_times=None, reverse=False):
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
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        #print('integration_times',integration_times)
        # Integrate. This is the call to torchdiffeq.
        
        state = odeint(
            self.time_deriv_func, # Calculates time derivatives.
            z, # Values to update.
            integration_times, # When to evaluate.
            method='dopri5', # Runge-Kutta
            atol=1e-5, # Error tolerance
            rtol=2e-5, # Error tolerance
        )
        
        z = state
        return z
    
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long,             device=x.device)
    return x[tuple(indices)]
