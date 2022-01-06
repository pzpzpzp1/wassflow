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
from Utils import InputMapping, BoundingBox, ImageDataset, SaveTrajectory, Sine, MiscTransforms
from Utils import SaveTrajectory as st

class velocMLP(nn.Module):
    """
    Calculates time derivatives.

    torchdiffeq requires this to be a torch.nn.Module.
    """
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2, out_features=2, sigmac = 3, n_freq = 70, tdiv = 1):
        super(velocMLP, self).__init__()
        self.imap, self.f = coordMLP.mlp(in_features, hidden_features, hidden_layers, out_features, sigmac, n_freq, tdiv)
        
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

#     d z_dot d z. assuming zdot was computed from z. otherwise output is just 0.

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
    def getGrads(self, tz):
        """
        tz: N (d+1)
        out: N d
        jacs: 
        """
        tz.requires_grad = True
        dv = tz.shape[1]-1;
        batchsize = tz.shape[0];
        z = tz[:,1:]
        t = tz[:,0:1]
        out = self.get_z_dot(t, z)
        
        jacobians = torch.zeros([batchsize, dv, dv+1], dtype=tz.dtype, device=tz.device);
        for i in range(dv):
            jacobians[:,i,:] = torch.autograd.grad( out[:, i].sum(), tz, create_graph=True)[0]
        
        return out, jacobians[:,:,0:dv], jacobians[:,:,dv:]
    
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
#         pdb.set_trace()
        
#         zt = state
#         batchsize = zt.shape[0]

#         with torch.set_grad_enabled(True):
#             zt.requires_grad_(True)
#             z_dot = self.f(zt)
            
#         return z_dot
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
    
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2, out_features=2, sigmac = 3, n_freq = 70, tdiv = 1):
        super(FfjordModel, self).__init__()
        self.modelshape = {'in_features': in_features, \
                           'hidden_features': hidden_features,\
                           'hidden_layers': hidden_layers,\
                           'out_features': out_features,\
                           'n_freq': n_freq,\
                          }
        self.velfunc = velocMLP(in_features, hidden_features, hidden_layers, out_features, sigmac, n_freq, tdiv)
    def save_state(self, fn='state.tar'):
        selfdict = self.state_dict()
        selfdict['modelshape'] = self.modelshape
        torch.save(selfdict, fn)
    def load_state(self, fn='state.tar'):
        self_dict = torch.load(fn)
        ms = self_dict.pop('modelshape')
        self.velfunc = velocMLP(ms['in_features'], ms['hidden_features'], ms['hidden_layers'], ms['out_features'], 5, ms['n_freq'], 5)
        
        # pdb.set_trace()
        self.load_state_dict(self_dict)
        self=self.to(device)
        
#     def getGrads(self, zt):
#         """
#         zt: N (d+1)
#         out: N d
#         jacs: 
#         """
#         out = self.forward_zt(zt)
#         dv = zt.shape[1]-1;
#         batchsize = zt.shape[0];
        
#         jacobians = torch.zeros([batchsize, dv, dv+1], dtype=zt.dtype, device=zt.device);
#         for i in range(dv):
#             jacobians[:,i,:] = torch.autograd.grad( out[:, i].sum(), zt, create_graph=True)[0]
            
#         acceleration = torch.zeros([batchsize, dv], dtype=zt.dtype, device=zt.device);
#         for i in range(dv):
#             tempdev = torch.autograd.grad(jacobians[:, i, dv].sum(), zt, create_graph=True)[0];
#             acceleration[:,i] = tempdev[:,dv]
            
#         return out, jacobians, acceleration
    def forward(self, z, integration_times=None, reverse=False):
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        state = odeint(
            self.velfunc, # Calculates time derivatives.
            z, # Values to update.
            integration_times, # When to evaluate.
            method='dopri5', # Runge-Kutta
            atol=1e-5, # Error tolerance
            rtol=2e-5, # Error tolerance
        )
        return state
    
def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long,             device=x.device)
    return x[tuple(indices)]

#     def forward_zt(self, zt):
#         """
#         zt: N (d+1)
#         out: N d
#         """
#         with torch.set_grad_enabled(True):
#             zt.requires_grad_(True)
#         nnout = self.f(zt)
#         return nnout
    
#     def forward(self, z, integration_times=None):
#         zt = MiscTransforms.z_t_to_zt(z, integration_times)
#         d = z.shape[1]
#         T = integration_times.shape[0]
        
#         with torch.set_grad_enabled(True):
#             zt.requires_grad_(True)
#         nnout = self.f(zt)
#         outvals = nnout.reshape((T,-1,d))
        
#         return outvals


class coordMLP(nn.Module):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2, out_features=2, sigmac = 10, n_freq = 256, tdiv = 6):
        super().__init__()
        
        # net1 is forward, net2 is backward
        self.imap1, self.net1 = coordMLP.mlp(in_features, hidden_features, hidden_layers, out_features, sigmac, n_freq, tdiv)
        self.imap2, self.net2 = coordMLP.mlp(in_features, hidden_features, hidden_layers, out_features, sigmac, n_freq, tdiv)
            
    def mlp(in_features=3, hidden_features=512, hidden_layers=2, out_features=2, sigmac = 10, n_freq = 256, tdiv = 6):
        imap = InputMapping(in_features, n_freq, sigma=sigmac, tdiv = tdiv)
        net = []
        net.append(imap)
        net.append(nn.Linear(imap.d_out, hidden_features))
        for i in range(hidden_layers):
            net.append(nn.Tanh())
            net.append(nn.Linear(hidden_features, hidden_features))
        net.append(nn.Softplus())
        net.append(nn.Linear(hidden_features, out_features));
        net = nn.Sequential(*net)
        return imap, net
    
    def save_state(self, fn='state.tar'):
        torch.save(self.state_dict(), fn)
    def load_state(self, fn='state.tar'):
        self.load_state_dict(torch.load(fn))
        
    def showmap(self, t=0, bound=1.1,N=40, ti=1,ax=plt):
        dx = 2*bound/(N-1)
        xvals = torch.linspace(-bound,bound,N)
        X, Y = torch.meshgrid(xvals, xvals,indexing='ij')
        Xc = X[:-1,:-1] + dx/2
        Yc = Y[:-1,:-1] + dx/2
        z = torch.sqrt((Xc-Xc.round())**2 + (Yc-Yc.round())**2)

        tt = torch.tensor(t);
        XYi = torch.cat((X.reshape((-1,1)), Y.reshape((-1,1)), tt.repeat(N**2,1)),dim=1);
        (XYo,blah) = self.forward(XYi.to(device))
        
        Xo = XYo[:,0].reshape((N,N)).detach().cpu().numpy()
        Yo = XYo[:,1].reshape((N,N)).detach().cpu().numpy()
        Xt = X*(1-ti)+Xo*ti
        Yt = Y*(1-ti)+Yo*ti
        
        ax.pcolormesh(Xt, Yt, z, edgecolors = 'none', alpha=.5, cmap='inferno')
        plt.xlim((-bound,bound))
        plt.ylim((-bound,bound))
        ax.axis('equal')
    def getGrads(self, zt):
        """
        zt: N (d+1)
        out: N d
        jacs: 
        """
        zt.requires_grad = True
        dv = zt.shape[1]-1;
        batchsize = zt.shape[0];
        z = zt[:,0:dv]
        t = zt[:,dv:]
        (out, throwaway) = self.forward(torch.cat((z,t),1))
        
        jacobians = torch.zeros([batchsize, dv, dv+1], dtype=zt.dtype, device=zt.device);
        for i in range(dv):
            jacobians[:,i,:] = torch.autograd.grad( out[:, i].sum(), zt, create_graph=True)[0]
        
        acceleration = torch.zeros([batchsize, dv], dtype=zt.dtype, device=zt.device);
        for i in range(dv):
            acceleration[:,i] = torch.autograd.grad(jacobians[:, i, dv].sum(), t, create_graph=True)[0].reshape(-1);
        
        return out, jacobians, acceleration
    
    def forward(self, coords):
        coords = coords.requires_grad_(True) # allows to take derivative w.r.t. input
        disp = self.net1(coords)
        
        dispT = torch.mul(coords[:,-1:], disp)
        output = coords[:,0:2] + dispT; # learn displacement. guarantees t=0 frame is identity.
        
        return output, coords
    def backward(self, coords):
        coords = coords.requires_grad_(True) # allows to take derivative w.r.t. input
        disp = self.net2(coords)
        
        dispT = torch.mul(coords[:,-1:], disp)
        output = coords[:,0:2] + dispT; # learn displacement. guarantees t=0 frame is identity.
        
        return output, coords

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
