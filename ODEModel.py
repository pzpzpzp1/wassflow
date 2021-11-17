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
        zt = MiscTransforms.z_t_to_zt(z, integration_times)
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

class Siren(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=2, out_features=2, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    def showmap(self, t=0, bound=1.1,N=40, ti=.5):
        dx = 2*bound/(N-1)
        xvals = torch.linspace(-bound,bound,N)
        X, Y = torch.meshgrid(xvals, xvals)
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
        plt.pcolormesh(Xt, Yt, z, edgecolors = 'none', alpha=.5, cmap='inferno')
    
    def getGrads(self, zt):
        """
        zt: N (d+1)
        out: N d
        jacs: 
        """
        (out, throwaway) = self.forward(zt)
        d = zt.shape[1]-1;
        batchsize = zt.shape[0];
        
        jacobians = torch.zeros([batchsize, d, d+1], dtype=zt.dtype, device=zt.device);
        for i in range(d):
            jacobians[:,i,:] = torch.autograd.grad( out[:, i].sum(), zt, create_graph=True)[0]
        return out, jacobians
    
    def forward(self, coords):
#         coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        coords = coords.requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=True):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
#         activations = OrderedDict()

#         activation_count = 0
#         x = coords.requires_grad_(True)
#         activations['input'] = x
#         for i, layer in enumerate(self.net):
#             if isinstance(layer, SineLayer):
#                 x, intermed = layer.forward_with_intermediate(x)
                
#                 if retain_grad:
#                     x.retain_grad()
#                     intermed.retain_grad()
                    
#                 activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
#                 activation_count += 1
#             else: 
#                 x = layer(x)
                
#                 if retain_grad:
#                     x.retain_grad()
                    
#             activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
#             activation_count += 1

#         return activations

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
    
    
