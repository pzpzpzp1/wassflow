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
from Utils import InputMapping, BoundingBox, ImageDataset, SaveTrajectory
from Utils import SaveTrajectory as st
import ODEModel
from ODEModel import ODEfunc
from ODEModel import FfjordModel

def learn_trajectory(z_target_full, my_loss, n_iters = 10, n_subsample = 100, model=FfjordModel(), save=False):
    """
        Learns a trajectory between multiple timesteps contained in z_target
        ----------
        z_target : torch.Tensor 
            Tensor of shape (T,n,d) where T is the number of timesteps
        my_loss : str
            Data fidelity loss, either 'sinkhorn_large_reg', 'sinkhorn_small_reg' or 'energy_distance'
        Returns
        -------
        model : 
            NN representing the vector field
        
        """
#     pdb.set_trace()
    z_target_full = ImageDataset.normalize_samples(z_target_full) # normalize to fit in [0,1] box.
    
    if my_loss == 'sinkhorn_large_reg':
        my_loss_f = SamplesLoss("sinkhorn", p=2, blur=0.01)
#         my_loss_f = SamplesLoss("sinkhorn", p=2, blur=0.005)
#         my_loss_f = SamplesLoss("hausdorff",p=2,blur=.05)
#         my_loss_f = SamplesLoss("energy") # Energy Distance
#         my_loss_f = SamplesLoss("energy") # Energy Distance
    elif my_loss == 'sinkhorn_small_reg':
        my_loss_f = SamplesLoss("sinkhorn", p=2, blur=1)
    else:
        my_loss_f = SamplesLoss("energy") # Energy Distance


#     model = FfjordModel(); 
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience=4,min_lr=1e-6)
    
    T = z_target_full.shape[0];

#     # get spacetime bounding box and spacetime sample grid
    BB = BoundingBox(z_target_full);
    
#     t_N = 30; x_N = 10; bbscale = 1.1;
#     t_sample = torch.linspace(1, T, t_N).to(device);
#     LL = z_target_full.min(0)[0].min(0)[0]; TR = z_target_full.max(0)[0].max(0)[0]; C = (LL+TR)/2; # lower left, top right, center
#     eLL = (LL-C)*1.1+C; eTR = (TR-C)*1.1+C; # extended bounding box.
#     xspace = torch.linspace(eLL[0],eTR[0],x_N); yspace = torch.linspace(eLL[1],eTR[1],x_N);
#     xgrid,ygrid=torch.meshgrid(xspace,yspace);
#     z_sample = torch.transpose(torch.reshape(torch.stack([xgrid,ygrid]),(2,-1)),0,1).to(device);
    
    separate_losses = np.empty((8, n_iters))
    losses = []
    start = time.time()
    print('training with %s'%my_loss)
    start0 = time.time()
    for batch in range(n_iters):
        # get spacetime bounding box and spacetime sample grid
        
        if (batch % 30 == 1):
            start = time.time()
#         if batch == 300:
#             for g in optimizer.param_groups:
#                 g['lr'] = 1e-5

        # subsample z_target_full to z_target for loss computation
        fullshape = z_target_full.shape; # [T, n_samples, d]
        z_target = torch.zeros([fullshape[0], n_subsample, fullshape[2]]).to(z_target_full)
        for i in range(fullshape[0]):
            # pdb.set_trace()
            subsample_inds = torch.randint(0, high=fullshape[1], size=[n_subsample]);
            z_target[i,:,:] = z_target_full[i,subsample_inds,:] + (torch.randn(*z_target[i,:,:].shape) * .00).to(device) # add randn noise to get full support.
        
        optimizer.zero_grad()
    
        ## FORWARD and BACKWARD fitting loss
        # integrate ODE forward in time
        cpt = time.time();
        z_t = model(z_target[0,:,:], integration_times = torch.linspace(0,T-1,T).to(device))
        fitloss = torch.tensor(0.).to(device)
        for t in range(1,T):
            fitloss += my_loss_f(z_target[t,:,:], z_t[t,:,:])
        # integrate ODE backward in time from last keyframe
        z_t_b = model(z_target[T-1,:,:], integration_times = torch.linspace(0,T-1,T).to(device), reverse=True)
        fitlossB = torch.tensor(0.).to(device)
        for t in range(1,T):
            fitlossB += my_loss_f(z_target[(T-1)-t,:,:], z_t_b[t,:,:])
        if batch==0:
            # scaling factor chosen at start to normalize fitting loss
            fitloss0 = fitloss.item(); # constant. not differentiated through
            fitlossB0 = fitlossB.item();
#             print(fitloss0, fitlossB0)
        fitloss/=fitloss0
        fitlossB/=fitlossB0
        loss = 1*fitloss + 1*fitlossB;
        separate_losses[0,batch] = fitloss
        separate_losses[1,batch] = fitlossB
        fitlosstime = time.time()-cpt
        
        # VELOCITY REGULARIZERS loss
        cpt = time.time();
        z_sample = BB.samplerandom(N = 3000, bbscale = 1.1);
        z_dots, zt_jacs = model.time_deriv_func.getJacobians(z_sample[:,0], z_sample[:,1:]);
        # divergence squared
        div2loss = (zt_jacs[:,0,0]+zt_jacs[:,1,1])**2
        # square norm of curl
        curl2loss = (zt_jacs[:,0,1]-zt_jacs[:,1,0])**2
        # rigid motion: x(t) -> e^[wt] x0 + kt. v = x_dot = [w]x0+k; dvdx = [w]. ==> skew symmetric velocity gradient is rigid.
        rigid2loss = ((zt_jacs[:,0,1]+zt_jacs[:,1,0])**2)/2 + (zt_jacs[:,0,0])**2 + (zt_jacs[:,1,1])**2 
        # v-field gradient loss
        vgradloss = zt_jacs[:,0,0]**2 + zt_jacs[:,1,1]**2+zt_jacs[:,0,1]**2 + zt_jacs[:,1,0]**2
        # kinetic energy loss
        KEloss = z_dots[:,0]**2 + z_dots[:,1]**2
        
        separate_losses[2,batch] = div2loss.mean().item()
        separate_losses[3,batch] = curl2loss.mean().item()
        separate_losses[4,batch] = rigid2loss.mean().item()
        separate_losses[5,batch] = vgradloss.mean().item()
        separate_losses[6,batch] = KEloss.mean().item()
        
#         timeIndices = (z_sample[:,0] < ((T-1.)/5.0)).detach()
        timeIndices = (z_sample[:,0] < ((T-1.)/.001)).detach()
        
        # combine energies
        regloss = 0*div2loss.mean() \
                + 0*.005*curl2loss.mean() \
                + 0*rigid2loss.mean() \
                + 0*vgradloss.mean() \
                + 0*KEloss.mean() 
#         - 1*torch.clamp(vgradloss.mean(), max = 10**10)  # make high noise velocity field
#         - 1*torch.clamp(curl2loss[timeIndices].mean(), max = 10**3)  # time negative time-truncated curl energy
        reglosstime = time.time()-cpt
        
        totalloss = loss + regloss
        losses.append(totalloss.item())
        
        totalloss.backward()
        optimizer.step()
        if (batch % 30 == 0):
            for g in optimizer.param_groups:
                print(g['lr'])
            scheduler.step(totalloss.item()) # timestep schedule.
            
            print('batch',batch,'loss',loss)
            plt.scatter(z_target.cpu().detach().numpy()[0,:,0], z_target.cpu().detach().numpy()[0,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
            for t in range(1,T):
                plt.scatter(z_t_b.cpu().detach().numpy()[t,:,0], z_t_b.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='red', edgecolors='black')
                plt.scatter(z_t.cpu().detach().numpy()[t,:,0], z_t.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
                plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
            plt.axis('equal')
            plt.show()
            
            ptime = time.time()-start
            print('fit time ',fitlosstime,' reg loss time',reglosstime)
            print('time elapsed',ptime,'total time',time.time()-start0)
            print('batch number',batch,'out of',n_iters)
            
            if save:
                model.save_state(fn='models/state_' + f"{batch:04}" + '_time_' + str(ptime) + '_' + str(losses[batch]) + '.tar')
                st.save_trajectory(model,z_target,my_loss + "_" + f"{batch:04}", savedir='imgs', nsteps=20, memory=0.01, n=1000)
                st.save_trajectory(model,z_target,my_loss + "_" + f"{batch:04}", savedir='imgs', nsteps=20, memory=0.01, n=1000, reverse = True)
#             ## check garbage collected tensor list for increase in tensor sizes or number of objects.
#             cc = 0;
#             for obj in gc.get_objects():
#                 try:
#                     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                         cc+=1;
#                         print(type(obj), obj.size(), obj.grad_fn)
#                 except:
#                     pass
#             print('nobjs ', cc);

        fitloss.detach();
        fitlossB.detach();
        z_t.detach();
        z_t_b.detach();
        loss.detach();
        totalloss.detach();
        del loss;
        torch.cuda.empty_cache()
    return model, losses, separate_losses

