import math
import numpy as np
from IPython.display import clear_output
import pdb
import time

import matplotlib.pyplot as plt

import torch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn  import functional as F 
from torch.autograd import Variable
#from torchdiffeq import odeint_adjoint as odeint
#from torch.distributions import MultivariateNormal
use_cuda = torch.cuda.is_available()
#from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import gc
import os

class ImageDataset():
    #"""Sample from a distribution defined by an image."""
    def __init__(self, img, MAX_VAL=.5, thresh=0):
        img[img<thresh]=0; # threshold to cut empty region of image
        h, w = img.shape
        xx = np.linspace(-MAX_VAL, MAX_VAL, w)
        yy = np.linspace(-MAX_VAL, MAX_VAL, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        self.means = np.concatenate([xx, yy], 1)
        self.probs = img.reshape(-1); 
        self.probs /= self.probs.sum();
#         self.noise_std = np.array([MAX_VAL/w, MAX_VAL/h])
        self.noise_std = np.array([.01,.01])*0; # add the noise in training for an effective image blur.
#         print(self.noise_std)

    def sample(self, batch_size=512):
        inds = np.random.choice(int(self.probs.shape[0]), int(batch_size), p=self.probs)
        m = self.means[inds]
        samples = np.random.randn(*m.shape) * self.noise_std + m
        return torch.from_numpy(samples).type(torch.FloatTensor)
    
    def import_img(file):
#         """
#         file : str
#             filename for an rgba image
#         Returns
#         gimg : 2D array
#             greyscale image
#         """
        img = plt.imread(file)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        gimg = np.dot(img[...,:3], rgb_weights)
        return gimg

    def make_image(n=10000):
#         """Make an X shape."""
        points = np.zeros((n,2))
        points[:n//2,0] = np.linspace(-1,1,n//2)
        points[:n//2,1] = np.linspace(1,-1,n//2)
        points[n//2:,0] = np.linspace(1,-1,n//2)
        points[n//2:,1] = np.linspace(1,-1,n//2)
        np.random.seed(42)
        noise = np.clip(np.random.normal(scale=0.1, size=points.shape),-0.2,0.2)
        np.random.seed(None)
        points += noise
        img, _ = np.histogramdd(points, bins=40, range=[[-1.5,1.5],[-1.5,1.5]])
        return img

    def normalize_samples(z_target):
        ## normalize a [K,N,D] tensor. K is number of frames. N is number of samples. D is dimension. Fit into [0,1] box without changing aspect ratio.
        dim = z_target.shape[2]
        z_target -= z_target.reshape(-1,dim).min(0)[0]
        z_target /= z_target.reshape(-1,dim).max()
        return z_target

class BoundingBox():
    ## use like:
    # BB = BoundingBox(z_target);
    # smps = BB.sampleuniform(t_N = 30, x_N = 10, y_N = 11, z_N=12, bbscale = 1.1);
    # smps = BB.samplerandom(N = 10000, bbscale = 1.1);
    
    def __init__(self, z_target_full):
        self.T = z_target_full.shape[0]; 
        self.dim = z_target_full.shape[2];
        
        # min corner, max corner, center
        self.mic = z_target_full.reshape(-1,self.dim).min(0)[0];
        self.mac = z_target_full.reshape(-1,self.dim).max(0)[0]; 
        self.C = (self.mic+self.mac)/2; 
        
    def extendedBB(self, bbscale):
        # extended bounding box.
        emic = (self.mic-self.C)*bbscale+self.C; 
        emac = (self.mac-self.C)*bbscale+self.C; 
        return emic, emac;
        
    def sampleuniform(self, t_N = 30, x_N = 10, y_N = 11, z_N = 12, bbscale = 1.1):
        [eLL,eTR] = self.extendedBB(bbscale);
        
        tspace = torch.linspace(0, self.T-1, t_N);
        xspace = torch.linspace(eLL[0], eTR[0], x_N);
        yspace = torch.linspace(eLL[1], eTR[1], y_N);
        if self.dim == 3:
            zspace = torch.linspace(eLL[2], eTR[2], z_N);
            xgrid,ygrid,zgrid,tgrid=torch.meshgrid(xspace,yspace,zspace,tspace);
            z_sample = torch.transpose(torch.reshape(torch.stack([tgrid,xgrid,ygrid,zgrid]),(4,-1)),0,1).to(device);
        else:
            xgrid,ygrid,tgrid=torch.meshgrid(xspace,yspace,tspace);
            z_sample = torch.transpose(torch.reshape(torch.stack([tgrid,xgrid,ygrid]),(3,-1)),0,1).to(device);
        
        return z_sample.to(device)
    
    def samplerandom(self, N = 10000, bbscale = 1.1):
        [eLL,eTR] = self.extendedBB(bbscale);
        # time goes from 0 to T-1
        dT = torch.Tensor([self.T-1]).to(device); # size of time begin to end
        TC = torch.Tensor([(self.T-1.0)/2.0]).to(device); # time center
        
        z_sample = torch.rand(N, self.dim + 1).to(device)-0.5;
        deltx = torch.cat((dT,eTR-eLL))
        z_sample = deltx*z_sample + torch.cat((TC,self.C));

        return z_sample
    
# fourier features mapping
class InputMapping(nn.Module):
    def __init__(self, d_in, n_freq, sigma=2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_freq, d_in) * sigma/np.sqrt(d_in)/2.0, requires_grad=False).to(device) # gaussian
#         self.B = nn.Parameter((torch.rand(n_freq, d_in)-.5) * 2 * sigma/np.sqrt(d_in), requires_grad=False).to(device) # uniform
        self.d_in = d_in;
        self.n_freq = n_freq;
#         self.d_out = n_freq * 2 + d_in - 1;
        self.d_out = n_freq * 2 + d_in;
    def forward(self, xi):
        # x = (xi/(2*np.pi)) @ self.B.T
        x = (2*np.pi*xi) @ self.B.T
        # pdb.set_trace()
        # return torch.cat([torch.sin(x), torch.cos(x), xi[:,[1, 2]]], dim=-1)
        return torch.cat([torch.sin(x), torch.cos(x), xi], dim=-1)

class SaveTrajectory():
    # """Sample from a distribution defined by an image."""
    
    # colors = ['red','orange','magenta','cyan']
    colors = ['green','green','green','green']

    def save_trajectory(model,z_target, my_loss, savedir='imgs', nsteps=20, memory=0.01, n=1000, reverse=False):
        """
        Plot the dynamics of the learned ODE.
        Saves images to `savedir`.
        Parameters
        ----------
        model : FfjordModel
            Model defining the dynamics.
        z_target : torch.Tensor 
            Tensor of shape (T,n,d) where T is the number of timesteps
        myloss : str
            Name of loss used to train the model
        savedir : str, optional
            Where to save output.
        ntimes : int, optional
            Number of timesteps to visualize.
        memory : float
            Controls how finely the density grid is sampled.
        n : int, optional
            Number of samples to visualize.
        """

        BB = BoundingBox(z_target);
        z_sample = BB.sampleuniform(t_N = 1, x_N = 20, y_N = 20);

        if reverse: 
            my_loss+='_neg';

        final_dir = savedir+'/'+my_loss
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        T = z_target.shape[0]
        integration_times = torch.linspace(0,T-1,nsteps).to(device);
        if reverse:
            x_traj = model(z_target[T-1,:,:], integration_times, reverse=reverse).cpu().detach(); 
        else:
            x_traj = model(z_target[0,:,:], integration_times, reverse=reverse).cpu().detach()

        x_traj = x_traj.detach().numpy()

        for i in range(nsteps):
            t = integration_times[i];
            if reverse:
                t = integration_times[(T-1)-i];

            z_dots = model.time_deriv_func.get_z_dot(z_sample[:,0]*0.0 + t, z_sample[:,1:]);
            z_sample_d = z_sample.cpu().detach().numpy();
            z_dots_d = z_dots.cpu().detach().numpy();
            plt.quiver(z_sample_d[:,1], z_sample_d[:,2], z_dots_d[:,0], z_dots_d[:,1])

            plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
            for t in range(T):
                plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
            plt.axis('equal')
            plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"))
            plt.clf()

    #         plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=2.3, alpha=1, linewidths=0.1, c='blue')
    #         plt.scatter(dat.detach().numpy()[:,0],dat.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat2.detach().numpy()[:,0],dat2.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat3.detach().numpy()[:,0],dat3.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"))
    #         plt.clf()

        SaveTrajectory.trajectory_to_video(my_loss, savedir, mp4_fn='transform.mp4');


    def trajectory_to_video(my_loss,savedir='imgs', mp4_fn='transform.mp4'):
        """Save the images written by `save_trajectory` as an mp4."""
        import subprocess
        final_dir = savedir+'/'+my_loss
        img_fns = os.path.join(final_dir, 'viz-%05d.jpg')
        video_fn = os.path.join(final_dir, mp4_fn)
        bashCommand = 'ffmpeg -loglevel quiet -y -i {} {}'.format(img_fns, video_fn)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE);
        output, error = process.communicate();

