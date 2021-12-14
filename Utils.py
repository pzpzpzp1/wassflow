import math
import numpy as np
from IPython.display import clear_output
import pdb
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.nn  import functional as F 
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint
# from torch.distributions import MultivariateNormal
use_cuda = torch.cuda.is_available()
#from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import gc
import os
import cv2 as cv

# from ODEModel import FfjordModel

def ezshow(dat, col='green'):
    plt.scatter(dat.detach().numpy()[:,0],dat.detach().numpy()[:,1],s=10, alpha=0.5, linewidths=0,c=col); plt.axis('equal'); 

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin( input)

class SpecialLosses():
    def __init(self):
        super().__init__()
    def grad_to_jac(grad):
        dim = grad.shape[1]
        return grad[:,0:dim,0:dim]
    def jacdetloss(jac,beta=100):  # at 0 input, output is .006. We do want it to be stricly positive. not just 0 det.
#         pdb.set_trace()
        dets = torch.det(jac);
        return 30*nn.Softplus(beta)(-dets);
    
class ImageDataset():
    #"""Sample from a distribution defined by an image."""
    def __init__(self, imgname, thresh=51, cannylow = 50, cannyhigh = 200, rgb_weights=[0.2989, 0.5870, 0.1140,0], noise_std = .005):
        imgrgb = cv.imread(imgname, cv.IMREAD_UNCHANGED);
        img = cv.cvtColor(imgrgb, cv.COLOR_BGR2GRAY);
        edges = cv.Canny(imgrgb,cannylow,cannyhigh)
        self.img = img.copy()
        self.edges = edges.copy()
        
        imgd = img.astype('float')
        edgesd = edges.astype('float')
        
        imgd[imgd<thresh]=0; 
        imgd[imgd>=thresh]=1;
        imgd=1-imgd
        h1, w1 = imgd.shape
        
        MAX_VAL=.5
        xx = np.linspace(-MAX_VAL, MAX_VAL, w1)
        yy = np.linspace(-MAX_VAL, MAX_VAL, h1)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        self.means = np.concatenate([xx, yy], 1)

        self.probs = imgd.reshape(-1); 
        self.probs /= self.probs.sum();
        self.silprobs = edgesd.reshape(-1);
        self.silprobs /= self.silprobs.sum();
        
        self.noise_std = noise_std
        
    def sample(self, n_inner=500, n_sil = 500, scale = [1,-1], center = [0,0]):
        inds = np.random.choice(int(self.probs.shape[0]), int(n_inner), p=self.probs)
        m = self.means[inds]
        samps = torch.from_numpy(m).type(torch.FloatTensor) * torch.tensor(scale) + torch.tensor(center)
        
        sinds = np.random.choice(int(self.silprobs.shape[0]), int(n_sil), p=self.silprobs)
        ms = self.means[sinds]
        silsamples = np.random.randn(*ms.shape) * self.noise_std + ms
        silsamps =  torch.from_numpy(silsamples).type(torch.FloatTensor) * torch.tensor(scale) + torch.tensor(center)
        
        return samps, silsamps
    
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

    def normalize_samples(z_target, aux=None):
        ## normalize a [K,N,D] tensor. K is number of frames. N is number of samples. D is dimension. Fit into [-1,1] box without changing aspect ratio. centered on tight bounding box center.
        BB0 = BoundingBox(z_target);
        z_target -= BB0.C;
        BB1 = BoundingBox(z_target);
        z_target /= max(BB1.mac) 
        z_target /= 1.1; # adds buffer to the keyframes from -1,1 border.

        if aux is not None:
            aux -= BB0.C;
            aux /= max(BB1.mac)
            aux /= 1.1
        
        return z_target, aux

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
    
    def samplecustom(N = 10000):
        # sample [-1.1,1.1] x [-1.1,1.1] x [0,1]
#         pdb.set_trace()
        z_sample = torch.rand(N, 3).to(device);
        z_sample[:,0:2]-=.5
        z_sample[:,0:2]*=2.2
        
        return z_sample
    
# fourier features mapping
class InputMapping(nn.Module):
    def __init__(self, d_in, n_freq, sigma=2, tdiv = 2):
        super().__init__()
        Bmat = torch.randn(n_freq, d_in) * sigma/np.sqrt(d_in)/2.0; # gaussian
        Bmat[:,d_in-1] /= tdiv; # time frequencies are a quarter of spacial frequencies.
        self.d_in = d_in;
        self.n_freq = n_freq;
        self.d_out = n_freq * 2 + d_in;
        self.B = nn.Linear(d_in, self.d_out, bias=False)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
    def forward(self, xi):
        y = self.B(2*np.pi*xi)
        # return torch.cat([torch.sin(x), torch.cos(x), xi[:,[1, 2]]], dim=-1)
        return torch.cat([torch.sin(y), torch.cos(y), xi], dim=-1)

class SaveTrajectory():
    # """Sample from a distribution defined by an image."""
    
    # colors = ['red','orange','magenta','cyan']
    colors = ['green','green','green','green']

    def gpu_usage(devnum=0):
#         print(torch.cuda.get_device_name(devnum))
        allocated = round(torch.cuda.memory_allocated(devnum)/1024**3,2);
        reserved = round(torch.cuda.memory_reserved(devnum)/1024**3,2);
        print('Allocated:', allocated, 'GB', ' Reserved:', reserved, 'GB')
    
    def save_trajectory_vel(model,z_target, my_loss, savedir='imgs', nsteps=20, memory=0.01, n=1000, reverse=False, dpiv=100):
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

            z_dots = model.velfunc.get_z_dot(z_sample[:,0]*0.0 + t, z_sample[:,1:]);
            z_sample_d = z_sample.cpu().detach().numpy();
            z_dots_d = z_dots.cpu().detach().numpy();
            
            for t in range(T):
                plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
            
            plt.quiver(z_sample_d[:,1], z_sample_d[:,2], z_dots_d[:,0], z_dots_d[:,1])

            plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
            
            plt.axis('equal')
            plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"),dpi=dpiv)
            plt.clf()

    #         plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=2.3, alpha=1, linewidths=0.1, c='blue')
    #         plt.scatter(dat.detach().numpy()[:,0],dat.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat2.detach().numpy()[:,0],dat2.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat3.detach().numpy()[:,0],dat3.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"))
    #         plt.clf()

        # SaveTrajectory.trajectory_to_video(my_loss, savedir, mp4_fn='transform.mp4');

    def save_trajectory(model,z_target, my_loss, savedir='imgs', nsteps=20, memory=0.01, n=1000, reverse=False, dpiv=100):
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

        if reverse: 
            my_loss+='_neg';

        final_dir = savedir+'/'+my_loss
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            
        # show map as grid deformation
        model.showmap(t=0)
        plt.savefig(os.path.join(final_dir, f"map0.jpg"),dpi=dpiv)
        plt.clf()
        model.showmap(t=1)
        plt.savefig(os.path.join(final_dir, f"map1.jpg"),dpi=dpiv)
        plt.clf()
        
        
        BB = BoundingBox(z_target);
        z_sample = BB.sampleuniform(t_N = 1, x_N = 20, y_N = 20);

#         SaveTrajectory.gpu_usage(devnum=0) # check gpu memory usage
        T = z_target.shape[0]
        integration_times = torch.linspace(0,T-1,nsteps).to(device);
        
        zt = MiscTransforms.z_t_to_zt(z=z_target[0,:,:], t = integration_times)
        (z_t_2, coords) = model(zt)
        x_traj = z_t_2.reshape((integration_times.shape[0],-1,z_target.shape[2]))
#         x_traj = model(z_target[0,:,:], integration_times).cpu().detach()
#         pdb.set_trace()
        
#         SaveTrajectory.gpu_usage(devnum=0) # check gpu memory usage
        
        x_traj = x_traj.detach().cpu().numpy()
#         SaveTrajectory.gpu_usage(devnum=0) # check gpu memory usage

        for i in range(nsteps):
            t = integration_times[i];
            if reverse:
                t = integration_times[(T-1)-i];

#             z_dots = model.time_deriv_func.get_z_dot(z_sample[:,0]*0.0 + t, z_sample[:,1:]);
            z_sample_d = z_sample.cpu().detach().numpy();
#             z_dots_d = z_dots.cpu().detach().numpy();
            
            for t in range(T):
                plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
            
#             plt.quiver(z_sample_d[:,1], z_sample_d[:,2], z_dots_d[:,0], z_dots_d[:,1])

#             pdb.set_trace()
            plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
            
            plt.axis('equal')
            plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"),dpi=dpiv)
            plt.clf()
        
#         SaveTrajectory.trajectory_to_video(my_loss, savedir=savedir, mp4_fn='transform.mp4')
            
    #         plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=2.3, alpha=1, linewidths=0.1, c='blue')
    #         plt.scatter(dat.detach().numpy()[:,0],dat.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat2.detach().numpy()[:,0],dat2.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.scatter(dat3.detach().numpy()[:,0],dat3.detach().numpy()[:,1],s=2.3, alpha=0.1, linewidths=5,c='green')
    #         plt.savefig(os.path.join(final_dir, f"viz-{i:05d}.jpg"))
    #         plt.clf()
        torch.cuda.empty_cache()
        
        

    def trajectory_to_video(my_loss,savedir='imgs', mp4_fn='transform.mp4'):
        """Save the images written by `save_trajectory` as an mp4."""
        import subprocess
        final_dir = savedir+'/'+my_loss
        img_fns = os.path.join(final_dir, 'viz-%05d.jpg')
        video_fn = os.path.join(final_dir, mp4_fn)
        bashCommand = 'ffmpeg -loglevel quiet -y -i {} {}'.format(img_fns, video_fn)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE);
        output, error = process.communicate();
        torch.cuda.empty_cache()
        
class MiscTransforms():
    def z_t_to_zt(z, t):
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
