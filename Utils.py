import math
import numpy as np, math, gc
from IPython.display import clear_output
import pdb
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation
import torch
import torchvision
from torch import Tensor, nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions import MultivariateNormal
use_cuda = torch.cuda.is_available()
from geomloss import SamplesLoss
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
            xgrid,ygrid,zgrid,tgrid=torch.meshgrid(xspace,yspace,zspace,tspace,indexing='ij');
            z_sample = torch.transpose(torch.reshape(torch.stack([tgrid,xgrid,ygrid,zgrid]),(4,-1)),0,1).to(device);
        else:
            xgrid,ygrid,tgrid=torch.meshgrid(xspace,yspace,tspace,indexing='ij');
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
    def __init__(self, d_in, n_freq, sigma=2, tdiv = 2, incrementalMask = True):
        super().__init__()
        Bmat = torch.randn(n_freq, d_in) * sigma/np.sqrt(d_in)/2.0; # gaussian
        Bmat[:,d_in-1] /= tdiv; # time frequencies are a quarter of spacial frequencies.
        
        Bnorms = torch.norm(Bmat,p=2,dim=1);
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices,:]
        
        self.d_in = d_in;
        self.n_freq = n_freq;
        self.d_out = n_freq * 2 + d_in;
        self.B = nn.Linear(d_in, self.d_out, bias=False)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
            self.mask = nn.Parameter(torch.zeros(1,n_freq), requires_grad=False)
        
        self.incrementalMask = incrementalMask
        if not incrementalMask:
            self.mask = nn.Parameter(torch.ones(1,n_freq), requires_grad=False)
        
    def step(self, progressPercent):
        if self.incrementalMask:
            float_filled = (progressPercent*self.n_freq)/.7
            int_filled = int(float_filled // 1)
            remainder = float_filled % 1
            
            if int_filled >= self.n_freq:
                # pdb.set_trace()
                self.mask[0,:]=1
            else:
                self.mask[0,0:int_filled]=1
                self.mask[0,int_filled]=remainder
            
    def forward(self, xi):
        y = self.B(2*np.pi*xi)
        # pdb.set_trace()
        return torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)
        # return torch.cat([torch.sin(y), torch.cos(y), xi], dim=-1)

class SaveTrajectory():

    def gpu_usage(devnum=0):
#         print(torch.cuda.get_device_name(devnum))
        allocated = round(torch.cuda.memory_allocated(devnum)/1024**3,2);
        reserved = round(torch.cuda.memory_reserved(devnum)/1024**3,2);
        print('Allocated:', allocated, 'GB', ' Reserved:', reserved, 'GB')
    
    def save_losses(losses_in, separate_losses_in, outfolder='results/outcache/', savename = 'losses.pdf', start = 1, end = 10000, maxcap=100):
        ## SEPARATE LOSSES PLOT
        losses=losses_in.copy()
        separate_losses=separate_losses_in.copy()
        separate_losses[separate_losses>maxcap]=maxcap; losses[losses>maxcap]=maxcap
        (fig,(ax1,ax2))=plt.subplots(2,1)
        ax1.plot(losses[0,start:end],'k'); ax1.set_ylabel(f'loss\n{losses[0,:].min().item():.2f}'); ax1.set_yscale("log")
        ax2.plot(separate_losses[0,start:end],'g'); 
        ax2.plot(separate_losses[1,start:end],'g'); 
        # ax2.plot(separate_losses[0,start:end]*100,'g'); 
        # ax2.plot(separate_losses[1,start:end]*100,'g'); 
        ax2.plot(separate_losses[6,start:end],'y'); # self adv
        ax2.plot(separate_losses[7,start:end],'c'); # accel
        ax2.plot(separate_losses[9,start:end],'r'); # kurv
        ax2.plot(separate_losses[12,start:end],'b'); # u div
        # ax2.plot(separate_losses[2,start:end],'k'); 
        # ax2.plot(separate_losses[4,start:end],'k'); 
        # ax2.plot(separate_losses[5,start:end],'k'); 
        # ax2.plot(separate_losses[6,start:end],'k'); 
        # ax2.plot(separate_losses[7,start:end],'k'); 
        # ax2.plot(separate_losses[8,start:end],'k'); 
        # ax2.plot(separate_losses[11,start:end],'k'); 
        # ax2.plot(separate_losses[12,start:end],'k'); 
        ax2.set_ylabel('loss') 
        plt.savefig(outfolder + savename); 
    
    def save_trajectory(model, z_target_full, savedir='results/outcache/', savename = '', nsteps=20, dpiv=100, n=4000):
        
        # save model
        if not os.path.exists(savedir+'models/'):
            os.makedirs(savedir+'models/')
        model.save_state(fn=savedir + 'models/state_' + savename + '.tar')
        
        # save trajectory video0
        if n > z_target_full.shape[1]:
            n = z_target_full.shape[1]
        subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
        z_target = z_target_full[:,subsample_inds,:]
        
        BB = BoundingBox(z_target);
        z_sample = BB.sampleuniform(t_N = 1, x_N = 20, y_N = 20)
        z_sample_d = z_sample.cpu().detach().numpy();
        
        T = z_target.shape[0]
        integration_times = torch.linspace(0,T-1,nsteps).to(device);
        x_traj_reverse_t = model(z_target[T-1,:,:], integration_times, reverse=True)
        x_traj_forward_t = model(z_target[0,:,:], integration_times, reverse=False)
        x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
        x_traj_forward = x_traj_forward_t.cpu().detach().numpy()
        
        # forward
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        fig = plt.figure(); 
        with moviewriter.saving(fig, savedir+'forward_'+savename+'.mp4', dpiv):
            for i in range(nsteps):
                for t in range(T):
                    plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
                x_traj = x_traj_forward

                # plot velocities
                z_dots_d = model.velfunc.get_z_dot(z_sample[:,0]*0.0 + integration_times[i], z_sample[:,1:]).cpu().detach().numpy();
                plt.quiver(z_sample_d[:,1], z_sample_d[:,2], z_dots_d[:,0], z_dots_d[:,1])
                plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
                
                plt.axis('equal')
                moviewriter.grab_frame()
                plt.clf()
            moviewriter.finish()
            
        # reverse
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        with moviewriter.saving(fig, savedir+'rev_'+savename+'.mp4', dpiv):
            for i in range(nsteps):
                for t in range(T):
                    plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
                x_traj = x_traj_reverse

                # plot velocities
                z_dots_d = model.velfunc.get_z_dot(z_sample[:,0]*0.0 + integration_times[(nsteps-1)-i], z_sample[:,1:]).cpu().detach().numpy();
                plt.quiver(z_sample_d[:,1], z_sample_d[:,2], -z_dots_d[:,0], -z_dots_d[:,1])
                plt.scatter(x_traj[i,:,0], x_traj[i,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
                
                plt.axis('equal')
                moviewriter.grab_frame()
                plt.clf()
            moviewriter.finish()
            
        # forward and back
        ts = torch.linspace(0,1,nsteps)
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        with moviewriter.saving(fig, savedir+'fb_'+savename+'.mp4', dpiv):
            for tt in range(T-1):
                integration_times = torch.linspace(tt,tt+1,nsteps).to(device);
                x_traj_reverse_t = model(z_target[tt+1,:,:], integration_times, reverse=True)
                x_traj_forward_t = model(z_target[tt,:,:], integration_times, reverse=False)
                x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
                x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

                endstep = nsteps if tt==T-2 else nsteps-1
                for i in range(endstep):
                    fs = x_traj_forward_t[i,:,:]
                    ft = x_traj_reverse_t[(nsteps-1)-i,:,:]

                    # ground truth keyframes
                    for t in range(T):
                        plt.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')

                    # forward and backwards separately
                    fsp = fs.cpu().detach().numpy()
                    ftp = ft.cpu().detach().numpy()
                    plt.scatter(fsp[:,0], fsp[:,1], s=10, alpha=.5, linewidths=0, c='yellow', edgecolors='black')
                    plt.scatter(ftp[:,0], ftp[:,1], s=10, alpha=.5, linewidths=0, c='orange', edgecolors='black')

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(z_sample[:,0]*0.0 + integration_times[i], z_sample[:,1:]).cpu().detach().numpy()
                    plt.quiver(z_sample_d[:,1], z_sample_d[:,2], z_dots_d[:,0], z_dots_d[:,1])

                    # W2 barycenter combination
                    fst = MiscTransforms.OT_registration(fs.detach(), ft.detach())
                    x_traj = (fs*(1-ts[i]) + fst*ts[i]).cpu().detach().numpy()
                    plt.scatter(x_traj[:,0], x_traj[:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')

                    plt.axis('equal')
                    moviewriter.grab_frame()
                    plt.clf()
            moviewriter.finish()
        
        plt.close(fig)
            
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
        return zt

    def OT_registration(source, target):
        # pdb.set_trace()
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.001)
        x = source
        y = target
        a = source[:,0]*0.0 + 1.0/source.shape[0]
        b = target[:,0]*0.0 + 1.0/target.shape[0]

        x.requires_grad = True
        z = x.clone()  # Moving point cloud

        # pdb.set_trace()
        if use_cuda:
            torch.cuda.synchronize()

        nits = 5
        for it in range(nits):
            # wasserstein_zy = Loss(a, z, b, y)
            wasserstein_zy = Loss(z, y)
            [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
            z -= grad_z / a[:, None]  # Apply the regularized Brenier map
        
        if (z.abs()>10).any().item():
            # ot registration is unstable and overshot.
            dic = {"source":source,"target":target}
            torch.save(dic,"otdebug.tar")
            print("SAVED OT REGISTRATION ERROR")
        
        return z
    
    ## CODE SNIPPET FOR DEBUGGING OT REGISTRATION IF IT BUGS OUT.
    # import Utils; importlib.reload(Utils); from Utils import MiscTransforms
    # dic = torch.load("otdebug.tar");
    # fs = dic["source"]
    # ft = dic["target"]
    # fsp = fs.detach().cpu().numpy()
    # ftp = ft.detach().cpu().numpy()
    # plt.scatter(fsp[:,0], fsp[:,1], s=10, alpha=.5, linewidths=0, c='red', edgecolors='black')
    # plt.scatter(ftp[:,0], ftp[:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
    # z = MiscTransforms.OT_registration(fs,ft)
    # zp = z.detach().cpu().numpy()
    # plt.scatter(zp[:,0], zp[:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')