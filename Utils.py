import cv2 as cv
import os
from geomloss import SamplesLoss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import torch
import trimesh
import ot
import glob
import pdb
from scipy.spatial.distance import squareform
from torch import nn

import pandas as pd
import plotly
import plotly.express as px
from plotly import tools
from plotly.graph_objs import * #all the types of plots that we will plot here
# plotly.offline.init_notebook_mode()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ezshow(dat, col='green'):
    ax = plt.gca()
    datp = dat.detach().cpu().numpy()
    d = datp.shape[1]
    if d == 2:
        plt.scatter(datp[:, 0], datp[:, 1], s=10,
                    alpha=0.5, linewidths=0, c=col)
    elif d == 3:
        ax.scatter(datp[:, 0], datp[:, 1], datp[:, 2],
                   alpha=1, linewidths=0, c=col)
    else:
        # raise NameError("asdf")
        raise Exception("incorrect dimension")
    plt.axis('equal')
    
def ezshow3D(xyz, col='green', alpha=.1, size=1, fig=None):
    trace = Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
                    marker=dict(size=size, color=xyz[:,2], colorscale='Viridis', opacity=alpha ))
    if fig is None:
        layout = Layout(margin=dict(l=0,r=0,b=0,t=0),scene_dragmode='orbit',scene=dict(aspectmode='data'))
        # layout = Layout(margin=dict(l=0,r=0,b=0,t=0),scene_dragmode='orbit',scene=dict(aspectmode='cube'))
        # layout = Layout(margin=dict(l=0,r=0,b=0,t=0),scene_dragmode='orbit',scene=dict(aspectmode='manual',aspectratio=dict(x=1,y=1,z=1)))
        fig = Figure(data=[], layout=layout)
    fig.add_trace(trace)
    
    return fig, trace

class SpecialLosses():
    def __init(self):
        super().__init__()

    def grad_to_jac(grad):
        dim = grad.shape[1]
        return grad[:, 0:dim, 0:dim]

    def radialKE(tz, z_dots):
        dir = tz[:, 1:]
        normalizedRadial = dir/dir.norm(p=2, dim=1, keepdim=True)
        return (z_dots*normalizedRadial).sum(dim=1)**2

    def jac_to_losses(z_jacs):
        dim = z_jacs.shape[1]
        N = z_jacs.shape[0]

        # divergence squared
        div2loss = torch.zeros(N).to(device)
        for i in range(dim):
            div2loss += z_jacs[:, i, i]
        div2loss = div2loss**2
        # square norm of curl
        curl2loss = torch.norm(
            z_jacs - z_jacs.transpose(1, 2), p='fro', dim=(1, 2))**2/2

        # rigid motion: x(t) -> e^[wt] x0 + kt.
        # v = x_dot = [w]x0+k; dvdx = [w].
        # ==> skew symmetric velocity gradient is rigid.
        # if J is displacement gradient, F=J+I is the deformation gradient,
        # then F'F-I is the green strain.
        # Linearizing this with small J results in J+J'
        rigid2loss = torch.norm(
            z_jacs + z_jacs.transpose(1, 2), p='fro', dim=(1, 2))**2/4
        # v-field gradient loss
        vgradloss = torch.norm(z_jacs, p='fro', dim=(1, 2))**2

        return div2loss, curl2loss, rigid2loss, vgradloss


class ImageDataset():
    """Sample from a distribution defined by an image."""

    def __init__(self, imgname, thresh=51, cannylow=50, cannyhigh=200,
                 rgb_weights=[0.2989, 0.5870, 0.1140, 0], noise_std=.005):
        imgrgb = cv.imread(imgname, cv.IMREAD_UNCHANGED)
        img = cv.cvtColor(imgrgb, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(imgrgb, cannylow, cannyhigh)
        self.img = img.copy()
        self.edges = edges.copy()

        imgd = img.astype('float')
        edgesd = edges.astype('float')

        imgd[imgd < thresh] = 0
        imgd[imgd >= thresh] = 1
        imgd = 1-imgd
        h1, w1 = imgd.shape

        MAX_VAL = .5
        xx = np.linspace(-MAX_VAL, MAX_VAL, w1)
        yy = np.linspace(-MAX_VAL, MAX_VAL, h1)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        self.means = np.concatenate([xx, yy], 1)

        self.probs = imgd.reshape(-1)
        self.probs /= self.probs.sum()
        self.silprobs = edgesd.reshape(-1)
        self.silprobs /= self.silprobs.sum()

        self.noise_std = noise_std

    def sample(self, n_inner=500, n_sil=500, scale=[1, -1], center=[0, 0]):
        inds = np.random.choice(
            int(self.probs.shape[0]), int(n_inner), p=self.probs)
        m = self.means[inds]
        samps = torch.from_numpy(m).type(
            torch.FloatTensor) * torch.tensor(scale) + torch.tensor(center)

        sinds = np.random.choice(
            int(self.silprobs.shape[0]), int(n_sil), p=self.silprobs)
        ms = self.means[sinds]
        silsamples = np.random.randn(*ms.shape) * self.noise_std + ms
        silsamps = torch.from_numpy(silsamples).type(
            torch.FloatTensor) * torch.tensor(scale) + torch.tensor(center)

        return samps, silsamps

    def make_image(n=10000):
        """Make an X shape."""
        points = np.zeros((n, 2))
        points[:n//2, 0] = np.linspace(-1, 1, n//2)
        points[:n//2, 1] = np.linspace(1, -1, n//2)
        points[n//2:, 0] = np.linspace(1, -1, n//2)
        points[n//2:, 1] = np.linspace(1, -1, n//2)
        np.random.seed(42)
        noise = np.clip(np.random.normal(
            scale=0.1, size=points.shape), -0.2, 0.2)
        np.random.seed(None)
        points += noise
        img, _ = np.histogramdd(points, bins=40, range=[
                                [-1.5, 1.5], [-1.5, 1.5]])
        return img

    def normalize_samples(z_target, aux=None):
        # normalize a [K,N,D] tensor.
        # K is number of frames. N is number of samples. D is dimension.
        # Fit into [-1,1] box without changing aspect ratio.
        # centered on tight bounding box center.
        BB0 = BoundingBox(z_target)
        z_target -= BB0.C
        BB1 = BoundingBox(z_target)
        z_target /= max(BB1.mac)
        z_target /= 1.1  # adds buffer to the keyframes from -1,1 border.

        if aux is not None:
            aux -= BB0.C
            aux /= max(BB1.mac)
            aux /= 1.1

        return z_target, aux


class MeshDataset():
    def __init__(self, mesh_file):
        self.mesh = trimesh.load(mesh_file)
        self.mesh_file = mesh_file
    
    def getCacheName(mesh_file):
        rname, ext = os.path.splitext(mesh_file)
        fname = os.path.basename(rname)
        dname = os.path.dirname(rname)
        return os.path.join(dname,"." + fname + "_pointstore") + ext
    
    def clearCache(self):
        cacheName = MeshDataset.getCacheName(self.mesh_file)
        if os.path.exists(cacheName):
            os.remove(cacheName)

    # saves/loads sampled points
    def sample(self, n_inner=70, n_surface=30, combined = False):
        # load cache. check for already sampled points.
        cacheName = MeshDataset.getCacheName(self.mesh_file)
        if os.path.exists(cacheName):
            cdict = torch.load(cacheName)
        else:
            cdict = {'pts_inner':np.empty((0,3)), 'pts_surface':np.empty((0,3))}
        old_pts_inner = cdict["pts_inner"]
        old_pts_surface = cdict["pts_surface"]
        
        # draw point samples to fill cache
        n_new_pts_inner = max(n_inner - old_pts_inner.shape[0],0)
        n_new_pts_surface = max(n_surface - old_pts_surface.shape[0],0)
        new_pts_inner, new_pts_surface = self.sample_new(n_inner=n_new_pts_inner, n_surface=n_new_pts_surface)
        
        # save cache
        pts_inner = np.append(old_pts_inner, new_pts_inner,axis=0)
        pts_surface = np.append(old_pts_surface, new_pts_surface,axis=0)
        if n_new_pts_inner != 0 or n_new_pts_surface != 0:
            cdict = {'pts_inner':pts_inner,'pts_surface':pts_surface}
            torch.save(cdict, cacheName)
        
        # draw points needed from cache
        subsample_inds_inner = torch.randperm(pts_inner.shape[0])[:n_inner]
        subsample_inds_surface = torch.randperm(pts_surface.shape[0])[:n_surface]
        
        inner_toreturn = pts_inner[subsample_inds_inner,:]
        surface_toreturn = pts_surface[subsample_inds_surface,:]
        if not combined:
            return inner_toreturn, surface_toreturn
        else:
            return np.append(inner_toreturn, surface_toreturn, axis=0)
        
    def sample_new(self, n_inner=70, n_surface=30):
        pts_surface, _ = trimesh.sample.sample_surface(self.mesh, n_surface)
        pts_inner = trimesh.sample.volume_mesh(self.mesh, n_inner*10)[:n_inner]

        return pts_inner, pts_surface

class BoundingBox():
    # use like:
    # BB = BoundingBox(z_target);
    # smps = BB.sampleuniform(t_N = 30, x_N = 10, y_N = 11, z_N=12, bbscale = 1.1);
    # smps = BB.samplerandom(N = 10000, bbscale = 1.1);

    def __init__(self, z_target_full, square=False):
        self.T = z_target_full.shape[0]
        self.dim = z_target_full.shape[2]

        # min corner, max corner, center
        self.mic = z_target_full.reshape(-1, self.dim).min(0)[0].detach()
        self.mac = z_target_full.reshape(-1, self.dim).max(0)[0].detach()
        self.C = (self.mic+self.mac)/2

        if square:
            # min corner, max corner, center
            self.mac = self.C + (self.mac - self.C).max()
            self.mic = self.C - (self.mac - self.C).max()

    def extendedBB(self, bbscale=1.1):
        # extended bounding box.
        emic = (self.mic-self.C)*bbscale+self.C
        emac = (self.mac-self.C)*bbscale+self.C

        return emic, emac

    def sampleuniform(self, t_N=30, x_N=10, y_N=11, z_N=12, bbscale=1.1):
        [eLL, eTR] = self.extendedBB(bbscale)

        tspace = torch.linspace(0, self.T-1, t_N)
        xspace = torch.linspace(eLL[0], eTR[0], x_N)
        yspace = torch.linspace(eLL[1], eTR[1], y_N)
        if self.dim == 3:
            zspace = torch.linspace(eLL[2], eTR[2], z_N)
            xgrid, ygrid, zgrid, tgrid = torch.meshgrid(
                xspace, yspace, zspace, tspace, indexing='ij')
            z_sample = torch.transpose(torch.reshape(torch.stack(
                [tgrid, xgrid, ygrid, zgrid]), (4, -1)), 0, 1).to(device)
        else:
            xgrid, ygrid, tgrid = torch.meshgrid(
                xspace, yspace, tspace, indexing='ij')
            z_sample = torch.transpose(torch.reshape(torch.stack(
                [tgrid, xgrid, ygrid]), (3, -1)), 0, 1).to(device)

        return z_sample.to(device)

    def samplerandom(self, N=10000, bbscale=1.1):
        [eLL, eTR] = self.extendedBB(bbscale)
        # time goes from 0 to T-1
        dT = torch.Tensor([self.T-1]).to(device)  # size of time begin to end
        TC = torch.Tensor([(self.T-1.0)/2.0]).to(device)  # time center

        z_sample = torch.rand(N, self.dim + 1).to(device)-0.5
        deltx = torch.cat((dT, eTR-eLL))
        z_sample = deltx*z_sample + torch.cat((TC, self.C))

        return z_sample


class InputMapping(nn.Module):
    """Fourier features mapping"""

    def __init__(self, d_in, n_freq, sigma=2, tdiv=2, incrementalMask=True):
        super().__init__()
        Bmat = torch.randn(n_freq, d_in) * sigma/np.sqrt(d_in)/2.0  # gaussian
        # time frequencies are a quarter of spacial frequencies.
        Bmat[:, d_in-1] /= tdiv

        Bnorms = torch.norm(Bmat, p=2, dim=1)
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices, :]

        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in
        self.B = nn.Linear(d_in, self.d_out, bias=False)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
            self.mask = nn.Parameter(torch.zeros(
                1, n_freq), requires_grad=False)

        self.incrementalMask = incrementalMask
        if not incrementalMask:
            self.mask = nn.Parameter(torch.ones(
                1, n_freq), requires_grad=False)

    def step(self, progressPercent):
        if self.incrementalMask:
            float_filled = (progressPercent*self.n_freq)/.7
            int_filled = int(float_filled // 1)
            remainder = float_filled % 1

            if int_filled >= self.n_freq:
                self.mask[0, :] = 1
            else:
                self.mask[0, 0:int_filled] = 1
                # self.mask[0, int_filled] = remainder

    def forward(self, xi):
        y = self.B(2*np.pi*xi)
        return torch.cat(
            [torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)


class SaveTrajectory():

    def gpu_usage(devnum=0):
        allocated = round(torch.cuda.memory_allocated(devnum)/1024**3, 2)
        reserved = round(torch.cuda.memory_reserved(devnum)/1024**3, 2)
        print('Allocated:', allocated, 'GB', ' Reserved:', reserved, 'GB')

    def save_losses(losses_in, separate_losses_in,
                    outfolder='results/outcache/', savename='losses.pdf',
                    start=1, end=10000, maxcap=100):
        # SEPARATE LOSSES PLOT
        losses = losses_in.copy()
        separate_losses = separate_losses_in.copy()
        separate_losses[separate_losses > maxcap] = maxcap
        losses[losses > maxcap] = maxcap
        (fig, (ax1, ax2)) = plt.subplots(2, 1)
        ax1.plot(losses[0, start:end], 'k')
        ax1.set_ylabel(f'loss\n{losses[0,:].min().item():.2f}')
        ax1.set_yscale("log")
        ax2.plot(separate_losses[0, start:end], 'g')
        ax2.plot(separate_losses[1, start:end], 'g')
        # ax2.plot(separate_losses[0,start:end]*100,'g');
        # ax2.plot(separate_losses[1,start:end]*100,'g');
        ax2.plot(separate_losses[6, start:end], 'y')  # self adv
        ax2.plot(separate_losses[7, start:end], 'c')  # accel
        ax2.plot(separate_losses[9, start:end], 'r')  # kurv
        ax2.plot(separate_losses[12, start:end], 'b')  # u div
        # ax2.plot(separate_losses[2,start:end],'k');
        # ax2.plot(separate_losses[4,start:end],'k');
        # ax2.plot(separate_losses[5,start:end],'k');
        # ax2.plot(separate_losses[6,start:end],'k');
        # ax2.plot(separate_losses[7,start:end],'k');
        # ax2.plot(separate_losses[8,start:end],'k');
        # ax2.plot(separate_losses[11,start:end],'k');
        # ax2.plot(separate_losses[12,start:end],'k');
        ax2.set_ylabel('loss')
        plt.savefig(outfolder + savename)

    def save_trajectory(model, z_target_full, savedir='results/outcache/',
                        savename='', nsteps=20, dpiv=100, n=4000, alpha=.5,
                        ot_type=2, writeTracers=False, rbf=True):
        # handler for different dimensions
        if z_target_full.shape[1]==2:
            SaveTrajectory.save_trajectory_2d(model, z_target_full, savedir,
                        savename, nsteps, dpiv, n, alpha,
                        ot_type, writeTracers, rbf)
        else:
            SaveTrajectory.save_trajectory_3d(model, z_target_full, savedir,
                        savename, nsteps, dpiv, n, alpha,
                        ot_type, writeTracers, rbf)
        
    def save_trajectory_3d(model, z_target_full, savedir='results/outcache/',
                        savename='', nsteps=20, dpiv=100, n=4000, alpha=.5,
                        ot_type=2, writeTracers=False, rbf=True):
        # save model
        if not os.path.exists(savedir+'models/'):
            os.makedirs(savedir+'models/')
        model.save_state(fn=savedir + 'models/state_' + savename + '.tar')

        # save trajectory video0
        if n > z_target_full.shape[1]:
            n = z_target_full.shape[1]
        subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
        z_target = z_target_full[:, subsample_inds, :]

        T = z_target.shape[0]
        integration_times = torch.linspace(0, T-1, nsteps).to(device)
        x_traj_reverse_t = model(
            z_target[T-1, :, :], integration_times, reverse=True)
        x_traj_forward_t = model(
            z_target[0, :, :], integration_times, reverse=False)
        x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
        x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

        allpoints = torch.cat(
            (x_traj_reverse_t, x_traj_forward_t, z_target), dim=0).detach()
        BB = BoundingBox(allpoints.detach(), square=False)
        emic, emac = BB.extendedBB(1.1)
        z_sample = BB.sampleuniform(t_N=1, x_N=20, y_N=20)
        z_sample_d = z_sample.cpu().detach().numpy()

        ## FORWARD
        nframes = x_traj_forward_t.shape[0]
        npoints = x_traj_forward_t.shape[1]
        dim = x_traj_forward_t.shape[2]
        xyz_t = x_traj_forward_t.reshape((nframes*npoints,dim))
        framenum_t = torch.repeat_interleave(torch.arange(nframes),npoints)
        sizelist_t = framenum_t*0 + 1
        colorslist_t = framenum_t/nframes
        df = pd.DataFrame(dict(xp=xyz_t[:,0].cpu().detach().numpy(), 
                               yp=xyz_t[:,1].cpu().detach().numpy(), 
                               zp=xyz_t[:,2].cpu().detach().numpy(), 
                               framenum = framenum_t.cpu().detach().numpy(), 
                               colors=colorslist_t.cpu().detach().numpy() , 
                               size = sizelist_t.cpu().detach().numpy()
                              ))
        # plot animation 
        fig = px.scatter_3d(df, x="xp", y="yp", z="zp", animation_frame="framenum", size="size", color="colors")
        # plot keyframes
        for i in range(T):
            fig, _trace = ezshow3D(z_target[i,:,:].cpu().detach().numpy(), col='green', alpha=.1, size=10, fig=fig)
        # Add 2 boudning box points and set aspectmode to data: Keeps aspect ratio and fixes axes.
        invisible_scale = Scatter3d(name="",visible=True,showlegend=False,opacity=0,hoverinfo='none',x=[emic[0].item(),emac[0].item()],y=[emic[1].item(),emac[1].item()],z=[emic[2].item(),emac[2].item()])
        fig.add_trace(invisible_scale)
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),scene_dragmode='orbit',scene=dict(aspectmode='data', aspectratio=dict(x=1,y=1,z=1)))
        # fig.show()
        plotly.offline.plot(fig, filename=savedir+'forward_'+savename+'.html')
        
        
        
        
        ## REVERSE
        nframes = x_traj_reverse_t.shape[0]
        npoints = x_traj_reverse_t.shape[1]
        dim = x_traj_reverse_t.shape[2]
        xyz_t = x_traj_reverse_t.reshape((nframes*npoints,dim))
        framenum_t = torch.repeat_interleave(torch.arange(nframes),npoints)
        sizelist_t = framenum_t*0 + 1
        colorslist_t = framenum_t/nframes
        df = pd.DataFrame(dict(xp=xyz_t[:,0].cpu().detach().numpy(), 
                               yp=xyz_t[:,1].cpu().detach().numpy(), 
                               zp=xyz_t[:,2].cpu().detach().numpy(), 
                               framenum = framenum_t.cpu().detach().numpy(), 
                               colors=colorslist_t.cpu().detach().numpy() , 
                               size = sizelist_t.cpu().detach().numpy()
                              ))
        # plot animation 
        fig = px.scatter_3d(df, x="xp", y="yp", z="zp", animation_frame="framenum", size="size", color="colors")
        # plot keyframes
        for i in range(T):
            fig, _trace = ezshow3D(z_target[i,:,:].cpu().detach().numpy(), col='green', alpha=.1, size=10, fig=fig)
        # Add 2 bounding box points and set aspectmode to data: Keeps aspect ratio and fixes axes.
        invisible_scale = Scatter3d(name="",visible=True,showlegend=False,opacity=0,hoverinfo='none',x=[emic[0].item(),emac[0].item()],y=[emic[1].item(),emac[1].item()],z=[emic[2].item(),emac[2].item()])
        fig.add_trace(invisible_scale)
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),scene_dragmode='orbit',scene=dict(aspectmode='data', aspectratio=dict(x=1,y=1,z=1)))
        # fig.show()
        plotly.offline.plot(fig, filename=savedir+'reverse_'+savename+'.html')
        

        
        
        # FORWARD AND BACK
        ts = torch.linspace(0, 1, nsteps)
        x_trajs = torch.zeros(n, 2, (T-1)*(nsteps-1)+1)
        t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
        trajsc = 0
        indices = torch.arange(0, z_target.shape[1])
        for tt in range(T-1):
            if tt > 0:
                _fst, indices = MiscTransforms.OT_registration_POT_2D(
                    x_traj_t.detach(), z_target[tt, :, :].detach())
            integration_times = torch.linspace(tt, tt+1, nsteps).to(device)
            x_traj_reverse_t = model(
                z_target[tt+1, :, :], integration_times, reverse=True)
            x_traj_forward_t = model(
                z_target[tt, indices, :], integration_times, reverse=False)
            x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
            x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

            endstep = nsteps if tt == T-2 else nsteps-1
            for i in range(endstep):
                fs = x_traj_forward_t[i, :, :]
                ft = x_traj_reverse_t[(nsteps-1)-i, :, :]

                # ground truth keyframes
                for t in range(T):
                    plt.scatter(z_target.cpu().detach().numpy()[t, :, 0],
                                z_target.cpu().detach().numpy()[t, :, 1],
                                s=10, alpha=alpha, linewidths=0, c='green',
                                edgecolors='black')

                # plot velocities
                z_dots_d = model.velfunc.get_z_dot(
                    z_sample[:, 0]*0.0 + integration_times[i],
                    z_sample[:, 1:]).cpu().detach().numpy()
                plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                           z_dots_d[:, 0], z_dots_d[:, 1], lw=.01)

                # forward and backwards separately
                fsp = fs.cpu().detach().numpy()
                ftp = ft.cpu().detach().numpy()
                plt.scatter(fsp[:, 0], fsp[:, 1], s=10, alpha=alpha,
                            linewidths=0, c='yellow', edgecolors='black')
                plt.scatter(ftp[:, 0], ftp[:, 1], s=10, alpha=alpha,
                            linewidths=0, c='orange', edgecolors='black')

                # W2 barycenter combination
                if ot_type == 1:
                    # this registration isn't 1-1 on point clouds. don't know why currently.
                    fst = MiscTransforms.OT_registration(
                        fs.detach(), ft.detach())
                elif ot_type == 2:
                    # full linear program version of OT. slightly slower than geomloss but frankly not that slow compared to other steps in the pipeline.
                    fst, indices = MiscTransforms.OT_registration_POT_2D(
                        fs.detach(), ft.detach())

                x_traj_t = (fs*(1-ts[i]) + fst*ts[i])
                x_traj = x_traj_t.cpu().detach().numpy()
                plt.scatter(x_traj[:, 0], x_traj[:, 1], s=10, alpha=alpha,
                            linewidths=0, c='blue', edgecolors='black')

                x_trajs[:, :, trajsc] = x_traj_t
                t_trajs[trajsc] = integration_times[i]
                trajsc += 1

        

        if writeTracers:
            fig, (ax) = plt.subplots(1, 1)
            n = x_trajs.shape[0]  # num particles
            d = x_trajs.shape[1]  # dimension
            nf = x_trajs.shape[2]  # number of frames in full trajectory
            nft = torch.linspace(0, 1, nf)  # color tracers
            cs = torch.tensor((.3, .5, 1))  # start color
            cf = torch.tensor((.2, 1, .2))  # end color
            x_trajs_f = x_trajs.transpose(1, 2)
            nanc = torch.zeros(n, 1, d)*float("nan")
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            ax.axis('equal')
            ax.set(xlim=(emic[0].item(), emac[0].item()),
                   ylim=(emic[1].item(), emac[1].item()))
            ax.axis('off')
            dullingfactor = .7
            with moviewriter.saving(fig, savedir + 'traj_'+savename+'.mp4',
                                    dpiv):
                keyframe_percentage_curr = -1
                for t in range(0, nf):
                    ctt = (cs*(1-nft[t])+cf*nft[t])
                    dctt = (cs*(1-nft[t])+cf*nft[t])*dullingfactor
                    ct = (ctt[0].item(), ctt[1].item(), ctt[2].item())
                    dct = (dctt[0].item(), dctt[1].item(), dctt[2].item())

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(
                        z_sample[:, 0]*0.0 + t_trajs[t],
                        z_sample[:, 1:]).cpu().detach().numpy()
                    qvr = plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                                     z_dots_d[:, 0], z_dots_d[:, 1],
                                     headwidth=1, headlength=3,
                                     headaxislength=2)


                    # plot keyframes as tracers pass by
                    keyframe_percentage = np.floor(t/(nf-1.)*(T-1))
                    if keyframe_percentage != keyframe_percentage_curr:
                        keyframe_percentage_curr = keyframe_percentage
                        tt = int(keyframe_percentage)
                        if rbf and False:
                            pass
                        else:
                            plt.scatter(
                                z_target.cpu().detach().numpy()[tt, :, 0],
                                z_target.cpu().detach().numpy()[tt, :, 1],
                                s=10, alpha=1, linewidths=0, color=dct,
                                zorder=2)

                    if t > 0:
                        # plot tracers. Using the [xy;xy;nan] trick from matlab to plot all segments of a timestep at once. it's faster than a for loop at least.
                        segment_t = x_trajs_f[:, t-1:t+1, :]
                        testp = torch.cat(
                            (segment_t, nanc), dim=1).reshape(-1, d).detach(
                            ).cpu().numpy()
                        plt.plot(testp[:, 0], testp[:, 1],
                                 alpha=.3, lw=.5, color=ct, zorder=1)

                    # plot endpoints
                    if rbf or False:
                        points = x_trajs[:, :, t].detach().cuda()
                        pdists = torch.tensor(
                            squareform(torch.pdist(points.cpu()))
                        ).cuda()
                        sigmas = pdists.topk(
                            11, largest=False).values[:, -1]
                        sigmas = sigmas*0 + 0.02
                        xs = torch.linspace(-1, 1, 1000).cuda()
                        ys = torch.linspace(-1, 1, 1000).cuda()
                        grid = torch.stack(torch.meshgrid(xs, ys,
                                                          indexing='xy'),
                                           dim=-1)
                        dists = (grid[:, :, None] -
                                 points[None, None]).norm(p=2, dim=-1)
                        zs = torch.exp(
                            -(dists.pow(2) /
                              (2 * sigmas[None, None]**2))).sum(-1)
                        zs -= zs.min()
                        zs /= zs.max()
                    else:
                        endpoints = x_trajs[:, :, t].detach().cpu().numpy()
                        scr = plt.scatter(endpoints[:, 0], endpoints[:, 1],
                                          s=10, alpha=1, linewidths=0,
                                          color=dct, zorder=2, edgecolors='k')

                    moviewriter.grab_frame()
                    scr.remove()
                    qvr.remove()
            moviewriter.finish()
            plt.close(fig)

        return x_trajs
    
    def save_trajectory_2d(model, z_target_full, savedir='results/outcache/',
                        savename='', nsteps=20, dpiv=100, n=4000, alpha=.5,
                        ot_type=2, writeTracers=False, rbf=True):
        # save model
        if not os.path.exists(savedir+'models/'):
            os.makedirs(savedir+'models/')
        model.save_state(fn=savedir + 'models/state_' + savename + '.tar')

        # save trajectory video0
        if n > z_target_full.shape[1]:
            n = z_target_full.shape[1]
        subsample_inds = torch.randperm(z_target_full.shape[1])[:n]
        z_target = z_target_full[:, subsample_inds, :]

        T = z_target.shape[0]
        integration_times = torch.linspace(0, T-1, nsteps).to(device)
        x_traj_reverse_t = model(
            z_target[T-1, :, :], integration_times, reverse=True)
        x_traj_forward_t = model(
            z_target[0, :, :], integration_times, reverse=False)
        x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
        x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

        allpoints = torch.cat(
            (x_traj_reverse_t, x_traj_forward_t, z_target), dim=0).detach()
        BB = BoundingBox(allpoints.detach(), square=False)
        emic, emac = BB.extendedBB(1.1)
        z_sample = BB.sampleuniform(t_N=1, x_N=20, y_N=20)
        z_sample_d = z_sample.cpu().detach().numpy()

        # forward
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        fig, (ax) = plt.subplots(1, 1)
        with moviewriter.saving(fig, savedir+'forward_'+savename+'.mp4', dpiv):
            for i in range(nsteps):
                for t in range(T):
                    plt.scatter(
                        z_target.cpu().detach().numpy()[t, :, 0],
                        z_target.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=alpha, linewidths=0, c='green',
                        edgecolors='black')
                x_traj = x_traj_forward

                # plot velocities
                z_dots_d = model.velfunc.get_z_dot(
                    z_sample[:, 0]*0.0 + integration_times[i],
                    z_sample[:, 1:]).cpu().detach().numpy()
                plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                           z_dots_d[:, 0], z_dots_d[:, 1])
                plt.scatter(x_traj[i, :, 0], x_traj[i, :, 1], s=10,
                            alpha=alpha, linewidths=0, c='blue',
                            edgecolors='black')

                ax.axis('equal')
                ax.set(xlim=(emic[0].item(), emac[0].item()),
                       ylim=(emic[1].item(), emac[1].item()))
                plt.axis('off')
                moviewriter.grab_frame()
                plt.clf()
            moviewriter.finish()

        # reverse
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        with moviewriter.saving(fig, savedir+'rev_'+savename+'.mp4', dpiv):
            for i in range(nsteps):
                for t in range(T):
                    plt.scatter(
                        z_target.cpu().detach().numpy()[t, :, 0],
                        z_target.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=alpha, linewidths=0, c='green',
                        edgecolors='black')
                x_traj = x_traj_reverse

                # plot velocities
                z_dots_d = model.velfunc.get_z_dot(
                    z_sample[:, 0]*0.0 + integration_times[(nsteps-1)-i],
                    z_sample[:, 1:]).cpu().detach().numpy()
                plt.quiver(z_sample_d[:, 1],
                           z_sample_d[:, 2], -z_dots_d[:, 0], -z_dots_d[:, 1])
                plt.scatter(x_traj[i, :, 0], x_traj[i, :, 1], s=10,
                            alpha=alpha, linewidths=0, c='blue',
                            edgecolors='black')

                ax.axis('equal')
                ax.set(xlim=(emic[0].item(), emac[0].item()),
                       ylim=(emic[1].item(), emac[1].item()))
                plt.axis('off')
                moviewriter.grab_frame()
                plt.clf()
            moviewriter.finish()

        # forward and back
        ts = torch.linspace(0, 1, nsteps)
        moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
        x_trajs = torch.zeros(n, 2, (T-1)*(nsteps-1)+1)
        t_trajs = torch.zeros((T-1)*(nsteps-1)+1)
        trajsc = 0
        indices = torch.arange(0, z_target.shape[1])
        with moviewriter.saving(fig, savedir+'fb_'+savename+'.mp4', dpiv):
            for tt in range(T-1):
                if tt > 0:
                    # this permutation is needed to keep x_trajs continuous. otherwise at keyframes, the permutation gets reset.
                    _fst, indices = MiscTransforms.OT_registration_POT_2D(
                        x_traj_t.detach(), z_target[tt, :, :].detach())
                integration_times = torch.linspace(tt, tt+1, nsteps).to(device)
                x_traj_reverse_t = model(
                    z_target[tt+1, :, :], integration_times, reverse=True)
                x_traj_forward_t = model(
                    z_target[tt, indices, :], integration_times, reverse=False)
                x_traj_reverse = x_traj_reverse_t.cpu().detach().numpy()
                x_traj_forward = x_traj_forward_t.cpu().detach().numpy()

                endstep = nsteps if tt == T-2 else nsteps-1
                for i in range(endstep):
                    fs = x_traj_forward_t[i, :, :]
                    ft = x_traj_reverse_t[(nsteps-1)-i, :, :]

                    # ground truth keyframes
                    for t in range(T):
                        plt.scatter(z_target.cpu().detach().numpy()[t, :, 0],
                                    z_target.cpu().detach().numpy()[t, :, 1],
                                    s=10, alpha=alpha, linewidths=0, c='green',
                                    edgecolors='black')

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(
                        z_sample[:, 0]*0.0 + integration_times[i],
                        z_sample[:, 1:]).cpu().detach().numpy()
                    plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                               z_dots_d[:, 0], z_dots_d[:, 1], lw=.01)

                    # forward and backwards separately
                    fsp = fs.cpu().detach().numpy()
                    ftp = ft.cpu().detach().numpy()
                    plt.scatter(fsp[:, 0], fsp[:, 1], s=10, alpha=alpha,
                                linewidths=0, c='yellow', edgecolors='black')
                    plt.scatter(ftp[:, 0], ftp[:, 1], s=10, alpha=alpha,
                                linewidths=0, c='orange', edgecolors='black')

                    # W2 barycenter combination
                    if ot_type == 1:
                        # this registration isn't 1-1 on point clouds. don't know why currently.
                        fst = MiscTransforms.OT_registration(
                            fs.detach(), ft.detach())
                    elif ot_type == 2:
                        # full linear program version of OT. slightly slower than geomloss but frankly not that slow compared to other steps in the pipeline.
                        fst, indices = MiscTransforms.OT_registration_POT_2D(
                            fs.detach(), ft.detach())

                    x_traj_t = (fs*(1-ts[i]) + fst*ts[i])
                    x_traj = x_traj_t.cpu().detach().numpy()
                    plt.scatter(x_traj[:, 0], x_traj[:, 1], s=10, alpha=alpha,
                                linewidths=0, c='blue', edgecolors='black')

                    x_trajs[:, :, trajsc] = x_traj_t
                    t_trajs[trajsc] = integration_times[i]
                    trajsc += 1

                    ax.axis('equal')
                    plt.axis('equal')
                    ax.set(xlim=(emic[0].item(), emac[0].item()),
                           ylim=(emic[1].item(), emac[1].item()))
                    plt.axis('off')
                    moviewriter.grab_frame()
                    plt.clf()
            moviewriter.finish()
        plt.close(fig)

        if writeTracers:
            fig, (ax) = plt.subplots(1, 1)
            n = x_trajs.shape[0]  # num particles
            d = x_trajs.shape[1]  # dimension
            nf = x_trajs.shape[2]  # number of frames in full trajectory
            nft = torch.linspace(0, 1, nf)  # color tracers
            cs = torch.tensor((.3, .5, 1))  # start color
            cf = torch.tensor((.2, 1, .2))  # end color
            x_trajs_f = x_trajs.transpose(1, 2)
            nanc = torch.zeros(n, 1, d)*float("nan")
            moviewriter = matplotlib.animation.writers['ffmpeg'](fps=15)
            ax.axis('equal')
            ax.set(xlim=(emic[0].item(), emac[0].item()),
                   ylim=(emic[1].item(), emac[1].item()))
            ax.axis('off')
            dullingfactor = .7
            with moviewriter.saving(fig, savedir + 'traj_'+savename+'.mp4',
                                    dpiv):
                keyframe_percentage_curr = -1
                for t in range(0, nf):
                    ctt = (cs*(1-nft[t])+cf*nft[t])
                    dctt = (cs*(1-nft[t])+cf*nft[t])*dullingfactor
                    ct = (ctt[0].item(), ctt[1].item(), ctt[2].item())
                    dct = (dctt[0].item(), dctt[1].item(), dctt[2].item())

                    # plot velocities
                    z_dots_d = model.velfunc.get_z_dot(
                        z_sample[:, 0]*0.0 + t_trajs[t],
                        z_sample[:, 1:]).cpu().detach().numpy()
                    qvr = plt.quiver(z_sample_d[:, 1], z_sample_d[:, 2],
                                     z_dots_d[:, 0], z_dots_d[:, 1],
                                     headwidth=1, headlength=3,
                                     headaxislength=2)


                    # plot keyframes as tracers pass by
                    keyframe_percentage = np.floor(t/(nf-1.)*(T-1))
                    if keyframe_percentage != keyframe_percentage_curr:
                        keyframe_percentage_curr = keyframe_percentage
                        tt = int(keyframe_percentage)
                        if rbf and False:
                            pass
                        else:
                            plt.scatter(
                                z_target.cpu().detach().numpy()[tt, :, 0],
                                z_target.cpu().detach().numpy()[tt, :, 1],
                                s=10, alpha=1, linewidths=0, color=dct,
                                zorder=2)

                    if t > 0:
                        # plot tracers. Using the [xy;xy;nan] trick from matlab to plot all segments of a timestep at once. it's faster than a for loop at least.
                        segment_t = x_trajs_f[:, t-1:t+1, :]
                        testp = torch.cat(
                            (segment_t, nanc), dim=1).reshape(-1, d).detach(
                            ).cpu().numpy()
                        plt.plot(testp[:, 0], testp[:, 1],
                                 alpha=.3, lw=.5, color=ct, zorder=1)

                    # plot endpoints
                    if rbf or False:
                        points = x_trajs[:, :, t].detach().cuda()
                        pdists = torch.tensor(
                            squareform(torch.pdist(points.cpu()))
                        ).cuda()
                        sigmas = pdists.topk(
                            11, largest=False).values[:, -1]
                        sigmas = sigmas*0 + 0.02
                        xs = torch.linspace(-1, 1, 1000).cuda()
                        ys = torch.linspace(-1, 1, 1000).cuda()
                        grid = torch.stack(torch.meshgrid(xs, ys,
                                                          indexing='xy'),
                                           dim=-1)
                        dists = (grid[:, :, None] -
                                 points[None, None]).norm(p=2, dim=-1)
                        zs = torch.exp(
                            -(dists.pow(2) /
                              (2 * sigmas[None, None]**2))).sum(-1)
                        zs -= zs.min()
                        zs /= zs.max()
                    else:
                        endpoints = x_trajs[:, :, t].detach().cpu().numpy()
                        scr = plt.scatter(endpoints[:, 0], endpoints[:, 1],
                                          s=10, alpha=1, linewidths=0,
                                          color=dct, zorder=2, edgecolors='k')

                    moviewriter.grab_frame()
                    scr.remove()
                    qvr.remove()
            moviewriter.finish()
            plt.close(fig)

        return x_trajs


class MiscTransforms():
    def z_t_to_zt(z, t):
        """
        z: N d
        t: T
        zz: (TN) d
        tt: (TN) 1
        zt: (TN) (d+1)
        """
        zz = torch.tile(z, (t.shape[0], 1))
        tt = t.repeat_interleave(z.shape[0]).reshape((-1, 1))
        zt = torch.cat((zz, tt), dim=1)
        return zt

    def OT_registration(source, target):
        
        # SCALING EFFECTS IF A PERMUTATION IS RECOVERED OR NOT
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.001, scaling=0.99)
        x = source
        y = target
        a = source[:, 0]*0.0 + 1.0/source.shape[0]
        b = target[:, 0]*0.0 + 1.0/target.shape[0]

        x.requires_grad = True
        z = x.clone()  # Moving point cloud

        # pdb.set_trace()
        if use_cuda:
            torch.cuda.synchronize()

        nits = 5
        for it in range(nits):
            wasserstein_zy = Loss(a, z, b, y)
            # wasserstein_zy = Loss(z, y)
            [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
            z -= grad_z / a[:, None]  # Apply the regularized Brenier map

        if (z.abs() > 10).any().item():
            # ot registration is unstable and overshot.
            dic = {"source": source, "target": target}
            torch.save(dic, "otdebug.tar")
            print("SAVED OT REGISTRATION ERROR")
        return z  # , grad_z

    # should work for 3d too actually.
    def OT_registration_POT_2D(source, target):
        M = ot.dist(source, target)
        M /= M.max()
        n = source.shape[0]
        a, b = torch.ones((n,)) / n, torch.ones((n,)) / n
        Wd = ot.emd(a, b, M)
        _vals, indices = Wd.transpose(0, 1).max(dim=0)
        return target[indices, :], indices

    
    # MiscTransforms.fill_scene_caches("scenes", n_inner = 10000, n_surface = 10000)
    def fill_scene_caches(scenedir, n_inner = 10000, n_surface = 10000):
        for scene in glob.glob(scenedir+"/*"):
            for objfile in glob.glob(scene+"/*.obj"):
                print(objfile)
                mesh = MeshDataset(objfile)
                pts_inner, pts_surface = mesh.sample(n_inner=n_inner, n_surface=n_surface);
    
    # CODE SNIPPET FOR DEBUGGING GEOMLOSS OT_REGISTRATION IF IT BUGS OUT.
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
