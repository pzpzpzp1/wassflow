import math, numpy as np, pdb, time, matplotlib.pyplot as plt, torch, gc, importlib, Utils, ODEModel
from IPython.display import clear_output
from torch import Tensor, nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions import MultivariateNormal
use_cuda = torch.cuda.is_available()
from geomloss import SamplesLoss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from Utils import InputMapping, BoundingBox, ImageDataset, SaveTrajectory, MiscTransforms, SaveTrajectory as st, SpecialLosses as sl
from ODEModel import velocMLP, coordMLP, FfjordModel
import os
from tqdm import tqdm
# import tensorflow as tf

def learn_vel_trajectory(z_target_full, n_iters = 10, n_subsample = 100, model=FfjordModel(), outname = 'results/outcache/', visualize=False):
    z_target_full, __ = ImageDataset.normalize_samples(z_target_full) # normalize to fit in [0,1] box.
    my_loss_f = SamplesLoss("sinkhorn", p=2, blur=0.00001)
    if not os.path.exists(outname):
        os.makedirs(outname)
        
    model.to(device)
    max_n_subsample = 1100; # more is too slow. 2000 is enough to get a reasonable capture of the image per iter.
    if n_subsample > max_n_subsample:
        n_subsample = max_n_subsample
    currlr = 1e-4;
    stepsperbatch=50
    optimizer = torch.optim.Adam(list(model.parameters()), lr=currlr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience=1,min_lr=1e-7)
    
    T = z_target_full.shape[0];
    BB = BoundingBox(z_target_full);
    
    separate_losses = np.empty((8, n_iters))
    losses = []
    lrs=[]
    n_subs=[]
    start = time.time()
    start0 = time.time()
    
    fullshape = z_target_full.shape; # [T, n_samples, d]
    if n_subsample > fullshape[1]:
        n_subsample = fullshape[1]
    subsample_inds = torch.randperm(fullshape[1])[:n_subsample];
    for batch in tqdm(range(n_iters)):
        if (batch % stepsperbatch == 1):
            start = time.time()

        # subsample z_target_full to z_target for loss computation
        z_target = torch.zeros([fullshape[0], n_subsample, fullshape[2]]).to(z_target_full)
        for i in range(fullshape[0]):
            subsample_inds = torch.randperm(fullshape[1])[:n_subsample];
            z_target[i,:,:] = z_target_full[i,subsample_inds,:];
        
        optimizer.zero_grad()
        ## FORWARD and BACKWARD fitting loss
        # integrate ODE forward in time
        cpt = time.time();
        fitloss = torch.tensor(0.).to(device)
        for t in range(0,T-1):
            z_t = model(z_target[t,:,:], integration_times = torch.linspace(t,t+1,2).to(device))
            fitloss += my_loss_f(z_target[t+1,:,:], z_t[1,:,:])
        # integrate ODE backward in time from last keyframe
        fitlossb = torch.tensor(0.).to(device)
        for t in range(0,T-1):
            z_t_b = model(z_target[(T-1)-t,:,:], integration_times = torch.linspace((T-1)-t,(T-1)-t-1,2).to(device))
            fitlossb += my_loss_f(z_target[(T-1)-t-1,:,:], z_t_b[1,:,:])
        if batch==0:
            # scaling factor chosen at start to normalize fitting loss
            fitloss0 = fitloss.item(); # constant. not differentiated through
            fitlossb0 = fitlossb.item(); # constant. not differentiated through
        fitloss/=fitloss0
        fitlossb/=fitlossb0
        separate_losses[0,batch] = fitloss
        separate_losses[1,batch] = fitlossb
        fitlosstime = time.time()-cpt
        
        # VELOCITY REGULARIZERS loss
        cpt = time.time();
        fbt = torch.cat((torch.rand(15).to(device)*(T-1.), torch.zeros(1).to(device)),0).sort()[0]
        subsample_inds = torch.randperm(fullshape[1])[:200];
        z_t = model(z_target_full[0,subsample_inds,:], integration_times = fbt).detach()[1:,:,:];
        zz = z_t.reshape(z_t.shape[0]*z_t.shape[1], z_t.shape[2])
        tt = fbt[1:].repeat_interleave(z_t.shape[1]).reshape((-1,1))
        tzm = torch.cat((tt,zz),1)
        z_dots, zt_jacs, zt_accel = model.velfunc.getGrads(tzm);
        # tzu = BB.samplerandom(N = 3000, bbscale = 1.1);
        # z_dots, zt_jacs, accels = model.velfunc.getGrads(tzu);
        
        # pdb.set_trace()
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
        # accel loss
        Aloss = zt_accel[:,0]**2 + zt_accel[:,1]**2
        
        separate_losses[2,batch] = div2loss.mean().item()
        separate_losses[3,batch] = curl2loss.mean().item()
        separate_losses[4,batch] = rigid2loss.mean().item()
        separate_losses[5,batch] = vgradloss.mean().item()
        separate_losses[6,batch] = KEloss.mean().item()
        separate_losses[7,batch] = Aloss.mean().item()
        
#         timeIndices = (z_sample[:,0] < ((T-1.)/5.0)).detach()
#         timeIndices = (z_sample[:,0] < ((T-1.)/.001)).detach()
        
        # combine energies
#         regloss = veloc_norms_2.mean()*1
        # regloss = noninversionloss.mean()*0 \
        #         + veloc_norms_2.mean()*0 \
        #         + Mnoninversionloss.mean()*0 \
        #         + KE.mean()*.00000 \
        #         + Forces.mean()*.0001
        regloss = 0 * div2loss.mean() \
                - .00* torch.clamp(curl2loss.mean(), 0, 10) \
                + 0 * rigid2loss.mean() \
                + 0 * vgradloss.mean() \
                + 0 * KEloss.mean() \
                + .01 * Aloss.mean() 
#         - 1*torch.clamp(vgradloss.mean(), max = 10**10)  # make high noise velocity field
#         - 1*torch.clamp(curl2loss[timeIndices].mean(), max = 10**3)  # time negative time-truncated curl energy
        reglosstime = time.time()-cpt
#         pdb.set_trace()
        loss = fitloss + fitlossb; 
        totalloss = loss + regloss
        losses.append(totalloss.item())
        n_subs.append(n_subsample)
        lrs.append(currlr)
        
        totalloss.backward()
        optimizer.step()
        
        if (batch>1 and batch % 50 == 0):
            # increase n_subsample by factor. note this does decrease wasserstein loss because sampling is a biased estimator.
            fac = 1.26; 
            n_subsample_p = n_subsample
            n_subsample=round(n_subsample*fac)
            if n_subsample > z_target_full.shape[1]:
                n_subsample = z_target_full.shape[1]
            if n_subsample > max_n_subsample:
                n_subsample = max_n_subsample
            
        if (batch % stepsperbatch == 0):
            scheduler.step(totalloss.item()) # timestep schedule.
            for g in optimizer.param_groups:
                currlr = g['lr'];
                
            if visualize:
                f, (ax1, ax2) = plt.subplots(1, 2)
                z_t = model(z_target[0,:,:], integration_times = torch.linspace(0,T-1,T).to(device))
                for t in range(0,T):
                    ax1.scatter(z_t.cpu().detach().numpy()[t,:,0], z_t.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='blue', edgecolors='black')
                    ax1.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c='green', edgecolors='black')
                ax1.axis('equal')

                z_t = model(z_target[T-1,:,:], integration_times = torch.linspace(T-1, 0, T).to(device))
                for t in range(0,T):
                    ax2.scatter(z_t.cpu().detach().numpy()[t,:,0], z_t.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c="blue", edgecolors='black')
                    ax2.scatter(z_target.cpu().detach().numpy()[t,:,0], z_target.cpu().detach().numpy()[t,:,1], s=10, alpha=.5, linewidths=0, c="green", edgecolors='black')
                ax2.axis('equal')
                plt.show()
            
            ptime = time.time()-start
            
            savetimebegin = time.time()
            if batch > 0:
                st.save_trajectory(model, z_target, savedir=outname, savename=f"{batch:04}", nsteps=10, n=4000, dpiv=200)
            savetime = time.time()-savetimebegin
            
            # print summary stats
            st.gpu_usage()
            print('(loss:',f"{loss.item():.4f})",'(lr:',f"{currlr})", '(n_subsample:', f"{n_subsample})",'\n(time elapsed:',f"{ptime:.4f})",'(total time:',f"{(time.time()-start0):.4f})",'(fit time:',f"{fitlosstime:.4f})",'(reg loss time:',f"{reglosstime:.4f})",'(savetime:',f"{savetime:.4f})")

    st.save_trajectory(model, z_target_full, savedir=outname, savename='final', nsteps=40, dpiv=400, n=1500)
    
    # save stats
    (fig,(ax1,ax2,ax3))=plt.subplots(3,1)
    ax1.plot(n_subs,'r'); ax1.set_ylabel('n_sub')
    ax2.plot(lrs,'g'); ax2.set_ylabel('lr') 
    ax3.plot(losses,'b'); ax3.set_ylabel('loss') 
    plt.savefig(outname + "stats.pdf"); 
    plt.close(fig)
    
    # save summary data:
    # file = open(outname + "summary.txt","w")
    
    return model, losses, separate_losses, lrs, n_subs

