from tqdm import tqdm
import os
from ODEModel import FfjordModel
from Utils import (BoundingBox, ImageDataset, SaveTrajectory as st,
                   SpecialLosses as sl)
from geomloss import SamplesLoss
import numpy as np
import time
import pdb
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn_vel_trajectory(z_target_full, n_iters=10, n_subsample=100,
                         model=FfjordModel(), outname='results/outcache/',
                         visualize=False, sqrtfitloss=True, detachTZM=False, lr = 4e-4, clipnorm = 1):
    # normalize to fit in [0,1] box.
    z_target_full, __ = ImageDataset.normalize_samples(z_target_full)
    my_loss_f = SamplesLoss("sinkhorn", p=2, blur=0.00001)
    if not os.path.exists(outname):
        os.makedirs(outname)
    model.to(device)
    
    fullshape = z_target_full.shape  # [T, n_samples, d]
    T = fullshape[0]
    n_total = fullshape[1]
    dim = fullshape[2]

    # more is too slow.
    # 2000 is enough to get a reasonable capture of the image per iter.
    max_n_subsample = 1100
    if dim==3:
        max_n_subsample = 3000
    n_subsample = min(n_subsample, max_n_subsample)
    currlr = lr
    stepsperbatch = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=currlr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=.5, patience=1, min_lr=1e-7)

    BB = BoundingBox(z_target_full)

    separate_losses = np.empty((50, n_iters))
    separate_times = np.empty((4, n_iters))
    savetime = 0
    losses = np.empty((1, n_iters))
    lrs = np.empty((1, n_iters))
    n_subs = np.empty((1, n_iters))
    
    n_subsample = min(n_subsample, n_total)
    subsample_inds = torch.randperm(n_total)[:n_subsample]
    
    start = time.time()
    start0 = time.time()
    for batch in tqdm(range(n_iters)):
        # subsample z_target_full to z_target for loss computation
        z_target = torch.zeros(
            [T, n_subsample, dim]).to(z_target_full)
        for i in range(T):
            subsample_inds = torch.randperm(n_total)[:n_subsample]
            z_target[i] = z_target_full[i, subsample_inds]

        optimizer.zero_grad()
        # FORWARD and BACKWARD fitting loss
        cpt = time.time()
        
        # integrate ODE forward in time
        fitloss = torch.tensor(0.).to(device)
        for t in range(T-1):
            z_t = model(z_target[t],
                        integration_times=torch.linspace(t, t+1, 2).to(device))
            fitloss += my_loss_f(z_target[t+1], z_t[1])

        # integrate ODE backward in time from last keyframe
        fitlossb = torch.tensor(0.).to(device)
        for t in range(T-1):
            z_t_b = model(z_target[(T-1)-t],
                          integration_times=torch.linspace(
                              (T-1)-t, (T-1)-t-1, 2).to(device))
            fitlossb += my_loss_f(z_target[(T-1)-t-1], z_t_b[1])

        if batch == 0:
            # scaling factor chosen at start to normalize fitting loss
            fitloss0 = fitloss.item()
            fitlossb0 = fitlossb.item()

        fitloss /= fitloss0
        fitlossb /= fitlossb0
        separate_losses[0, batch] = fitloss
        separate_losses[1, batch] = fitlossb
        fitlosstime = time.time() - cpt

        # MASS BASED VELOCITY REGULARIZERS
        cpt = time.time()
        n_tzm_points = 30
        n_tzm_times = 5
        fbt = torch.cat((torch.rand(n_tzm_times).to(device),
                         torch.zeros(1).to(device)), 0).sort()[0]
        tzm = torch.zeros(0, dim+1).to(device)
        for i in range(T-1):
            subsample_inds = torch.randperm(n_total)[:n_tzm_points]
            # forward
            tf = i + fbt
            z_t = model(z_target_full[i, subsample_inds],
                        integration_times=tf)[1:]
            zz = z_t.reshape(n_tzm_times*n_tzm_points, dim)
            tt = tf[1:].repeat_interleave(n_tzm_points).reshape(-1, 1)
            tzm = torch.cat([tzm, torch.cat([tt, zz], 1)], 0)
            # backward
            tb = (i + 1) - fbt
            z_t = model(z_target_full[i+1, subsample_inds],
                        integration_times=tb)[1:]
            zz = z_t.reshape(n_tzm_times*n_tzm_points, dim)
            tt = tb[1:].repeat_interleave(n_tzm_points).reshape(-1, 1)
            tzm = torch.cat([tzm, torch.cat([tt, zz], 1)], 0)
        if detachTZM:
            # faster reg computation and faster backward() step.
            # not a proper gradient though.
            tzm = tzm.detach()
        z_dots, z_jacs, z_accel, z_jerk = model.velfunc.getGrads(tzm, getJerk = True)
        n_points = z_dots.shape[0]

        # div, curl, rigid, grad
        div2loss, curl2loss, rigid2loss, vgradloss = sl.jac_to_losses(z_jacs)
        # kinetic energy loss
        z_dot_norms = torch.norm(z_dots, p=2, dim=1, keepdim=True)
        KEloss = z_dot_norms[:,0]**2
        # accel loss
        Aloss = torch.norm(z_accel,p=2,dim=1)**2 
        # jerk loss
        jerkloss = torch.norm(z_jerk,p=2,dim=1)**2 
        # AV loss. (accel paralell to veloc)
        accel_in_v_dir = torch.bmm(
            z_dots.view(-1, 1, dim), z_accel).view(-1, 1) / z_dot_norms
        AVloss = accel_in_v_dir ** 2
        # self advection loss
        selfadvect = torch.bmm(
            z_jacs, z_dots.reshape(n_points, dim, 1)) + z_accel
        selfadvectloss = torch.norm(selfadvect,p=2,dim=1)**2 
        # Kurvature loss.
        z_dots_pad = z_dots
        z_accel_pad = z_accel.reshape(-1, dim)
        if dim==2:
            z_dots_pad = F.pad(z_dots_pad, (0, 1))
            z_accel_pad = F.pad(z_accel_pad, (0, 1))
        kurvature = torch.norm(
            torch.cross(z_dots_pad, z_accel_pad), p=2, dim=1,
            keepdim=True) / z_dot_norms ** 3
        Kloss = (kurvature - 1)**2
        # radial kinetic energy
        radialKE = sl.radialKE(tzm, z_dots)

        # UNIFORM SPACETIME VELOCITY REGULARIZERS
        tzu = BB.samplerandom(N=1500, bbscale=1.1)
        z_dots_u, z_jacs_u, z_accel_u,z_jerk_u = model.velfunc.getGrads(tzu, getJerk = False)
        n_points_u = z_dots_u.shape[0]

        # global div, curl, rigid, grad
        u_div2loss, u_curl2loss, u_rigid2loss, u_vgradloss = sl.jac_to_losses(z_jacs_u)
        # global self advection loss
        selfadvect_u = torch.bmm(
            z_jacs_u, z_dots_u.reshape(n_points_u, dim, 1)
        ) + z_accel_u
        u_selfadvectloss = torch.norm(selfadvect_u,p=2,dim=1)**2 
        # acceleration
        u_aloss = torch.norm(z_accel_u,p=2,dim=1)**2 
        # jerk loss
        u_jerkloss = torch.norm(z_jerk_u,p=2,dim=1)**2 

        separate_losses[2, batch] = div2loss.mean().item()
        separate_losses[3, batch] = rigid2loss.mean().item()
        separate_losses[4, batch] = vgradloss.mean().item()
        separate_losses[5, batch] = KEloss.mean().item()
        separate_losses[6, batch] = selfadvectloss.mean().item()
        separate_losses[7, batch] = Aloss.mean().item() # dampens wiggling. but also dampens rotations.
        separate_losses[8, batch] = AVloss.mean().item()
        separate_losses[9, batch] = Kloss.mean().item()
        separate_losses[10, batch] = curl2loss.mean().item()
        separate_losses[11, batch] = u_selfadvectloss.mean().item()
        separate_losses[12, batch] = u_div2loss.mean().item()
        separate_losses[13, batch] = u_aloss.mean().item()
        separate_losses[14, batch] = radialKE.mean().item()
        separate_losses[15, batch] = jerkloss.mean().item()

        # combine energies
        # timeIndices = (z_sample[:,0] < ((T-1.)/5.0)).detach()
        # timeIndices = (z_sample[:,0] < ((T-1.)/.001)).detach()
        regloss = .01 * div2loss.mean() \
            + 0 * rigid2loss.mean() \
            + .01 * vgradloss.mean() \
            + 0 * KEloss.mean() \
            + .000 * selfadvectloss.mean() \
            + .00 * Aloss.mean() \
            + .00 * AVloss.mean() \
            + .00 * Kloss.mean() \
            - 0 * torch.clamp(curl2loss.mean(), 0, .02) \
            + .0 * u_selfadvectloss.mean() \
            + .0 * u_div2loss.mean() \
            + 0 * u_aloss.mean() \
            + .00 * radialKE.mean() \
            + .01 * jerkloss.mean()
        # - 1*torch.clamp(curl2loss[timeIndices].mean(), max = 10**3)  # time negative time-truncated curl energy
        reglosstime = time.time() - cpt

        loss = fitloss + fitlossb
        if sqrtfitloss:
            loss = loss.sqrt()
        totalloss = loss + regloss
        losses[0, batch] = totalloss.item()
        n_subs[0, batch] = n_subsample
        lrs[0, batch] = currlr

        cpt = time.time()
        totalloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = clipnorm)
        optimizer.step()
        model.velfunc.imap.step((batch+1) / n_iters)
        steptime = time.time() - cpt

        if batch > 1 and batch % 50 == 0:
            # increase n_subsample by factor.
            # note this does decrease wasserstein loss
            # because sampling is a biased estimator.
            fac = 1.26
            n_subsample = round(n_subsample * fac)
            if n_subsample > z_target_full.shape[1]:
                n_subsample = z_target_full.shape[1]
            if n_subsample > max_n_subsample:
                n_subsample = max_n_subsample

        if (batch % stepsperbatch == 0 or batch == n_iters-1):
            scheduler.step(totalloss.item())  # timestep schedule.
            for g in optimizer.param_groups:
                currlr = g['lr']

            if visualize:
                if dim!=2:
                    raise Exception("no viz for 3d yet.")
                
                f, (ax1, ax2) = plt.subplots(1, 2)
                z_t = model(z_target[0],
                            integration_times=torch.linspace(
                                0, T-1, T).to(device))
                for t in range(T):
                    ax1.scatter(
                        z_t.cpu().detach().numpy()[t, :, 0],
                        z_t.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=.5, linewidths=0, c='blue',
                        edgecolors='black')
                    ax1.scatter(
                        z_target.cpu().detach().numpy()[t, :, 0],
                        z_target.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=.5, linewidths=0, c='green',
                        edgecolors='black')
                ax1.axis('equal')

                z_t = model(z_target[T-1],
                            integration_times=torch.linspace(
                                T-1, 0, T).to(device))
                for t in range(T):
                    ax2.scatter(
                        z_t.cpu().detach().numpy()[t, :, 0],
                        z_t.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=.5, linewidths=0, c="blue",
                        edgecolors='black')
                    ax2.scatter(
                        z_target.cpu().detach().numpy()[t, :, 0],
                        z_target.cpu().detach().numpy()[t, :, 1],
                        s=10, alpha=.5, linewidths=0, c="green",
                        edgecolors='black')
                ax2.axis('equal')
                plt.show()

            ptime = time.time() - start

            cpt = time.time()
            if batch > 0:
                st.save_trajectory(model, z_target_full, savedir=outname,
                                   savename=f"{batch:04}", nsteps=11, n=300,
                                   dpiv=400)
            savetime = time.time() - cpt

            # print summary stats
            st.gpu_usage()
            print(f"[Loss: {totalloss.item():.4f}",
                  f"| lr: {currlr}",
                  f"| n_subsample: {n_subsample}]",
                  f"\n[Total time : {(time.time()-start0):.4f}",
                  f"| Iter: {ptime:.4f}",
                  f"| fit: {fitlosstime:.4f}",
                  f"| reg: {reglosstime:.4f}",
                  f"| save: {savetime:.4f})",
                  f"| autograd: {steptime:.4f}]")

            start = time.time()  # reset clock to next save

        separate_times[0, batch] = fitlosstime
        separate_times[1, batch] = reglosstime
        separate_times[2, batch] = steptime
        separate_times[3, batch] = savetime

    # save stats
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
    ax1.plot(n_subs[0, :], 'r')
    ax1.set_ylabel('n_sub')
    ax2.plot(lrs[0, :], 'g')
    ax2.set_ylabel('lr')
    ax3.plot(losses[0, :], 'b')
    ax3.set_ylabel(f"loss\n{losses[0,:].min():.4f}")
    ax4.plot(separate_times[0, :], 'r')  # fit
    ax4.plot(separate_times[1, :], 'g')  # reg
    ax4.plot(separate_times[2, :], 'b')  # step
    ax4.set_ylabel(f"runtimes\n{(time.time()-start0):.4f}")
    ax5.plot(separate_times[3, :], 'r')  # save
    ax5.set_ylabel('savetimes')
    plt.savefig(outname + "stats.pdf")
    plt.close(fig)

    # save summary data:
    summarydata = {'losses': losses,
                   'separate_losses': separate_losses,
                   'lrs': lrs,
                   'n_subs': n_subs,
                   'separate_times': separate_times
                   }
    torch.save(summarydata, outname + "summary.tar")

    st.save_losses(losses, separate_losses, outfolder=outname, maxcap=10000)

    st.save_trajectory(model, z_target_full, savedir=outname,
                       savename='final', nsteps=20, dpiv=400, n=1000, writeTracers=True)

    return model, losses, separate_losses, lrs, n_subs, separate_times
