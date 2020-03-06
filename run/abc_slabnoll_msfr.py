#!/bin/python
'''
ABC for slab model + Noll attenuation curve EDM that has linear log M*
and linear log SFR dependence  

A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                (k'(lambda) + D(lambda, E_b))/k_V x 
                (lambda / lambda_V)^delta

tauV    = m_tau1 (log M* - 10.) + m_tau2 logSFR + c_tau
delta   = m_delta1  (log M* - 10.) + m_delta2 logSFR + c_delta         -2.2 < delta < 0.4
E_b     = m_E delta + c_E

8 free parameter 
theta[0]: m_tau1
theta[1]: m_tau2
theta[2]: c_tau
theta[3]: m_delta1
theta[4]: m_delta2
theta[5]: c_delta
theta[6]: m_E
theta[7]: c_E
'''
import os 
import sys 
import h5py 
import numpy as np 
# -- abcpmc -- 
import abcpmc
from mpi4py import MPI 
from schwimmbad import MPIPool
# -- galpopfm --
from galpopfm import dust_infer as dustInfer

######################################################
dat_dir = os.environ['GALPOPFM_DIR']

#m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E 
prior_min = np.array([-5., -5., 0., -4., -4., -2.2, -4., 0.]) 
prior_max = np.array([5., 5., 4., 4., 4., 0.4, 0., 2.]) 

eps0 = [10., 10.] 

dem = 'slab_noll_msfr'
######################################################

# this will run on all processes =X
# read SED for sims 
sim_sed = dustInfer._read_sed('simba') 

# pass through the minimal amount of memory 
wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 8e3) 
# only keep centrals and impose mass limit as well 
downsample = np.zeros(len(sim_sed['logmstar'])).astype(bool)
downsample[::10] = True
cens = sim_sed['censat'].astype(bool) & (sim_sed['logmstar'] > 8.5) & downsample

# global variable that can be accessed by multiprocess (~2GB) 
shared_sim_sed = {} 
shared_sim_sed['logmstar']      = sim_sed['logmstar'][cens].copy()
shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 


def _sumstat_model_wrap(theta, dem=dem): 
    x_mod = dustInfer.sumstat_model(theta, sed=shared_sim_sed, dem=dem) 
    return x_mod 


def abc(pewl, name=None, niter=None, npart=None, restart=None): 
    # read in observations 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 

    F_mag_sdss = sdss['ABSMAG'][...][:,0]
    N_mag_sdss = sdss['ABSMAG'][...][:,1]
    R_mag_sdss = sdss['ABSMAG'][...][:,4]
    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]

    x_obs = dustInfer.sumstat_obs(F_mag_sdss, N_mag_sdss, R_mag_sdss, Haflux_sdss, Hbflux_sdss, sdss['Z'][...])

    #_sumstat_model_wrap = Sumstat_model_wrap() 
    
    if restart is not None:
        # read pool 
        theta_init  = np.loadtxt(
                os.path.join(abc_dir, 'theta.t%i.dat' % restart)) 
        rho_init    = np.loadtxt(
                os.path.join(abc_dir, 'rho.t%i.dat' % restart)) 
        w_init      = np.loadtxt(
                os.path.join(abc_dir, 'w.t%i.dat' % restart)) 
        init_pool = abcpmc.PoolSpec(restart, None, None, theta_init, rho_init, w_init) 

        npart = len(theta_init) 
        print('%i particles' % npart) 
    else: 
        init_pool = None

    #--- inference with ABC-PMC below ---
    # prior 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 

    # sampler 
    abcpmc_sampler = abcpmc.Sampler(
            N=npart,                # N_particles
            Y=x_obs,                # data
            postfn=_sumstat_model_wrap,   # simulator 
            dist=dustInfer.distance_metric,   # distance metric 
            pool=pewl,
            postfn_kwargs={'dem': dem},
            dist_kwargs={'method': 'L2'}
            )      

    # threshold 
    eps = abcpmc.ConstEps(niter, eps0) 

    print('eps0', eps.eps)

    for pool in abcpmc_sampler.sample(prior, eps, pool=init_pool):
        eps_str = ", ".join(["{0:>.4f}".format(e) for e in pool.eps])
        print("T: {0}, eps: [{1}], ratio: {2:>.4f}".format(pool.t, eps_str, pool.ratio))

        for i, (mean, std) in enumerate(zip(*abcpmc.weighted_avg_and_std(pool.thetas, pool.ws, axis=0))):
            print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
        print('dist', pool.dists)
        
        # write out theta, weights, and distances to file 
        dustInfer.writeABC('eps', pool, abc_dir=abc_dir)
        dustInfer.writeABC('theta', pool, abc_dir=abc_dir) 
        dustInfer.writeABC('w', pool, abc_dir=abc_dir) 
        dustInfer.writeABC('rho', pool, abc_dir=abc_dir) 

        # update epsilon based on median thresholding 
        eps.eps = np.median(pool.dists, axis=0)
        print('eps%i' % pool.t, eps.eps)
        print('----------------------------------------')
        #if pool.ratio <0.2: break
    abcpmc_sampler.close()


if __name__=="__main__": 
    pewl = MPIPool()

    if not pewl.is_master(): 
        pewl.wait()
        sys.exit(0)

    ####################### inputs #######################
    # >>> mpiexec -n 2 python abc_slabnoll.py slabnoll 3 False 5
    name    = sys.argv[1] # name of ABC run
    niter   = int(sys.argv[2]) # number of iterations
    restart = (sys.argv[3] == 'True')
    print('Runnin test ABC with ...') 
    print('%i iterations' % niter)
    if not restart: 
        npart   = int(sys.argv[4]) # number of particles 
        print('%i particles' % npart)
        trest = None 
    else: 
        trest = int(sys.argv[4]) 
        print('T=%i restart' % trest) 
    ######################################################
    abc_dir = os.path.join(dat_dir, 'abc', name) 

    abc(pewl, name=name, niter=niter, npart=npart, restart=trest) 
