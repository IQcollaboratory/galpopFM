#!/bin/python
'''
'''
import os 
import sys 
import h5py 
import numpy as np 
import multiprocessing as mp 
from mpi4py import MPI
# -- abcpmc -- 
import abcpmc
from abcpmc import mpi_util
# -- galpopfm --
from galpopfm import dust_infer as dustInfer

os.environ["OMP_NUM_THREADS"] = "1"

####################### inputs #######################
print('Runnin test ABC with ...') 
name    = sys.argv[1] # name of ABC run
niter   = int(sys.argv[2]) # number of iterations
print('%i iterations' % niter)
nthrd = int(sys.argv[3]) 
print('%i threads' % nthrd) 
trest = int(sys.argv[4]) 
print('T=%i restart' % trest) 
######################################################
dat_dir = os.environ['GALPOPFM_DIR']
abc_dir = os.path.join(dat_dir, 'abc', name) 

prior_min = np.array([0., 0., 2.]) 
prior_max = np.array([5., 4., 4.]) 

eps0 = [10., 10.] 

dem = 'slab_calzetti'
######################################################
# read in observations 
fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
sdss = h5py.File(fsdss, 'r') 

F_mag_sdss = sdss['ABSMAG'][...][:,0]
N_mag_sdss = sdss['ABSMAG'][...][:,1]
R_mag_sdss = sdss['ABSMAG'][...][:,4]
Haflux_sdss = sdss['HAFLUX'][...]
Hbflux_sdss = sdss['HBFLUX'][...]

x_obs = dustInfer.sumstat_obs(F_mag_sdss, N_mag_sdss, R_mag_sdss, Haflux_sdss, Hbflux_sdss, sdss['Z'][...])

# read SED for sims 
sim_sed = dustInfer._read_sed('simba') 

# pass through the minimal amount of memory 
wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 1e4) 
# only keep centrals and impose mass limit as well 
cens = sim_sed['censat'].astype(bool) & (sim_sed['logmstar'] > 8.5) 

# global variable that can be accessed by multiprocess (~2GB) 
shared_sim_sed = {} 
shared_sim_sed['logmstar']      = sim_sed['logmstar'][cens].copy()
shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 
print(sys.getsizeof(shared_sim_sed))

#import time
def _sumstat_model_wrap(theta, dem='slab_calzetti'): 
    ''' wrapper for sumstat_model that works with shared memory? 
    '''
    x_mod = dustInfer.sumstat_model(theta, sed=shared_sim_sed, dem=dem) 
    return x_mod 

#--- inference with ABC-PMC below ---
# prior 
prior = abcpmc.TophatPrior(prior_min, prior_max) 

# read pool 
theta_init  = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % trest)) 
rho_init    = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % trest)) 
w_init      = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % trest)) 
init_pool = abcpmc.PoolSpec(trest, None, None, theta_init, rho_init, w_init) 

npart = len(theta_init) 
print('%i particles' % npart) 

# sampler 
abcpmc_sampler = abcpmc.Sampler(
        N=npart,                # N_particles
        Y=x_obs,                # data
        postfn=_sumstat_model_wrap,   # simulator 
        dist=dustInfer.distance_metric,   # distance metric 
        threads=nthrd,
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
    # plot ABC particles 
    #try: 
    #    dustInfer.plotABC(pool, prior=prior, dem=dem, abc_dir=abc_dir)
    #except: 
    #    pass 

    # update epsilon based on median thresholding 
    eps.eps = np.median(pool.dists, axis=0)
    print('eps%i' % pool.t, eps.eps)
    print('----------------------------------------')
    #if pool.ratio <0.2: break
abcpmc_sampler.close()