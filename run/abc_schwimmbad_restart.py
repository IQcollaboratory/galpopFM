#!/bin/python
'''

script to run ABC with MPI through schwimmbad. 

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

prior_min = np.array([0., 0., 2.]) 
prior_max = np.array([5., 4., 4.]) 

eps0 = [10., 10.] 

dem = 'slab_calzetti'
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

#class Sumstat_model_wrap(object): 
#    ''' wrapper for sumstat_model that works with shared memory? 
#    '''
#    def __init__(self):
#        # read SED for sims 
#        sim_sed = dustInfer._read_sed('simba') 
#
#        # pass through the minimal amount of memory 
#        wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 8e3) 
#        # only keep centrals and impose mass limit as well 
#        downsample = np.zeros(len(sim_sed['logmstar'])).astype(bool)
#        downsample[::10] = True
#        cens = sim_sed['censat'].astype(bool) & (sim_sed['logmstar'] > 8.5) & downsample
#
#        # global variable that can be accessed by multiprocess (~2GB) 
#        self.shared_sim_sed = {} 
#        self.shared_sim_sed['logmstar']      = sim_sed['logmstar'][cens].copy()
#        self.shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
#        self.shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
#        self.shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 
#
#    def __call__(self, theta, dem='slab_calzetti'): 
#        x_mod = dustInfer.sumstat_model(theta, sed=self.shared_sim_sed, dem=dem) 
#        return x_mod 
def _sumstat_model_wrap(theta, dem='slab_calzetti'): 
    x_mod = dustInfer.sumstat_model(theta, sed=shared_sim_sed, dem=dem) 
    return x_mod 


def abc(pewl): 
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
    name    = sys.argv[1] # name of ABC run
    niter   = int(sys.argv[2]) # number of iterations
    print('Runnin test ABC with ...') 
    print('%i iterations' % niter)
    trest = int(sys.argv[3]) 
    print('T=%i restart' % trest) 
    ######################################################
    abc_dir = os.path.join(dat_dir, 'abc', name) 

    abc(pewl) 
