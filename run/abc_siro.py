#!/bin/python
'''
script to run abc on sirocco with MPI

to run ABC from scratch the inputs are: 
    sim dem_model abc_name n_iter False n_part

 >>> mpiexec -n 2 python abc_slabnoll.py simba slabnoll_m simba_slabnoll_m 20 False 1000

to restart ABC from existing pool the inputs are: 
    sim dem_model abc_name n_iter True t_restart

 >>> mpiexec -n 2 python abc_slabnoll.py simba slabnoll_m simba_slabnoll_m 20 True 5

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

####################  params  ###################
dat_dir = os.environ['GALPOPFM_DIR']
distance_method = 'L2'
eps0 = [0.01, 0.001]

sim     = sys.argv[1] # name of simulation
dem     = sys.argv[2] # name of EDM model to use 

######################################################
# this will run on all processes =X
# read SED for sims 
sim_sed = dustInfer._read_sed(sim) 

# pass through the minimal amount of memory 
wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 8e3) 
# only keep centrals and impose mass limit as well.
# the lower limit log M* > 9.4 is padded by >0.25 dex to conservatively account
# for log M* and R magnitude scatter  
downsample = np.zeros(len(sim_sed['logmstar'])).astype(bool)
downsample[::10] = True
f_downsample = 0.1
cens = sim_sed['censat'].astype(bool) & (sim_sed['logmstar'] > 9.4) & downsample

# global variable that can be accessed by multiprocess (~2GB) 
shared_sim_sed = {} 
shared_sim_sed['sim']           = sim 
shared_sim_sed['logmstar']      = sim_sed['logmstar'][cens].copy()
shared_sim_sed['logsfr.100']    = sim_sed['logsfr.100'][cens].copy() 
shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 

# read SDSS observables
_, _, _, _x_obs, _x_obs_err = np.load(os.path.join(dat_dir, 'obs',
        'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_GR_FUVNUV.npy'),
        allow_pickle=True)
x_obs = [np.sum(_x_obs), _x_obs]
x_obs_err = [1., _x_obs_err]
######################################################
# functions  
###################################################### 
def dem_prior(dem_name): 
    '''
    Noll attenuation curve 
    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    slab_noll_m: 
    -----------
    ABC for slab model + Noll attenuation curve EDM that has linear log M*
    dependence  

    tauV    = m_tau (log M* - 10.) + c_tau
    delta   = m_delta  (log M* - 10.) + c_delta -2.2 < delta < 0.4
    E_b     = m_E delta + c_E

    7 free parameters:  
        theta = m_tau c_tau m_delta c_delta m_E c_E f_nebular 

    slab_noll_msfr: 
    --------------
    ABC for slab model + Noll attenuation curve EDM that has linear log M*
    and linear log SFR dependence  

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = m_tau1 (log M* - 10.) + m_tau2 logSFR + c_tau
    delta   = m_delta1  (log M* - 10.) + m_delta2 logSFR + c_delta         -2.2 < delta < 0.4
    E_b     = m_E delta + c_E

    9 free parameters:  
        theta = m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E f_nebular 
    '''
    if dem_name == 'slab_calzetti': 
        # m_tau, c_tau, fneb 
        prior_min = np.array([0., 0., 1.]) 
        prior_max = np.array([5., 6., 4.]) 
    elif dem_name == 'slab_noll_m':
        #m_tau c_tau m_delta c_delta m_E c_E fneb
        prior_min = np.array([-5., 0., -5., -4., -4., 0., 1.]) 
        prior_max = np.array([5., 6., 5., 4., 0., 4., 4.]) 
    elif dem_name == 'slab_noll_msfr':
        #m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E fneb
        prior_min = np.array([-5., -5., 0., -4., -4., -2.2, -4., 0., 1.]) 
        prior_max = np.array([5., 5., 6., 4., 4., 0.4, 0., 4., 4.]) 
    return prior_min, prior_max 


def _sumstat_model_wrap(theta, dem=dem): 
    x_mod = dustInfer.sumstat_model(theta, sed=shared_sim_sed, dem=dem,
            f_downsample=f_downsample) 
    return x_mod 


def _distance_metric_wrap(x_obs, x_model): 
    return dustInfer.distance_metric(x_obs, x_model, method=distance_method, phi_err=phi_err)


def abc(pewl, name=None, niter=None, npart=None, restart=None): 
    # read in observations 
    x_obs = dustInfer.sumstat_obs(name='sdss')
    
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
            dist=_distance_metric_wrap,   # distance metric 
            pool=pewl,
            postfn_kwargs={'dem': dem}#, dist_kwargs={'method': 'L2', 'phi_err': phi_err}
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
    return None 


if __name__=="__main__": 
    pewl = MPIPool()

    if not pewl.is_master(): 
        pewl.wait()
        sys.exit(0)

    name    = sys.argv[3] # name of ABC run
    niter   = int(sys.argv[4]) # number of iterations
    restart = (sys.argv[5] == 'True')
    print('Runnin ABC with ...') 
    print('%s simulation' % sim) 
    print('%s DEM' % dem)
    print('%i iterations' % niter)
    if not restart: 
        npart   = int(sys.argv[6]) # number of particles 
        print('%i particles' % npart)
        trest = None 
    else: 
        trest = int(sys.argv[6]) 
        print('T=%i restart' % trest) 

    abc_dir = os.path.join(dat_dir, 'abc', name) 
    if not os.path.isdir(abc_dir): 
        os.system('mkdir %s' % abc_dir)

    prior_min, prior_max = dem_prior(dem)

    abc(pewl, name=name, niter=niter, npart=npart, restart=trest) 
