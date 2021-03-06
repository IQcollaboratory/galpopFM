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
# -- galpopfm --
from galpopfm import dust_infer as dustInfer

####################  params  ###################
dat_dir = os.environ['GALPOPFM_DIR']

machine = sys.argv[1]
sim     = sys.argv[2] # name of simulation
dem     = sys.argv[3] # name of EDM model to use 

distance_method = sys.argv[4]
statistic       = sys.argv[5]
sfr0            = sys.argv[6] # prescription for SFR=0 galaxies 
######################################################
# read SDSS summary statistics  
x_obs = dustInfer.sumstat_obs(statistic=statistic)
if distance_method == 'L2_only': x_obs = x_obs[1:]

rho_max_3d = np.sum(x_obs[0]**2)

# starting distance threshold 
if statistic == '1d': 
    if distance_method == 'L2': 
        eps0 = [4.e-5, 0.0005, 0.0002]
    elif distance_method == 'L1':
        eps0 = [0.01, 0.05, 0.02]
elif statistic == '2d': 
    if distance_method == 'L2': 
        eps0 = [4.e-5, 0.002, 0.0005]
    elif distance_method == 'L1': 
        eps0 = [0.01, 0.2, 0.1]
elif statistic == '3d': 
    if distance_method == 'L2': 
        eps0 = [4.e-5, 0.01]
    elif distance_method == 'L1': 
        eps0 = [0.01, 0.2]
    elif distance_method == 'L2_only': 
        eps0 = [rho_max_3d]
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
downsample[::5] = True
f_downsample = 0.2
#downsample[:] = True
#f_downsample = 1.

mlim = (sim_sed['logmstar'] > 9.4) 
cuts = mlim & downsample # mass limit 

# global variable that can be accessed by multiprocess (~2GB) 
shared_sim_sed = {} 
shared_sim_sed['sim']           = sim 
shared_sim_sed['logmstar']      = sim_sed['logmstar'][cuts].copy()
shared_sim_sed['logsfr.inst']   = sim_sed['logsfr.inst'][cuts].copy() 
#shared_sim_sed['logsfr.100']    = sim_sed['logsfr.100'][cens].copy() 
shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 


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
    
    slab_noll_msfr_fixbump: 
    ----------------------
    ABC for slab model + Noll attenuation curve EDM that has linear log M*
    and linear log SFR dependence. **UV bump to delta relation fixed**

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = m_tau1 (log M* - 10.) + m_tau2 logSFR + c_tau
    delta   = m_delta1  (log M* - 10.) + m_delta2 logSFR + c_delta         -2.2 < delta < 0.4
    E_b     =  -1.9 * delta + 0.85 (Kriek & Conroy 2013) 

    7 free parameters:  
        theta = m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta f_nebular 
    '''
    if dem_name == 'slab_calzetti': 
        # m_tau, c_tau, fneb 
        prior_min = np.array([0., 0., 1.]) 
        prior_max = np.array([5., 6., 4.]) 
    elif dem_name == 'slab_noll_m':
        #m_tau c_tau m_delta c_delta m_E c_E fneb
        prior_min = np.array([-5., 0., -5., -4., -4., 0., 1.]) 
        prior_max = np.array([5.0, 6., 5.0, 4.0, 0.0, 4., 4.]) 
    elif dem_name == 'slab_noll_msfr':
        #m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E fneb
        prior_min = np.array([-5., -5., 0., -4., -4., -4., -4., 0., 1.]) 
        prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    elif dem_name == 'tnorm_noll_msfr': 
        # m_mu1, m_mu2, c_mu, m_sig1, m_sig2, c_sig,  m_delta1, m_delta2, c_delta, m_E, c_E, f_nebular
        prior_min = np.array([-5., -5., 0., -5., -5., 0.1, -4., -4., -4., -4., 0., 1.]) 
        prior_max = np.array([5.0, 5.0, 6., 5.0, 5.0, 3., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    elif dem_name == 'slab_noll_msfr_fixbump':
        #m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta
        prior_min = np.array([-5., -5., 0., -4., -4., -4.]) 
        prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0]) 
    elif dem_name == 'tnorm_noll_msfr_fixbump': 
        # m_mu1, m_mu2, c_mu, m_sig1, m_sig2, c_sig,  m_delta1, m_delta2, c_delta, f_nebular
        prior_min = np.array([-5., -5., 0., -5., -5., 0.1, -4., -4., -4., 1.]) 
        prior_max = np.array([5.0, 5.0, 6., 5.0, 5.0, 3., 4.0, 4.0, 4.0, 4.]) 
    elif dem_name == 'slab_noll_msfr_kink_fixbump':
        #m_tau,M*0 m_tau,M*1 m_tau,SFR0 m_tau,SFR1 c_tau m_delta1 m_delta2 c_delta fneb
        prior_min = np.array([-5., -5., -5.,  -5., 0., -4., -4., -4., 1.]) 
        prior_max = np.array([5.0, 5.0, 5.0, 5.0, 6., 4.0, 4.0, 4.0, 4.]) 
    elif dem_name == 'slab_noll_mssfr_fixbump':
        #m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta
        prior_min = np.array([-5., -5., 0., -4., -4., -4.]) 
        prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0]) 
    return prior_min, prior_max 


def _sumstat_model_wrap(theta, dem=dem): 
    print('    [%s]' % ','.join('%.1f' % tt for tt in theta))
    x_mod = dustInfer.sumstat_model(theta, sed=shared_sim_sed, dem=dem,
            f_downsample=f_downsample, statistic=statistic, noise=True, 
            sfr0_prescription=sfr0) 
    if distance_method == 'L2_only':
        # nbar not included in distance metric 
        return x_mod[1:]
    return x_mod 


def _distance_metric_wrap(x_obs, x_model): 
    if distance_method == 'L2_only': 
        rho = dustInfer.distance_metric(x_obs, x_model, method='L2')
    else: 
        rho = dustInfer.distance_metric(x_obs, x_model, method=distance_method)
    print('     rho = %s' % ', '.join(['%.5f' % _rho for _rho in rho]))
    return rho 


def abc(pewl, name=None, niter=None, npart=None, restart=None): 
    if restart is not None:
        # read pool 
        theta_init  = np.loadtxt(
                os.path.join(abc_dir, 'theta.t%i.dat' % restart)) 
        rho_init    = np.loadtxt(
                os.path.join(abc_dir, 'rho.t%i.dat' % restart)) 
        w_init      = np.loadtxt(
                os.path.join(abc_dir, 'w.t%i.dat' % restart)) 

        fepsilon = open(os.path.join(abc_dir, 'epsilon.dat')) 
        for i, line in enumerate(fepsilon): 
            if i == restart: 
                eps_init = float(line.split(']')[0].split('[')[1]) 
                ratio_init = float(line.split(']')[1])
        npart = len(theta_init) 
        print('--- restarting from %i ---' % restart) 
        print('%i particles' % npart) 
        print('initial eps = %.5f' % eps_init) 

        init_pool = abcpmc.PoolSpec(restart, eps_init, ratio_init, theta_init, rho_init, w_init) 
        eps = abcpmc.ConstEps(niter, eps_init) 
    else: 
        init_pool = None
        # threshold 
        eps = abcpmc.ConstEps(niter, eps0) 

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

    for pool in abcpmc_sampler.sample(prior, eps, pool=init_pool):
        try: 
            eps_str = ", ".join(["{0:>.4f}".format(e) for e in pool.eps])
        except TypeError: 
            eps_str = "{0:>.4f}".format(pool.eps)

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
    if machine == 'siro': 
        # run with MPI 
        from mpi4py import MPI 
        from schwimmbad import MPIPool

        pewl = MPIPool()

        if not pewl.is_master(): 
            pewl.wait()
            sys.exit(0)
    else: 
        pewl = None 

    name    = sys.argv[7] # name of ABC run
    niter   = int(sys.argv[8]) # number of iterations
    restart = (sys.argv[9] == 'True')
    print('Runnin ABC with ...') 
    print('%s simulation' % sim) 
    print('%s DEM' % dem)
    print('%s distance' % distance_method)
    print('%s summary statistic' % statistic)
    print('%i iterations' % niter)
    if not restart: 
        npart   = int(sys.argv[10]) # number of particles 
        print('%i particles' % npart)
        trest = None 
    else: 
        trest = int(sys.argv[10]) 
        print('T=%i restart' % trest) 
        npart = None 
    
    abc_dir = os.path.join(dat_dir, 'abc', name) 
    if not os.path.isdir(abc_dir): 
        os.system('mkdir %s' % abc_dir)

    prior_min, prior_max = dem_prior(dem)

    abc(pewl, name=name, niter=niter, npart=npart, restart=trest) 
