#!/bin/python
'''

script to plot ABC 

'''
import os 
import sys 
import h5py 
import numpy as np 
import corner as DFM 
# -- abcpmc -- 
import abcpmc
# -- galpopfm --
from galpopfm import dustfm as dustFM
from galpopfm import dust_infer as dustInfer
from galpopfm import measure_obs as measureObs
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def plot_pool(T, prior=None, dem='slab_calzetti', abc_dir=None):
    ''' plot ABC pool 
    '''
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    pool    = abcpmc.PoolSpec(T, None, None, theta_T, rho_T, w_T) 

    dustInfer.plotABC(pool, prior=prior, dem=dem, abc_dir=abc_dir)
    return None 


def abc_sumstat(T, dem='slab_calzetti', abc_dir=None):
    ''' compare ABC summary statistics to data 
    '''
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    theta_med = np.median(theta_T, axis=0) 

    # read simulations 
    _sim_sed = dustInfer._read_sed('simba') 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 8.5) 

    sim_sed = {} 
    sim_sed['logmstar']      = _sim_sed['logmstar'][cens].copy()
    sim_sed['wave']          = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']     = _sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']   = _sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 

    R_mag_med, FUV_NUV_med, balmer_ratio_med =\
            dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem, _model=True) 

    # read in SDSS measurements 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    F_mag_sdss = sdss['ABSMAG'][...][:,0]
    N_mag_sdss = sdss['ABSMAG'][...][:,1]
    R_mag_sdss = sdss['ABSMAG'][...][:,4]
    FUV_NUV_sdss = F_mag_sdss - N_mag_sdss

    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]
    Ha_sdss = Haflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    Hb_sdss = Hbflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    balmer_ratio_sdss = Ha_sdss/Hb_sdss 

    HaHb_I = 2.86 # intrinsic balmer ratio 
    
    kwargs_median = {'rmin': -16, 'rmax': -24, 'nbins': 16} 
    ########################################################################
    fig = plt.figure(figsize=(12,5))
    # Mr - Balmer ratio
    sub = fig.add_subplot(121)
    DFM.hist2d(R_mag_sdss, np.log10(balmer_ratio_sdss/HaHb_I), color='k', 
            levels=[0.68, 0.95], range=[[-16, -24], [-0.1, 0.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_med, np.log10(balmer_ratio_med/HaHb_I), color='C1', 
            levels=[0.68, 0.95], range=[[-16, -24], [-0.1, 0.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    for i in range(20): 
        _R_mag, _, _balmer_ratio = dustInfer.sumstat_model(theta_med,
                sed=sim_sed, dem=dem, _model=True) 
        _rmid, _med = dustInfer.median_alongr(_R_mag,
                np.log10(_balmer_ratio/HaHb_I), **kwargs_median)
        sub.plot(_rmid, _med, c='C1', lw=1, alpha=0.1)

    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss,
            np.log10(balmer_ratio_sdss/HaHb_I), **kwargs_median) 
    rmid_med, med_med = dustInfer.median_alongr(R_mag_med,
            np.log10(balmer_ratio_med/HaHb_I), **kwargs_median)
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x', label='SDSS')
    sub.scatter(rmid_med, med_med, c='C1', s=30, marker='x', 
            label=r'SIMBA dust($\theta_{\rm median}$)') 

    sub.legend(loc='upper left', fontsize=15, handletextpad=0.2) 
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-16, -24) 
    sub.set_ylabel(r'$\log (H_\alpha/H_\beta)/(H_\alpha/H_\beta)_I$', fontsize=20) 
    sub.set_ylim(-0.1, 0.5) 
    
    # Mr - A_FUV
    sub = fig.add_subplot(122)
    DFM.hist2d(R_mag_sdss, FUV_NUV_sdss, color='k', 
            levels=[0.68, 0.95], range=[[-16, -24], [-0.5, 2.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_med, FUV_NUV_med, color='C1', 
            levels=[0.68, 0.95], range=[[-16, -24], [-0.5, 2.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    for i in range(20): 
        _R_mag, _FUV_NUV, _ = dustInfer.sumstat_model(theta_med, sed=sim_sed,
                dem=dem, _model=True) 
        _rmid, _med = dustInfer.median_alongr(_R_mag, _FUV_NUV, **kwargs_median)
        sub.plot(_rmid, _med, c='C1', lw=1, alpha=0.5)

    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss, FUV_NUV_sdss,
            **kwargs_median) 
    rmid_med, med_med = dustInfer.median_alongr(R_mag_med, FUV_NUV_med,
            **kwargs_median) 
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x')
    sub.scatter(rmid_med, med_med, c='C1', s=30, marker='x')

    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-16, -24) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(-0.5, 2.5) 
    fig.savefig(os.path.join(abc_dir, 'abc_sumstat.t%i.png' % T), bbox_inches='tight') 
    return None 


def abc_attenuationt(T, dem='slab_calzetti', abc_dir=None):
    '''
    '''
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    theta_med = np.median(theta_T, axis=0) 

    wave = np.linspace(1e3, 1e4, 101) 
    flux = np.ones(len(wave))
    i3000 = (np.abs(wave - 3000.)).argmin()  # index at 3000A
    
    # read simulations 
    _sim_sed = dustInfer._read_sed('simba') 
    cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 8.5) 
    logmstar = _sim_sed['logmstar'][cens].copy()
    
    A_lambdas = [] 
    for lms in logmstar[::100]:  
        A_lambda = -2.5 * np.log10(dustFM.DEM_slabcalzetti(theta_med, wave,
            flux, lms, nebular=False)) 
        A_lambdas.append(A_lambda) 

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(211)
    for A_lambda in A_lambdas: 
        sub.plot(wave, A_lambda, c='k', lw=0.1) 
    sub.set_xlim(1.5e3, 1e4) 
    sub.set_xticklabels([]) 
    sub.set_ylabel(r'$A_\lambda$', fontsize=25) 
    #sub.set_ylim(0., None) 
    sub = fig.add_subplot(212)
    for A_lambda in A_lambdas: 
        sub.plot(wave, A_lambda/A_lambda[i3000], c='k', lw=0.1) 
    sub.set_xlabel('Wavelength [$A$]', fontsize=25) 
    sub.set_xlim(1.5e3, 1e4) 
    sub.set_ylabel(r'$A_\lambda/A_{3000}$', fontsize=25) 
    fig.savefig(os.path.join(abc_dir, 'abc_attenuation.t%i.png' % T), bbox_inches='tight') 
    return None 


def run_params(name):
    ''' parameters for abc set up given name  
    '''
    params = {} 
    if name == 'test' : 
        params['dem'] = 'slab_calzetti'
        params['prior_min'] = np.array([0., 0., 2.]) 
        params['prior_max'] = np.array([5., 4., 4.]) 
    elif name == 'slabnoll_m': 
        params['dem'] = 'slab_noll_m'
        params['prior_min'] = np.array([-5., 0., -5., -2.2, -4., 0.]) 
        params['prior_max'] = np.array([5., 4., 5., 0.4, 0., 2.]) 
    elif name == 'slabnoll_msfr': 
        params['dem'] = 'slab_noll_msfr'
        params['prior_min'] = np.array([-5., -5., 0., -4., -4., -2.2, -4., 0.]) 
        params['prior_max'] = np.array([5., 5., 4., 4., 4., 0.4, 0., 2.]) 
    return params 


if __name__=="__main__": 
    ####################### inputs #######################
    name    = sys.argv[1] # name of ABC run
    niter   = int(sys.argv[2]) # number of iterations
    print('plot %s ABC iteration %i' % (name, niter)) 
    ######################################################
    dat_dir = os.environ['GALPOPFM_DIR']
    abc_dir = os.path.join(dat_dir, 'abc', name) 
    
    params = run_params(name)
    dem = params['dem'] 
    prior_min = params['prior_min'] 
    prior_max = params['prior_max'] 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 
    
    # plot the pools 
    plot_pool(niter, prior=prior, dem=dem, abc_dir=abc_dir)
    # plot ABCC summary statistics  
    #abc_sumstat(niter, dem=dem, abc_dir=abc_dir)
    # plot attenuation 
    #abc_attenuationt(niter, dem=dem, abc_dir=abc_dir)
