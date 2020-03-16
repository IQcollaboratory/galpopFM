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


def abc_sumstat(T, sim='simba', dem='slab_calzetti', abc_dir=None):
    ''' compare ABC summary statistics to data 
    '''
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    theta_med = np.median(theta_T, axis=0) 

    # read simulations 
    _sim_sed = dustInfer._read_sed(sim) 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    #cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 8.5) 
    downsample = np.zeros(len(_sim_sed['logmstar'])).astype(bool)
    downsample[::10] = True
    f_downsample = 0.1
    cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 9.4) & downsample

    sim_sed = {} 
    sim_sed['sim']          = sim
    sim_sed['logmstar']     = _sim_sed['logmstar'][cens].copy()
    sim_sed['logsfr.100']   = _sim_sed['logsfr.100'][cens].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 

    sed_dusty = dustFM.Attenuate(
            theta_med, 
            sim_sed['wave'], 
            sim_sed['sed_noneb'], 
            sim_sed['sed_onlyneb'], 
            sim_sed['logmstar'],
            sim_sed['logsfr.100'],
            dem=dem) 
     
    R_mag_med, FUV_NUV_med, balmer_ratio_med =\
            dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
                    _model=True, f_downsample=f_downsample) 

    # get luminosity function 
    _, phi_sim = measureObs.LumFunc(R_mag_med, name=sim_sed['sim'], mr_bin=None)
    phi_sim /= f_downsample # in case you downsample 

    # read in SDSS measurements 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.) 
    F_mag_sdss = sdss['ABSMAG'][...][:,0][mr_complete]
    N_mag_sdss = sdss['ABSMAG'][...][:,1][mr_complete]
    R_mag_sdss = sdss['mr_tinker'][...][mr_complete] #sdss['ABSMAG'][...][:,4]
    FUV_NUV_sdss = F_mag_sdss - N_mag_sdss

    Haflux_sdss = sdss['HAFLUX'][...][mr_complete]
    Hbflux_sdss = sdss['HBFLUX'][...][mr_complete]
    balmer_ratio_sdss = Haflux_sdss/Hbflux_sdss 

    # read in SDSS LF 
    fphi = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.phi_Mr.dat') 
    mr_low, mr_high, phi_sdss, phi_sdss_err = np.loadtxt(fphi, unpack=True,
            usecols=[0,1,2,3]) 
    ########################################################################
    HaHb_I = 2.86 # intrinsic balmer ratio 
    
    kwargs_median = {'rmin': -20, 'rmax': -24, 'nbins': 8} 
    ########################################################################
    fig = plt.figure(figsize=(16,5))
    # luminosity function 
    sub = fig.add_subplot(131)
    sub.plot(0.5*(mr_low + mr_high), phi_sim, c='C0')
    sub.errorbar(0.5*(mr_low + mr_high), phi_sdss, yerr=phi_sdss_err,
            fmt='.k')
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-20, -23) 
    sub.set_yscale('log') 
    sub.set_ylim(5e-6, 1e-2) 

    # Mr - Balmer ratio
    sub = fig.add_subplot(132)
    DFM.hist2d(R_mag_sdss, np.log10(balmer_ratio_sdss/HaHb_I), color='k', 
            levels=[0.68, 0.95], range=[[-20, -24], [-0.1, 0.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_med, np.log10(balmer_ratio_med/HaHb_I), color='C1', 
            levels=[0.68, 0.95], range=[[-20, -24], [-0.1, 0.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss,
            np.log10(balmer_ratio_sdss/HaHb_I), **kwargs_median) 
    rmid_med, med_med = dustInfer.median_alongr(R_mag_med,
            np.log10(balmer_ratio_med/HaHb_I), **kwargs_median)
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x', label='SDSS')
    sub.scatter(rmid_med, med_med, c='C1', s=30, marker='x', 
            label=r'SIMBA dust($\theta_{\rm median}$)') 

    sub.legend(loc='upper left', fontsize=15, handletextpad=0.2) 
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-20, -24) 
    sub.set_ylabel(r'$\log (H_\alpha/H_\beta)/(H_\alpha/H_\beta)_I$', fontsize=20) 
    sub.set_ylim(-0.1, 0.5) 
    
    # Mr - A_FUV
    sub = fig.add_subplot(133)
    DFM.hist2d(R_mag_sdss, FUV_NUV_sdss, color='k', 
            levels=[0.68, 0.95], range=[[-20, -24], [-0.5, 2.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_med, FUV_NUV_med, color='C1', 
            levels=[0.68, 0.95], range=[[-20, -24], [-0.5, 2.5]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub) 
    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss, FUV_NUV_sdss,
            **kwargs_median) 
    rmid_med, med_med = dustInfer.median_alongr(R_mag_med, FUV_NUV_med,
            **kwargs_median) 
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x')
    sub.scatter(rmid_med, med_med, c='C1', s=30, marker='x')

    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-20, -24) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(-0.5, 2.5) 
    fig.savefig(os.path.join(abc_dir, 'abc_sumstat.t%i.png' % T), bbox_inches='tight') 
    return None 


def abc_attenuationt(T, sim='simba', dem='slab_calzetti', abc_dir=None):
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
    _sim_sed = dustInfer._read_sed(sim) 
    cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 9.4)#
    logms = _sim_sed['logmstar'][cens].copy()
    logsfr = _sim_sed['logsfr.100'][cens].copy() 
    
    A_lambdas, highmass, sfing = [], [], [] 
    for i in np.arange(np.sum(cens))[::100]:  
        if dem == 'slab_calzetti': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_slabcalzetti(theta_med, wave,
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'slab_noll_m': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_slab_noll_m(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'slab_noll_msfr': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_slab_noll_msfr(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        A_lambdas.append(A_lambda) 

        if logms[i] > 10.5: 
            highmass.append(True)
        else: highmass.append(False)

        if logsfr[i] - logms[i] > -11.: sfing.append(True)
        else: sfing.append(False)

    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(311)
    for hm, A_lambda in zip(highmass, A_lambdas): 
        if hm: 
            sub.plot(wave, A_lambda, c='C1', lw=0.1) 
        else: 
            sub.plot(wave, A_lambda, c='C0', lw=0.1) 
    sub.set_xlim(1.5e3, 1e4) 
    sub.set_xticklabels([]) 
    sub.set_ylabel(r'$A_\lambda$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(1e-4, 20.) 
    
    sub = fig.add_subplot(312)
    for sf, A_lambda in zip(sfing, A_lambdas): 
        if sf: 
            sub.plot(wave, A_lambda, c='C0', lw=0.1) 
        else: 
            sub.plot(wave, A_lambda, c='C1', lw=0.1) 
    sub.set_xlim(1.5e3, 1e4) 
    sub.set_xticklabels([]) 
    sub.set_ylabel(r'$A_\lambda$', fontsize=25) 
    sub.set_yscale('log') 
    sub.set_ylim(0.1, 20.) 

    sub = fig.add_subplot(313)
    for A_lambda in A_lambdas: 
        sub.plot(wave, A_lambda/A_lambda[i3000], c='k', lw=0.1) 
    sub.set_xlabel('Wavelength [$A$]', fontsize=25) 
    sub.set_xlim(1.5e3, 1e4) 
    sub.set_ylim(0., 10.) 
    sub.set_ylabel(r'$A_\lambda/A_{3000}$', fontsize=25) 
    fig.savefig(os.path.join(abc_dir, 'abc_attenuation.t%i.png' % T), bbox_inches='tight') 
    return None 


def run_params(name):
    ''' parameters for abc set up given name  
    '''
    params = {} 
    if name == 'test' : 
        params['sim'] = 'simba'
        params['dem'] = 'slab_calzetti'
        params['prior_min'] = np.array([0., 0., 2.]) 
        params['prior_max'] = np.array([5., 4., 4.]) 
    elif name == 'simba_slab_noll_m': 
        params['sim'] = 'simba'
        params['dem'] = 'slab_noll_m'
        params['prior_min'] = np.array([-5., 0., -5., -2.2, -4., 0., 2.]) 
        params['prior_max'] = np.array([5., 4., 5., 0.4, 0., 2., 4.]) 
    elif name == 'simba_slab_noll_msfr': 
        params['sim'] = 'simba'
        params['dem'] = 'slab_noll_msfr'
        params['prior_min'] = np.array([-5., -5., 0., -4., -4., -2.2, -4., 0., 2.]) 
        params['prior_max'] = np.array([5., 5., 4., 4., 4., 0.4, 0., 2., 4.]) 
    elif name == 'tng_slab_noll_m': 
        params['sim'] = 'tng'
        params['dem'] = 'slab_noll_m'
        params['prior_min'] = np.array([-5., 0., -5., -2.2, -4., 0., 2.]) 
        params['prior_max'] = np.array([5., 4., 5., 0.4, 0., 2., 4.]) 
    elif name == 'tng_slab_noll_msfr': 
        params['sim'] = 'tng'
        params['dem'] = 'slab_noll_msfr'
        params['prior_min'] = np.array([-5., -5., 0., -4., -4., -2.2, -4., 0., 2.]) 
        params['prior_max'] = np.array([5., 5., 4., 4., 4., 0.4, 0., 2., 4.]) 
    else: 
        raise NotImplementedError
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
    sim = params['sim'] 
    dem = params['dem'] 
    prior_min = params['prior_min'] 
    prior_max = params['prior_max'] 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 
    
    # plot the pools 
    plot_pool(niter, prior=prior, dem=dem, abc_dir=abc_dir)
    # plot ABCC summary statistics  
    abc_sumstat(niter, sim=sim, dem=dem, abc_dir=abc_dir)
    # plot attenuation 
    abc_attenuationt(niter, sim=sim, dem=dem, abc_dir=abc_dir)
