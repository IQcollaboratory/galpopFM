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
    ####################################################################################
    # read in SDSS measurements 
    ####################################################################################
    Rmag_edges, gr_edges, fuvnuv_edges, x_obs, x_obs_err = np.load(os.path.join(dat_dir, 'obs', 
        'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_GR_FUVNUV.npy'), allow_pickle=True)
    dRmag   = Rmag_edges[1] - Rmag_edges[0]
    dGR     = gr_edges[1] - gr_edges[0]
    dfuvnuv = fuvnuv_edges[1] - fuvnuv_edges[0]
    ranges = [(Rmag_edges[0], Rmag_edges[-1]), (gr_edges[0], gr_edges[-1]),
            (fuvnuv_edges[0], fuvnuv_edges[-1])]
    nbar_obs = np.sum(x_obs)
    x_obs = [nbar_obs, x_obs]
    ####################################################################################
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    theta_med = np.median(theta_T, axis=0) 
    #theta_med = np.array([2., 2.]) 
    #theta_med = np.array([1., 1.]) 
    print('median ABC theta', theta_med)

    # read simulations 
    _sim_sed = dustInfer._read_sed(sim) 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
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

    x_mod = dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
            f_downsample=f_downsample)
    rho = dustInfer.distance_metric(x_obs, x_mod, method='chi2', 
            x_err=[1., x_obs_err])
    rho = dustInfer.distance_metric(x_obs, x_mod, method='L2', 
            x_err=[1., x_obs_err])
    rho = dustInfer.distance_metric(x_obs, x_mod, method='L1',
            x_err=[1., x_obs_err])
    if (np.sum(x_mod[0]) == 0.): 
        data_mod = dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
                f_downsample=f_downsample, return_datavector=True)
        print('%f < R < %f' % (-1*data_mod[0].max(), -1*data_mod[0].min()))
        print('%f < G-R < %f' % (data_mod[1].min(), data_mod[1].max()))
        print('%f < FUV-NUV < %f' % (data_mod[2].min(), data_mod[2].max()))
    ########################################################################
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(221)
    sub.pcolormesh(Rmag_edges, gr_edges, dfuvnuv * np.sum(x_obs[1], axis=2).T,
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Greys')
    sub.text(0.95, 0.95, r'SDSS', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylabel(r'$G-R$', fontsize=20) 
    sub.set_ylim(ranges[1]) 

    sub = fig.add_subplot(222)
    sub.text(0.95, 0.95, r'SIMBA', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.pcolormesh(Rmag_edges, gr_edges, dfuvnuv * np.sum(x_mod[1], axis=2).T, 
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Oranges')
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylim(ranges[1]) 
    sub.set_yticklabels([])
    
    sub = fig.add_subplot(223)
    h = sub.pcolormesh(Rmag_edges, fuvnuv_edges, dGR * np.sum(x_obs[1], axis=1).T,
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Greys')
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(ranges[2]) 

    sub = fig.add_subplot(224)
    sub.pcolormesh(Rmag_edges, fuvnuv_edges, dGR * np.sum(x_mod[1], axis=1).T,
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Oranges')
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylim(ranges[2]) 
    sub.set_yticklabels([])

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.02, 0.7])
    fig.colorbar(h, cax=cbar_ax)

    fig.savefig(os.path.join(abc_dir, 'abc_sumstat.t%i.png' % T), bbox_inches='tight') 
    return None 


def _examine_distance(T, sim='simba', dem='slab_calzetti'):
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

    # read SDSS observable
    mag_edges, gr_edges, fuvnuv_edges, _x_obs, err_x = np.load(os.path.join(dat_dir, 'obs',
            'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_GR_FUVNUV.npy'),
            allow_pickle=True)
    x_obs = [np.sum(_x_obs), _x_obs]
    print('----------------------------------------') 
    print('median posterior theta:', theta_med) 
    x_mod = dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
            f_downsample=f_downsample) 
    rho = dustInfer.distance_metric(x_obs, x_mod, method='L2', x_err=err_x)
    print(rho)
    data_mod = dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
            f_downsample=f_downsample, return_datavector=True)
    print('%f < R < %f' % (-1*data_mod[0].max(), -1*data_mod[0].min()))
    print('%f < G-R < %f' % (data_mod[1].min(), data_mod[1].max()))
    print('%f < FUV-NUV < %f' % (data_mod[2].min(), data_mod[2].max()))
    print('theta:', np.array([2., 2.]))
    x_mod = dustInfer.sumstat_model(np.array([2., 2.]), sed=sim_sed, dem=dem,
            f_downsample=f_downsample) 
    rho = dustInfer.distance_metric(x_obs, x_mod, method='L2', x_err=err_x)
    print(rho)
    data_mod = dustInfer.sumstat_model(np.array([2., 2.]), sed=sim_sed, dem=dem,
            f_downsample=f_downsample, return_datavector=True)
    print('%f < R < %f' % (-1*data_mod[0].max(), -1*data_mod[0].min()))
    print('%f < G-R < %f' % (data_mod[1].min(), data_mod[1].max()))
    print('%f < FUV-NUV < %f' % (data_mod[2].min(), data_mod[2].max()))

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
    if name == 'simba_slab_noll_simple': 
        params['sim'] = 'simba'
        params['dem'] = 'slab_noll_simple'
        params['prior_min'] = np.array([0., -4]) 
        params['prior_max'] = np.array([10., 4.]) 
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
    #plot_pool(niter, prior=prior, dem=dem, abc_dir=abc_dir)
    # plot ABCC summary statistics  
    #abc_sumstat(niter, sim=sim, dem=dem, abc_dir=abc_dir)
    _examine_distance(niter, sim=sim, dem=dem)
