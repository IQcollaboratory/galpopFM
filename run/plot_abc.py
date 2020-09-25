#!/bin/python
'''

script to plot ABC 

'''
import os 
import sys 
import h5py 
import getpass
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


def get_abc(T, name, pwd, abc_dir=None):
    ''' scp ABC files from sirocco
    '''
    if not os.path.isdir(abc_dir): 
        os.system('mkdir -p %s' % abc_dir) 

    f = open('scp_abc.expect', 'w') 
    cntnt = '\n'.join([
        '#!/usr/bin/expect',
        'spawn scp sirocco:/home/users/hahn/data/galpopfm/abc/%s/*t%i.dat %s/' % (name, T, abc_dir),
        '', 
        'expect "yes/no" {',
        '       send "yes\r"'
	'       expect "*?assword" { send "%s\r" }' % pwd, 
        '       } "*?assword" { send "%s\r" }' % pwd, 
        '', 
        'expect "yes/no" {', 
        '       send "yes\r"'
	'       expect "*?assword" { send "%s\r" }' % pwd, 
        '       } "*?assword" { send "%s\r" }' % pwd, 
        '', 
        'interact']) 
    f.write(cntnt) 
    f.close()
    cmd = 'expect scp_abc.expect' 
    os.system(cmd) 
    os.system('rm scp_abc.expect') 
    return None 


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
    r_edges, gr_edges, fn_edges, x_obs = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-1., 3.), (-1., 10.)]
    nbar_obs, x_obs_gr, x_obs_fn = x_obs
    ####################################################################################
    # read pool 
    ####################################################################################
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    theta_med = np.median(theta_T, axis=0) 
    ####################################################################################
    # read simulations 
    ####################################################################################
    _sim_sed = dustInfer._read_sed(sim) 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    downsample = np.zeros(len(_sim_sed['logmstar'])).astype(bool)
    downsample[::10] = True
    f_downsample = 0.1

    cens = _sim_sed['censat'].astype(bool) 
    mlim = (_sim_sed['logmstar'] > 9.4)
    zerosfr = (_sim_sed['logsfr.inst'] == -999)

    cuts = cens & mlim & ~zerosfr & downsample 

    sim_sed = {} 
    sim_sed['sim']          = sim
    sim_sed['logmstar']     = _sim_sed['logmstar'][cuts].copy()
    sim_sed['logsfr.inst']   = _sim_sed['logsfr.inst'][cuts].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 

    # observables for SFR = 0 simulated galaxies are directly sampled from SDSS
    # distribution 
    zerosfr_obs = dustInfer._observable_zeroSFR(
            _sim_sed['wave'][wlim], 
            _sim_sed['sed_neb'][cens & mlim & zerosfr & downsample,:][:,wlim])

    x_mod = dustInfer.sumstat_model(theta_med, sed=sim_sed, dem=dem,
            f_downsample=f_downsample, statistic='2d', extra_data=zerosfr_obs) 
    nbar_mod, x_mod_gr, x_mod_fn = x_mod
    ########################################################################
    print('obs nbar = %.4e' % nbar_obs)
    print('mod nbar = %.4e' % nbar_mod)
    ########################################################################
    fig = plt.figure(figsize=(10,10))
    sub = fig.add_subplot(221)
    sub.pcolormesh(r_edges, gr_edges, x_obs_gr.T,
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Greys')
    sub.text(0.95, 0.95, r'SDSS', ha='right', va='top', transform=sub.transAxes, fontsize=25) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylabel(r'$G-R$', fontsize=20) 
    sub.set_ylim(ranges[1]) 

    sub = fig.add_subplot(222)
    sub.pcolormesh(r_edges, gr_edges, x_mod_gr.T, 
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Oranges')
    sub.text(0.95, 0.95, sim_sed['sim'].upper(), ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylim(ranges[1]) 
    sub.set_yticklabels([])

    sub = fig.add_subplot(223)
    h = sub.pcolormesh(r_edges, fn_edges, x_obs_fn.T,
            vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap='Greys')
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(ranges[2]) 

    sub = fig.add_subplot(224)
    sub.pcolormesh(r_edges, fn_edges, x_mod_fn.T,
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
    #logsfr = _sim_sed['logsfr.100'][cens].copy() 
    logsfr = _sim_sed['logsfr.inst'][cens].copy() 
    
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
        elif dem == 'slab_noll_simple': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_slab_noll_simple(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'tnorm_noll_msfr': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_tnorm_noll_msfr(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'slab_noll_msfr_fixbump': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_slab_noll_msfr_fixbump(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'tnorm_noll_msfr_fixbump': 
            A_lambda = -2.5 * np.log10(dustFM.DEM_tnorm_noll_msfr_fixbump(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        elif dem == 'slab_noll_msfr_kink_fixbump':
            A_lambda = -2.5 * np.log10(dustFM.DEM_slab_noll_msfr_kink_fixbump(theta_med, wave, 
                flux, logms[i], logsfr[i], nebular=False)) 
        else:
            raise NotImplementedError
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
        return params 
    
    params['sim'] = name.split('.')[0]
    params['dem'] = name.split('.')[1] 
    params['distance'] = name.split('.')[2]
    params['statistic'] = name.split('.')[3] 

    if params['dem'] == 'slab_noll_m':
        #m_tau c_tau m_delta c_delta m_E c_E fneb
        params['prior_min'] = np.array([-5., 0., -5., -4., -4., 0., 1.]) 
        params['prior_max'] = np.array([5.0, 6., 5.0, 4.0, 0.0, 4., 4.]) 
    elif params['dem'] == 'slab_noll_msfr':
        #m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E fneb
        params['prior_min'] = np.array([-5., -5., 0., -4., -4., -4., -4., 0., 1.]) 
        params['prior_max'] = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    elif params['dem'] == 'slab_noll_simple': 
        params['prior_min'] = np.array([0., -4]) 
        params['prior_max'] = np.array([10., 4.]) 
    elif params['dem'] == 'tnorm_noll_msfr': 
        params['prior_min'] = np.array([-5., -5., 0., -5., -5., 0.1, -4., -4., -4., -4., 0., 1.]) 
        params['prior_max'] = np.array([5.0, 5.0, 6., 5.0, 5.0, 3., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    elif params['dem'] == 'slab_noll_msfr_fixbump':
        params['prior_min'] = np.array([-5., -5., 0., -4., -4., -4.]) 
        params['prior_max'] = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0]) 
    elif params['dem'] == 'tnorm_noll_msfr_fixbump': 
        params['prior_min'] = np.array([-5., -5., 0., -5., -5., 0.1, -4., -4., -4., 1.]) 
        params['prior_max'] = np.array([5.0, 5.0, 6., 5.0, 5.0, 3., 4.0, 4.0, 4.0, 4.]) 
    elif params['dem'] == 'slab_noll_msfr_kink_fixbump':
        #m_tau,M*0 m_tau,M*1 m_tau,SFR0 m_tau,SFR1 c_tau m_delta1 m_delta2 c_delta fneb
        params['prior_min'] = np.array([-5., -5., -5.,  -5., 0., -4., -4., -4., 1.]) 
        params['prior_max'] = np.array([5.0, 5.0, 5.0, 5.0, 6., 4.0, 4.0, 4.0, 4.]) 
    else: 
        raise NotImplementedError
    return params 


if __name__=="__main__": 
    ####################### inputs #######################
    fetch   = sys.argv[1] == 'True'
    name    = sys.argv[2] # name of ABC run
    i0      = int(sys.argv[3])
    i1      = int(sys.argv[4]) 
    if fetch: 
        pwd = getpass.getpass('sirocco password: ') 
    ######################################################
    dat_dir = os.environ['GALPOPFM_DIR']
    abc_dir = os.path.join(dat_dir, 'abc', name) 

    params = run_params(name)
    sim = params['sim'] 
    dem = params['dem'] 
    prior_min = params['prior_min'] 
    prior_max = params['prior_max'] 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 

    for niter in range(i0, i1+1):
        print('plot %s ABC iteration %i' % (name, niter)) 
        if fetch: 
            print('  downloading ABC %i' % niter) 
            get_abc(niter, name, pwd, abc_dir=abc_dir)   

        # plot the pools 
        plot_pool(niter, prior=prior, dem=dem, abc_dir=abc_dir)
        # plot ABCC summary statistics  
        abc_sumstat(niter, sim=sim, dem=dem, abc_dir=abc_dir)
        # plot attenuation 
        #abc_attenuationt(niter, sim=sim, dem=dem, abc_dir=abc_dir)
