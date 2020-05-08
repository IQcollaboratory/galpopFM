#!/bin/python
'''

script for generating plots for the paper 


'''
import os 
import sys 
import h5py 
import numpy as np 
import corner as DFM 
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


dat_dir = os.environ['GALPOPFM_DIR']
fig_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'paper', 'figs') 

vol_sdss = 766021.225579427
vol_simba = 100.**3 # (Mpc/h)^3
vol_tng = 75.**3 # (Mpc/h)^3

def SDSS():
    ''' figure illustrating our SDSS 
    '''
    # read in SDSS 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5')
    sdss = h5py.File(fsdss, 'r')
    
    # get M_r and log M* 
    R_mag = sdss['mr_tinker'][...]
    logms = np.log10(sdss['ms_tinker'][...])

    Rlim = (R_mag < -20.) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)

    sub.scatter(logms, R_mag, c='k', s=1, label='SDSS VAGC')
    sub.scatter(logms[Rlim], R_mag[Rlim], c='C1', s=1, label='$M_r < -20$') 
    
    sub.legend(loc='upper left', handletextpad=0.1, markerscale=10, fontsize=25) 
    sub.set_xlabel(r'$\log(\,M_*$ [$M_\odot$]$)$', fontsize=25) 
    sub.set_xlim(9.6, 12.)

    sub.set_ylabel(r'$M_r$', fontsize=25) 
    sub.set_ylim(-17., -23.4) 

    ffig = os.path.join(fig_dir, 'sdss.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def SMFs():
    ''' figure comparing the stellar mass fucntions of SDSS and the simulations
    to show that SMFs, which more or less agree
    '''
    #########################################################################
    # read in SDSS 
    #########################################################################
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.phi_logMstar.dat') 
    logms_low, logms_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)
    #########################################################################
    # read simulations 
    #########################################################################
    fsimba = os.path.join(dat_dir, 'sed', 'simba.hdf5')
    simba = h5py.File(fsimba, 'r')
    cen_simba = simba['censat'][...].astype(bool)

    ftng = os.path.join(dat_dir, 'sed', 'tng.hdf5')
    tng = h5py.File(ftng, 'r')
    cen_tng = tng['censat'][...].astype(bool)
    #########################################################################
    # calculate SMFs
    #########################################################################
    logms_bin = np.linspace(8., 13., 21)
    dlogms = logms_bin[1:] - logms_bin[:-1]

    Ngal_simba, _   = np.histogram(simba['logmstar'][...][cen_simba], bins=logms_bin)
    Ngal_tng, _     = np.histogram(tng['logmstar'][...][cen_tng], bins=logms_bin)

    phi_simba   = Ngal_simba.astype(float) / vol_simba / dlogms
    phi_tng     = Ngal_tng.astype(float) / vol_tng / dlogms
    #########################################################################
    # plot SMFs
    #########################################################################
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.errorbar(0.5*(logms_low + logms_high), phi_sdss, yerr=err_phi_sdss,
            fmt='.k', label='SDSS Centrals')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_simba, c='C0',
            label='SIMBA')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_tng, c='C1', 
            label='TNG')
    sub.legend(loc='lower left', handletextpad=0.3, fontsize=20)
    sub.set_xlabel(r'log( $M_*$ [$M_\odot$] )', labelpad=5, fontsize=25)
    sub.set_xlim(9.7, 12.5)
    sub.set_xticks([10., 10.5, 11., 11.5, 12., 12.5]) 
    sub.set_ylabel(r'Central Stellar Mass Function ($\Phi^{\rm cen}_{M_*}$)', fontsize=24)
    sub.set_yscale("log")
    sub.set_ylim(5e-6, 3e-2) 
    
    ffig = os.path.join(fig_dir, 'smfs.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def DEM(): 
    ''' comparison of DEM attenuation curve to standard attenuation curves in
    the literature.
    '''
    k_V_calzetti = 4.87789

    def _dem(lam, logm, logsfr): 
        tauV = np.clip(1.*(logm - 10.) + 1. * logsfr + 2., 1e-3, None) 
        delta = -0.1 * (logm - 10.) + -0.1 * logsfr + -0.2 
        E_b =  -1.9 * delta + 0.85
    
        # randomly sample the inclinatiion angle from 0 - pi/2 
        incl = 0.
        sec_incl = 1./np.cos(incl) 
        #Eq. 14 of Somerville+(1999) 
        A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
        dlam = 350. # width of bump from Noll+(2009)
        lam0 = 2175. # wavelength of bump 
        # bump 
        D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)
        # calzetti is already normalized to k_V
        A_lambda = A_V * (dustFM.calzetti_absorption(lam) + D_bump / k_V_calzetti) * (lam / 5500.)**delta 
        return A_lambda

    def _salim2018(_lam, logm, logsfr): 
        lam = _lam/10000. 
        if logsfr < -0.5:  # quiescent
            RV = 3.15
            B = 1.57
            a0 = -4.30
            a1 = 2.71
            a2 = -0.191
            a3 = 0.0121
        else: 
            RV = 2.61
            B = 2.21
            a0 = -3.72
            a1 = 2.20
            a2 = -0.062
            a3 = 0.0080

        Dl = B * lam**2 * 0.035**2 / ((lam**2 - 0.2175**2)**2 + lam**2 * 0.035**2)
        kl = a0 + a1/lam + a2 / lam**2 + a3/lam**3 + Dl + RV
        return kl / RV

    def _calzetti(lam): 
        return dustFM.calzetti_absorption(lam)

    wave = np.linspace(1e3, 1e4, 1e3) 

    fig = plt.figure(figsize=(11,4))
    
    # SFing galaxies
    logSFR = 0.5
    sub = fig.add_subplot(121) 
    sub.plot(wave, _dem(wave, 9.5, logSFR), c='k', ls='--')
    sub.plot(wave, _dem(wave, 11.0, logSFR), c='k', ls='-')
    sub.plot(wave, _salim2018(wave, 11.0, logSFR), c='C0')
    sub.plot(wave, _calzetti(wave), c='C1') 
    #sub.text(0.95, 0.95, r'Star-Forming ($\log {\rm SFR} = 0.5$)', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.text(0.95, 0.95, r'Star-Forming', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlim(1.2e3, 1e4)
    sub.set_ylabel('$A(\lambda)$', fontsize=20)
    sub.set_ylim(0., 7.) 

    # Quiescent galaxies
    logSFR = -2 
    sub = fig.add_subplot(122) 
    sub.plot(wave, _dem(wave, 9.5, logSFR), lw=5, c='k', ls='--')
    _plt_lowm, = sub.plot(wave, _dem(wave, 9.5, logSFR), c='k', ls='--')
    _plt_highm, = sub.plot(wave, _dem(wave, 11.0, logSFR), c='k')
    _plt_salim, = sub.plot(wave, _salim2018(wave, 11.0, logSFR), c='C0')
    _plt_cal, = sub.plot(wave, _calzetti(wave), c='C1') 

    sub.legend([_plt_highm, _plt_lowm, _plt_cal, _plt_salim], 
            ['DEM ($M_*=10^{11}M_\odot$)', 'DEM ($M_*=10^{9.5}M_\odot$)',
                'Calzetti+(2001)', 'Salim+(2018)'], loc='lower right',
            bbox_to_anchor=(1., 0.3), handletextpad=0.25, fontsize=16) 
    #sub.text(0.95, 0.95, r'Quiescent ($\log {\rm SFR} = -2.$)', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.text(0.95, 0.95, r'Quiescent', ha='right', va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlim(1.2e3, 1e4)
    sub.set_yticklabels([]) 
    sub.set_ylim(0., 7.) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'Wavelength [$\AA$]', labelpad=5, fontsize=20) 
    #bkgd.set_ylabel(r'$P_0/P^{\rm fid}_0$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1)

    ffig = os.path.join(fig_dir, 'dems.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def Observables(): 
    ''' Figure presenting the observables along with simulations without any
    attenuation.
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(name='sdss',
            statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.5), (-1., 4.)]

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.)

    x_obs = [-1.*sdss['mr_tinker'][...][mr_complete], 
            sdss['mg_tinker'][...][mr_complete] - sdss['mr_tinker'][...][mr_complete], 
            sdss['ABSMAG'][...][:,0][mr_complete] - sdss['ABSMAG'][...][:,1][mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    x_simba, sfr0_simba  = _sim_observables('simba', np.array([0. for i in range(9)]), 
            zero_sfr_sample=False)
    x_tng, sfr0_tng      = _sim_observables('tng', np.array([0. for i in range(9)]),
            zero_sfr_sample=False)
    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_obs, x_simba, x_tng]
    names   = ['SDSS', 'SIMBA (no dust)', 'TNG (no dust)']
    clrs    = ['k', 'C1', 'C0']
    sfr0s   = [sfr0_obs, sfr0_simba, sfr0_tng] 

    fig = plt.figure(figsize=(5*len(xs),10))

    for i, _x, _sfr0, name, clr in zip(range(len(xs)), xs, sfr0s, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(_x[0][~_sfr0], _x[1][~_sfr0], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        sub.scatter(_x[0][_sfr0], _x[1][_sfr0], c='k', s=1)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=25) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1.])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        DFM.hist2d(_x[0][~_sfr0], _x[2][~_sfr0], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sfr0 = sub.scatter(_x[0][_sfr0], _x[2][_sfr0], c='k', s=1)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=25) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 

    sub.legend([sfr0], ['SFR = 0'], loc='lower right', handletextpad=0,
            markerscale=7, fontsize=20) 
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _sim_observables(sim, theta, model='slab', zero_sfr_sample=False): 
    ''' read specified simulations and return data vector 

    :param zero_sfr_sample: 
        If False, observables for SFR=0 galaxies are run through the
        regular model. 
        If True, observables for SFR=0 galaxies are sampled to match the Q
        population observables. 

    :return x_mod: 
        data vector of model(theta)  

    :return _zero_sfr: 
        indices of data vector that correspond to SFR=0 galaxies.  
    '''
    _sim_sed = dustInfer._read_sed(sim) 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    #downsample = np.zeros(len(_sim_sed['logmstar'])).astype(bool)
    #downsample[::10] = True
    downsample = np.ones(len(_sim_sed['logmstar'])).astype(bool)
    f_downsample = 1.#0.1

    cens    = _sim_sed['censat'].astype(bool) 
    mlim    = (_sim_sed['logmstar'] > 9.4) 
    zerosfr = (_sim_sed['logsfr.inst'] == -999)

    if not zero_sfr_sample: 
        cuts = cens & mlim & downsample 
    else: 
        cuts = cens & mlim & ~zerosfr & downsample 

    sim_sed = {} 
    sim_sed['sim']          = sim 
    sim_sed['logmstar']     = _sim_sed['logmstar'][cuts].copy()
    sim_sed['logsfr.inst']   = _sim_sed['logsfr.inst'][cuts].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 
    
    if not zero_sfr_sample: 
        # SFR=0 observables are *not* sampled. Returns indices 
        x_mod = dustInfer.sumstat_model(theta, sed=sim_sed,
                dem='%s_noll_msfr' % model, f_downsample=f_downsample, statistic='2d',
                return_datavector=True)
        _zerosfr = sim_sed['logsfr.inst'] == -999
    else: 
        zerosfr_obs = dustInfer._observable_zeroSFR(
                _sim_sed['wave'][wlim], 
                _sim_sed['sed_neb'][cens & mlim & zerosfr & downsample,:][:,wlim])

        x_mod = dustInfer.sumstat_model(theta, sed=sim_sed,
                dem='%s_noll_msfr' % model, f_downsample=f_downsample, statistic='2d',
                extra_data=zerosfr_obs, return_datavector=True)
        _zerosfr = np.zeros(x_mod.shape[1]).astype(bool)
        _zerosfr[np.sum(cuts):] = True
    mr_cut = x_mod[0] > 20
    return x_mod[:,mr_cut], (_zerosfr & mr_cut) 


def slab_tnorm_comparison(): 
    ''' figure comparing the A_V distributions of the slab model, tnorm and
    observed SDSS. 
    '''
    from scipy.stats import truncnorm
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=int(1e4))
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    slab_AV = lambda tauV: -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl))

    tnorm_AV = lambda mu_Av, sig_Av: truncnorm.rvs((0. - mu_Av)/sig_Av, np.inf, loc=mu_Av, scale=sig_Av,
            size=int(1e4)) 
    fake_sdss = 1. + 0.7 * np.random.randn(int(1e4))

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111) 
    _ = sub.hist(np.array(fake_sdss), range=(-3., 7), bins=51, density=True, 
            color='C0', linestyle='-', histtype='stepfilled', label='SDSS')
    _ = sub.hist(np.array(slab_AV(2.)), range=(-3., 7), bins=51, density=True, 
            color='k', linestyle='--', linewidth=2, histtype='step', 
            label=r'slab model')
    _ = sub.hist(np.array(tnorm_AV(1., 0.8)), range=(-3., 7), bins=51, density=True, 
            color='C1', linestyle='--', linewidth=2, histtype='step', 
            label=r'$\mathcal{N}_T$ model')
    sub.legend(loc='upper right', handletextpad=0.2, fontsize=20) 
    sub.set_xlabel(r'$A_V$', fontsize=25) 
    sub.set_ylabel(r'$p(A_V)$', fontsize=25) 
    sub.set_xlim(-0.2, 5.) 
    sub.set_ylim(0., 2.) 
    
    ffig = os.path.join(fig_dir, 'slab_tnorm.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_corner(): 
    ''' example corner plot of DEM parameters
    '''
    import abcpmc
    # update these values as I see fit
    name = 'tng.slab_noll_msfr.L2.3d'
    T = 9 

    dat_dir = os.environ['GALPOPFM_DIR']
    abc_dir = os.path.join(dat_dir, 'abc', name) 
    # read pool 
    theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
    rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
    w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
    
    prior_min = np.array([-5., -5., 0., -4., -4., -4., -4., 0., 1.]) 
    prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    prior_range = [(_min, _max) for _min, _max in zip(prior_min, prior_max)]
        
    lbls = [r'$m_{\tau,1}$', r'$m_{\tau,2}$', r'$c_{\tau}$', 
            r'$m_{\delta,1}$', r'$m_{\delta,2}$', r'$c_\delta$',
            r'$m_E$', r'$c_E$', r'$f_{\rm neb}$'] 

    fig = DFM.corner(
            theta_T, 
            range=prior_range,
            weights=w_T,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95],
            nbin=20, 
            smooth=True, 
            color='C0', 
            labels=lbls, 
            label_kwargs={'fontsize': 25}) 

    ffig = os.path.join(fig_dir, 'abc.png')
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _ABC_Observables(): 
    ''' new idea for Figure presenting the observables of slab model ABC posteriors  
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(name='sdss',
            statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.5), (-1., 4.)]

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.)

    x_obs = [-1.*sdss['mr_tinker'][...][mr_complete], 
            sdss['mg_tinker'][...][mr_complete] - sdss['mr_tinker'][...][mr_complete], 
            sdss['ABSMAG'][...][:,0][mr_complete] - sdss['ABSMAG'][...][:,1][mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'simba.slab_noll_msfr.L2.3d', 'theta.t6.dat')) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'tng.slab_noll_msfr.L2.3d', 'theta.t4.dat')) 
    theta_tng = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba = _sim_observables('simba', theta_simba,
            zero_sfr_sample=True)
    x_tng, sfr0_tng = _sim_observables('tng', theta_tng,
            zero_sfr_sample=True)

    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_obs, x_simba, x_tng]
    names   = ['SDSS', 'SIMBA (w/ DEM)', 'TNG (w/ DEM)']
    clrs    = ['k', 'C1', 'C0']
    sfr0s   = [sfr0_obs, sfr0_simba, sfr0_tng] 


    obs_lims = [(20, 22.5), (-0.05, 1.5), (-1., 4)]
    obs_lbls = [r'$M_r$ luminosity', '$G - R$', '$FUV - NUV$']

    fig = plt.figure(figsize=(12, 12))
    
    for i in range(3): 
        for j in range(3): 
            sub = fig.add_subplot(3,3,3*j+i+1)
            if i > j:
                # compare SIMBA
                DFM.hist2d(x_obs[i], x_obs[j], levels=[0.68, 0.95],
                        range=[ranges[i], ranges[j]], bins=20, color='k', 
                        plot_datapoints=False, fill_contours=False,
                        plot_density=False, linestyle='--', ax=sub)
                DFM.hist2d(x_simba[i], x_simba[j], levels=[0.68, 0.95],
                        range=[ranges[i], ranges[j]], bins=20, color='C1', 
                        plot_datapoints=False, fill_contours=False,
                        plot_density=True, ax=sub)
            elif i < j: 
                # compare SIMBA
                DFM.hist2d(x_obs[i], x_obs[j], levels=[0.68, 0.95],
                        range=[ranges[i], ranges[j]], bins=20, color='k', 
                        plot_datapoints=False, fill_contours=False,
                        plot_density=False, linestyle='--', ax=sub)
                DFM.hist2d(x_tng[i], x_tng[j], levels=[0.68, 0.95],
                        range=[ranges[i], ranges[j]], bins=20, color='C0', 
                        plot_datapoints=False, fill_contours=False,
                        plot_density=True, ax=sub)
            else: 
                if i == 0: 
                    mr_bin = np.linspace(20, 23, 7) 
                    dmr = mr_bin[1:] - mr_bin[:-1]

                    Ngal_simba, _ = np.histogram(x_simba[i], bins=mr_bin)
                    Ngal_tng, _ = np.histogram(x_tng[i], bins=mr_bin)

                    phi_simba   = Ngal_simba.astype(float) / vol_simba / dmr
                    phi_tng     = Ngal_tng.astype(float) / vol_tng / dmr

                    sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_simba, c='C1')
                    sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_tng, c='C0')

                    fsdss = os.path.join(dat_dir, 'obs',
                            'tinker_SDSS_centrals_M9.7.phi_Mr.dat') 
                    mr_low, mr_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)
                    sub.errorbar(-0.5*(mr_low + mr_high), phi_sdss, yerr=err_phi_sdss,
                            fmt='.k', label='SDSS Centrals')
                    sub.set_yscale('log') 
                    sub.set_ylim(5e-5, 1e-2) 
                    print('number densities')
                    print('simba: %.5e' % (np.sum(Ngal_simba.astype(float))/vol_simba))
                    print('tng: %.5e' % (np.sum(Ngal_tng.astype(float))/vol_tng))
                    print('sdss: %.5e' % (float(len(x_obs[i]))/vol_sdss))
                else: 
                    _ = sub.hist(x_simba[i][x_simba[0] > 20], 
                            weights=np.repeat(1./vol_simba, np.sum(x_simba[0] > 20)),
                            range=ranges[i], bins=20, color='C1', histtype='step') 
                    _ = sub.hist(x_tng[i][x_tng[0] > 20], 
                            weights=np.repeat(1./vol_tng, np.sum(x_tng[0] > 20)),
                            range=ranges[i], bins=20, color='C0', histtype='step') 
                    _ = sub.hist(x_obs[i],
                            weights=np.repeat(1./vol_sdss, len(x_obs[i])), 
                            range=ranges[i], bins=20, color='k',
                            linestyle='--', histtype='step') 

            sub.set_xlim(obs_lims[i])
            if i !=j: sub.set_ylim(obs_lims[j])

            if i == 0: sub.set_ylabel(obs_lbls[j], fontsize=25) 
            else: sub.set_yticklabels([]) 
            if j == 2: sub.set_xlabel(obs_lbls[i], fontsize=25) 
            else: sub.set_xticklabels([]) 

    ffig = os.path.join(fig_dir, '_abc_observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    #fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_Observables(): 
    ''' Figure presenting the observables along with simulations without any
    attenuation.
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(name='sdss',
            statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.5), (-1., 4.)]

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.)

    x_obs = [-1.*sdss['mr_tinker'][...][mr_complete], 
            sdss['mg_tinker'][...][mr_complete] - sdss['mr_tinker'][...][mr_complete], 
            sdss['ABSMAG'][...][:,0][mr_complete] - sdss['ABSMAG'][...][:,1][mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'simba.slab_noll_msfr.L2.3d', 'theta.t6.dat')) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'tng.slab_noll_msfr.L2.3d', 'theta.t4.dat')) 
    theta_tng = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba = _sim_observables('simba', theta_simba,
            zero_sfr_sample=True)
    x_tng, sfr0_tng = _sim_observables('tng', theta_tng,
            zero_sfr_sample=True)

    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_obs, x_simba, x_tng]
    names   = ['SDSS', 'SIMBA (w/ DEM)', 'TNG (w/ DEM)']
    clrs    = ['k', 'C1', 'C0']
    sfr0s   = [sfr0_obs, sfr0_simba, sfr0_tng] 

    fig = plt.figure(figsize=(5*len(xs),10))

    #for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
    for i, _x, _sfr0, name, clr in zip(range(len(xs)), xs, sfr0s, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        #sub.scatter(_x[0][_sfr0], _x[1][_sfr0], c='k', s=1)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1.])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        #sub.scatter(_x[0][_sfr0], _x[2][_sfr0], c='k', s=1)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()

    obs_lims = [(20, 22.5), (0., 1.5), (-0.5, 4)]
    obs_lbls = [r'$M_r$ luminosity', '$G - R$', '$FUV - NUV$']
    yobs_lbls = [r'$\Phi^{\rm cen}_{M_r}$', '$p(G - R)$', '$p(FUV - NUV)$']

    fig = plt.figure(figsize=(15,4))
    for i in range(3):
        sub = fig.add_subplot(1,3,i+1)
        
        if i == 0: 
            mr_bin = np.linspace(20, 23, 7) 
            dmr = mr_bin[1:] - mr_bin[:-1]

            Ngal_simba, _ = np.histogram(x_simba[i], bins=mr_bin)
            Ngal_tng, _ = np.histogram(x_tng[i], bins=mr_bin)

            phi_simba   = Ngal_simba.astype(float) / vol_simba / dmr
            phi_tng     = Ngal_tng.astype(float) / vol_tng / dmr

            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_simba, c='C1')
            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_tng, c='C0')

            fsdss = os.path.join(dat_dir, 'obs',
                    'tinker_SDSS_centrals_M9.7.phi_Mr.dat') 
            mr_low, mr_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)
            sub.errorbar(-0.5*(mr_low + mr_high), phi_sdss, yerr=err_phi_sdss,
                    fmt='.k', label='SDSS Centrals')
            sub.set_yscale('log') 
            sub.set_ylim(5e-5, 8e-3) 
        else: 
            _ = sub.hist(x_simba[i], 
                    weights=np.repeat(1./vol_simba, len(x_simba[i])),
                    range=ranges[i], bins=20, color='C1', histtype='step') 
            _ = sub.hist(x_tng[i][x_tng[0] > 20], 
                    weights=np.repeat(1./vol_tng, len(x_tng[i])),
                    range=ranges[i], bins=20, color='C0', histtype='step') 
            _ = sub.hist(x_obs[i],
                    weights=np.repeat(1./vol_sdss, len(x_obs[i])), 
                    range=ranges[i], bins=20, color='k',
                    linestyle='--', histtype='step') 
    
        sub.set_xlabel(obs_lbls[i], fontsize=20) 
        sub.set_xlim(obs_lims[i]) 
        sub.set_ylabel(yobs_lbls[i], fontsize=20)
        
    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(fig_dir, 'abc_observables.1d.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_tnorm_Observables(): 
    ''' figure presenting the ABC posterior observables for tnorm models 
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(name='sdss',
            statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.5), (-1., 4.)]

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.)

    x_obs = [-1.*sdss['mr_tinker'][...][mr_complete], 
            sdss['mg_tinker'][...][mr_complete] - sdss['mr_tinker'][...][mr_complete], 
            sdss['ABSMAG'][...][:,0][mr_complete] - sdss['ABSMAG'][...][:,1][mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'simba.tnorm_noll_msfr.L2.3d', 'theta.t6.dat')) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'tng.tnorm_noll_msfr.L2.3d', 'theta.t5.dat')) 
    theta_tng = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba = _sim_observables('simba', theta_simba,
            model='tnorm', zero_sfr_sample=True)
    x_tng, sfr0_tng = _sim_observables('tng', theta_tng,
            model='tnorm', zero_sfr_sample=True)
    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_obs, x_simba, x_tng]
    names   = ['SDSS', 'SIMBA (w/ DEM)', 'TNG (w/ DEM)']
    clrs    = ['k', 'C1', 'C0']
    sfr0s   = [sfr0_obs, sfr0_simba, sfr0_tng] 

    fig = plt.figure(figsize=(5*len(xs),10))

    #for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
    for i, _x, _sfr0, name, clr in zip(range(len(xs)), xs, sfr0s, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        #sub.scatter(_x[0][_sfr0], _x[1][_sfr0], c='k', s=1)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1.])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        #sub.scatter(_x[0][_sfr0], _x[2][_sfr0], c='k', s=1)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_tnorm_observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    
    obs_lims = [(20, 22.5), (0., 1.5), (-0.5, 4)]
    obs_lbls = [r'$M_r$ luminosity', '$G - R$', '$FUV - NUV$']
    yobs_lbls = [r'$\Phi^{\rm cen}_{M_r}$', '$p(G - R)$', '$p(FUV - NUV)$']

    fig = plt.figure(figsize=(15,4))
    for i in range(3):
        sub = fig.add_subplot(1,3,i+1)
        
        if i == 0: 
            mr_bin = np.linspace(20, 23, 7) 
            dmr = mr_bin[1:] - mr_bin[:-1]

            Ngal_simba, _ = np.histogram(x_simba[i], bins=mr_bin)
            Ngal_tng, _ = np.histogram(x_tng[i], bins=mr_bin)

            phi_simba   = Ngal_simba.astype(float) / vol_simba / dmr
            phi_tng     = Ngal_tng.astype(float) / vol_tng / dmr

            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_simba, c='C1')
            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_tng, c='C0')

            fsdss = os.path.join(dat_dir, 'obs',
                    'tinker_SDSS_centrals_M9.7.phi_Mr.dat') 
            mr_low, mr_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)
            sub.errorbar(-0.5*(mr_low + mr_high), phi_sdss, yerr=err_phi_sdss,
                    fmt='.k', label='SDSS Centrals')
            sub.set_yscale('log') 
            sub.set_ylim(5e-5, 8e-3) 
        else: 
            _ = sub.hist(x_simba[i], 
                    weights=np.repeat(1./vol_simba, len(x_simba[i])),
                    range=ranges[i], bins=20, color='C1', histtype='step') 
            _ = sub.hist(x_tng[i][x_tng[0] > 20], 
                    weights=np.repeat(1./vol_tng, len(x_tng[i])),
                    range=ranges[i], bins=20, color='C0', histtype='step') 
            _ = sub.hist(x_obs[i],
                    weights=np.repeat(1./vol_sdss, len(x_obs[i])), 
                    range=ranges[i], bins=20, color='k',
                    linestyle='--', histtype='step') 
    
        sub.set_xlabel(obs_lbls[i], fontsize=20) 
        sub.set_xlim(obs_lims[i]) 
        sub.set_ylabel(yobs_lbls[i], fontsize=20)
        
    fig.subplots_adjust(wspace=0.6)
    ffig = os.path.join(fig_dir, 'abc_tnorm_observables.1d.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def fig_tex(ffig, pdf=False):
    ''' given filename of figure return a latex friendly file name
    '''
    path, ffig_base = os.path.split(ffig)
    ext = ffig_base.rsplit('.', 1)[-1]
    ffig_name = ffig_base.rsplit('.', 1)[0]

    _ffig_name = ffig_name.replace('.', '_')
    if pdf: ext = 'pdf'
    return os.path.join(path, '.'.join([_ffig_name, ext]))


if __name__=="__main__": 
    #SDSS()
    #SMFs() 
    #DEM()
    #Observables()
    slab_tnorm_comparison()
    #ABC_corner() 
    #_ABC_Observables()
    #ABC_Observables()
    #ABC_tnorm_Observables()
