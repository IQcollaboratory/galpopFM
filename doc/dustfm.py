#!/bin/python
'''

script for generating plots for the paper 


'''
import os 
import sys 
import h5py 
import numpy as np 
import corner as DFM 
from astrologs.astrologs import Astrologs
# -- galpopfm --
from galpopfm.catalogs import Catalog
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

sims = ['SIMBA', 'TNG', 'EAGLE']                    # simulations 
clrs = {'simba': 'C1', 'tng': 'C0', 'eagle': 'C2'}  # colors
nabc = {'simba': 13, 'tng': 25, 'eagle': 27}        # Niteration 

sfr0_prescript = 'sfrmin'                           # prescription of SFR=0 
dem = 'slab_noll_mssfr_fixbump'
abc_run = lambda _sim: \
        '%s.%s.L2_only.3d.%s' % (_sim.lower(), dem, sfr0_prescript)
dem_attenuate = dustFM.DEM_slab_noll_mssfr_fixbump  # dustFM function 

# parameters
param_lbls = np.array([
        r'$m_{\tau,M_*}$', r'$m_{\tau,{\rm SSFR}}$', r'$c_{\tau}$', 
        r'$m_{\delta,M_*}$', r'$m_{\delta,{\rm SSFR}}$', r'$c_\delta$'])
prior_min = np.array([-5., -5., 0., -4., -4., -4.])
prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0])

prior_range = np.array([(_min, _max) for _min, _max in zip(prior_min, prior_max)])


def SDSS():
    ''' figure illustrating our SDSS 
    '''
    # read in SDSS 
    tinker = Astrologs('tinkergroup', mlim='9.7') 

    # get M_r and log M* 
    R_mag = tinker.data['M_r']
    logms = tinker.data['log.M_star'] 
    print('%i of %i are centrals' % (np.sum(tinker.data['iscen']),
        len(tinker.data['iscen'])))

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


def NSA():
    ''' figure illustrating our SDSS 
    '''
    # read in SDSS 
    nsa = Astrologs('nsa') 

    # get M_r and log M* 
    R_mag   = nsa.data['M_r']
    logms   = np.log10(nsa.data['M_star'] / 0.7**2) 
    z       = nsa.data['redshift'] 

    Rlim = (R_mag < -19.5) 

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)

    sub.scatter(logms, R_mag, c='k', s=1, label='NASA-Sloan Atlas')
    sub.scatter(logms[Rlim], R_mag[Rlim], c='C1', s=1, label='$M_r < -19.5$') 
    
    sub.legend(loc='upper left', handletextpad=0.1, markerscale=10, fontsize=25) 
    sub.set_xlabel(r'$\log(\,M_*$ [$M_\odot$]$)$', fontsize=25) 
    sub.set_xlim(8.5, 12.)

    sub.set_ylabel(r'$M_r$', fontsize=25) 
    sub.set_ylim(-17., -23.4) 

    ffig = os.path.join(fig_dir, 'nsa.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()

    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(111)

    sub.scatter(z, R_mag, c='k', s=1, label='NASA-Sloan Atlas')
    
    sub.legend(loc='upper left', handletextpad=0.1, markerscale=10, fontsize=25) 
    sub.set_xlabel(r'redshift', fontsize=25) 
    sub.set_xlim(0., 0.07)
    sub.set_ylabel(r'$M_r$', fontsize=25) 
    sub.set_ylim(-17., -23.4) 

    ffig = os.path.join(fig_dir, 'nsa_z.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def SMF_MsSFR(): 
    ''' figure comparing the stellar mass fucntions of the different simulations
    and M*-SFR relation 
    '''
    #########################################################################
    # read sdss and sims  
    #########################################################################
    sdss = Catalog('tinker') 
    sdss_ms = sdss.data['log.M_star']
    sdss_sfr = sdss_ms + sdss.data['log.ssfr'] 

    simba = Catalog('simba') 
    simba_ms  = simba.data['logmstar']
    simba_sfr = simba.data['logsfr.inst']

    tng = Catalog('tng')  
    tng_ms  = tng.data['logmstar']
    tng_sfr = tng.data['logsfr.inst']

    eag = Catalog('eagle')
    eag_ms  = eag.data['logmstar']
    eag_sfr = eag.data['logsfr.inst']
    #########################################################################
    # calculate SMFs
    #########################################################################
    logms_bin = np.linspace(8., 13., 21)
    dlogms = logms_bin[1:] - logms_bin[:-1]
    
    Ngal_simba, _   = np.histogram(simba_ms, bins=logms_bin)
    Ngal_tng, _     = np.histogram(tng_ms, bins=logms_bin)
    Ngal_eag, _     = np.histogram(eag_ms, bins=logms_bin)
    
    phi_simba   = Ngal_simba.astype(float) / simba.cosmic_volume / dlogms
    phi_tng     = Ngal_tng.astype(float) / tng.cosmic_volume / dlogms
    phi_eag     = Ngal_eag.astype(float) / eag.cosmic_volume / dlogms
    # read sdss smf 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker.smf.dat') 
    logms_low, logms_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)
    #########################################################################
    # plot SMFs
    #########################################################################
    fig = plt.figure(figsize=(20,5))
    outer = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 3], figure=fig) 
    #make nested gridspecs
    gs1 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
    gs2 = mpl.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
            wspace=0.05)

    sub = plt.subplot(gs1[0])
    sub.errorbar(0.5*(logms_low + logms_high)[logms_low > 9.7], phi_sdss[logms_low > 9.7], 
            yerr=err_phi_sdss[logms_low > 9.7],	fmt='.k', label='SDSS')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_simba, c='C0',
            label='SIMBA')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_tng, c='C1', 
            label='TNG')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_eag, c='C2', 
            label='EAGLE')
    sub.legend(loc='lower left', handletextpad=0.3, fontsize=20)
    sub.set_xlabel(r'log( $M_*$ [$M_\odot$] )', labelpad=5, fontsize=25)
    sub.set_xlim(9., 12.5)
    sub.set_xticks([9., 10., 11., 12.]) 
    sub.set_ylabel(r'$\Phi_{M_*}$ [$({\rm Mpc}/h)^{-3}{\rm dex}^{-1}$]', fontsize=24)
    sub.set_yscale("log")
    sub.set_ylim(5e-6, 1e-1) 
    
    names   = ['SIMBA', 'TNG', 'EAGLE']
    clrs    = ['C1', 'C0', 'C2']
    ms      = [simba_ms, tng_ms, eag_ms]
    sfrs    = [simba_sfr, tng_sfr, eag_sfr]

    for i in range(len(names)): 
        sub = plt.subplot(gs2[i]) 
        DFM.hist2d(sdss_ms, sdss_sfr, levels=[0.68, 0.95],	
                range=[[7.8, 12.], [-4., 2.]], color='k', 	
                contour_kwargs={'linewidths': 0.75, 'linestyles': 'dashed'}, 	
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)


        DFM.hist2d(ms[i], sfrs[i], color=clrs[i], 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        sub.text(0.05, 0.95, names[i], transform=sub.transAxes, ha='left', va='top', fontsize=25) 
        sub.set_xlim([9., 11.5]) 
        sub.set_ylim([-3., 2.]) 
        if i != 0: sub.set_yticklabels([]) 
        else: sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
        if i == 1: sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
        sub.set_xticks([9., 10., 11.]) 
        #sub.set_xticklabels([9., 10., 11.]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.2,
            fontsize=20) 

    fig.subplots_adjust(wspace=0.2, hspace=0.1)

    ffig = os.path.join(fig_dir, 'smf_m_sfr.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def DEM(): 
    ''' comparison of DEM attenuation curve to standard attenuation curves in
    the literature.

    todo: 
    * compile the following attenuation curves: 
        * Cardelli+(1989) MW
        * Wild+(2011)
        * Kriek & Conroy (2013)
        * Reddy+(2015)    
    '''
    _dem = lambda lam, logm, logsfr: -2.5 * np.log10(dem_attenuate(
        np.array([2., -2., 2., -0.1, -0.1, -0.2, 1.]), 
        lam, 
        np.ones(len(lam)), 
        logm, 
        logsfr, 
        incl=0.,
        nebular=False)).flatten()

    wave = np.linspace(1000, 10000, 1000) 

    fig = plt.figure(figsize=(8,5))
    
    logSFR_sf = 0.5
    logSFR_q = -2.
    M_low = 10.
    M_high = 11.

    sub = fig.add_subplot(111) 
    # low mass SFing galaxies
    sub.plot(wave, _dem(wave, M_low, logSFR_sf), c='C0',  
            label=r'star-forming, low $M_*$')
    # high mass SFing galaxies
    sub.plot(wave, _dem(wave, M_high, logSFR_sf), c='C2',  
            label=r'star-forming, high $M_*$')
    # low mass Quiescent galaxies
    sub.plot(wave, _dem(wave, M_low, logSFR_q), c='C1',  
            label=r'quiescent, low $M_*$')
    # high mass Quiescent galaxies
    sub.plot(wave, _dem(wave, M_high, logSFR_q), c='C3',  
            label=r'quiescent, high $M_*$')
    # calzetti for reference
    sub.plot(wave, dustFM.calzetti_absorption(wave), c='k', 
            ls='--', label='Calzetti+(2001)') 
    sub.set_xlim(1.2e3, 1e4)
    sub.set_ylim(0., 7.) 
    sub.legend(loc='upper right', handletextpad=0.2, fontsize=20) 

    sub.set_xlabel(r'Wavelength [$\AA$]', labelpad=5, fontsize=20) 
    sub.set_ylabel(r'$A(\lambda)$', labelpad=10, fontsize=25) 

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
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d',
            return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    nontheta = np.zeros(6) 
    x_simba, _, sfr0_simba  = _sim_observables('simba', nontheta)
    x_tng, _, sfr0_tng      = _sim_observables('tng', nontheta)
    x_eagle, _, sfr0_eagle  = _sim_observables('eagle', nontheta)
    sfr0s = [sfr0_simba, sfr0_tng, sfr0_eagle]
    #########################################################################
    # plotting 
    #########################################################################
    fig = plt.figure(figsize=(5*len(sims),10))
    for i, _x, sim in zip(range(len(sims)), [x_simba, x_tng, x_eagle], sims): 

        # R vs (G - R)
        sub = fig.add_subplot(2,len(sims),i+1)
        DFM.hist2d(x_obs[0], x_obs[1], range=[ranges[0], ranges[1]],
                levels=[0.68, 0.95], bins=20, color='k',
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
        DFM.hist2d(_x[0], _x[1], range=[ranges[0], ranges[1]],
                levels=[0.68, 0.95], bins=20, color=clrs[sim.lower()], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub)
        #sub.scatter(_x[0][sfr0s[i]], _x[1][sfr0s[i]], c='k', s=1) 
        sub.text(0.95, 0.95, '%s (no dust)' % sim, 
            ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: sub.set_ylabel(r'$G-R$', fontsize=25) 
        else: sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1., 1.5])


        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(sims),i+len(sims)+1)
        DFM.hist2d(x_obs[0], x_obs[2], range=[ranges[0], ranges[2]],
            levels=[0.68, 0.95], bins=20, color='k', 
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub)
        DFM.hist2d(_x[0], _x[2], range=[ranges[0], ranges[2]],
            levels=[0.68, 0.95], bins=20, color=clrs[sim.lower()], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
        #sub.scatter(_x[0][sfr0s[i]], _x[2][sfr0s[i]], c='k', s=1) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: sub.set_ylabel(r'$FUV - NUV$', fontsize=25) 
        else: sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 


    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', ncol=2, handletextpad=0,
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


def _sim_observables(sim, theta, model='slab', noise=True,
        sfr0_prescription=sfr0_prescript, no_Mr_cut=False): 
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

    mlim    = np.ones(len(_sim_sed['logmstar'])).astype(bool)#(_sim_sed['logmstar'] > 9.4) 
    zerosfr = (_sim_sed['logsfr.inst'] == -999)
    cuts    = mlim 

    sim_sed = {} 
    sim_sed['sim']          = sim 
    sim_sed['logmstar']     = _sim_sed['logmstar'][cuts].copy()
    sim_sed['logsfr.inst']  = _sim_sed['logsfr.inst'][cuts].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 
    
    x_mod = dustInfer.sumstat_model(
            theta, 
            sed=sim_sed, 
            dem=dem,
            statistic='2d', 
            sfr0_prescription=sfr0_prescription, 
            noise=noise, 
            return_datavector=True) 
    
    # impose Mr cut (or not) 
    if not no_Mr_cut: mr_cut = x_mod[0] > 20
    else: mr_cut = np.ones(len(x_mod[0])).astype(bool) 

    sim_sed['logmstar'] = sim_sed['logmstar'][mr_cut] 
    sim_sed['logsfr.inst'] = sim_sed['logsfr.inst'][mr_cut] 

    return x_mod[:,mr_cut], sim_sed, zerosfr[cuts][mr_cut]


def slab_model(): 
    ''' figure comparing the A_V distributions of the slab model to the
    observed SDSS distribution 
    '''
    from pydl.pydlutils.spheregroup import spherematch
    # MPA-JHU Av from SED fitting 
    mpajhu_av, gal_type, mpajhu_ra, mpajhu_dec  = np.loadtxt(os.path.join(dat_dir, 'obs', 'SDSS_Av.txt'),
            unpack=True, usecols=[1, 2, -2, -1]) 
    starforming = (gal_type == 1) 
    mpajhu_av = mpajhu_av[starforming]
    mpajhu_ra = mpajhu_ra[starforming] 
    mpajhu_dec = mpajhu_dec[starforming]

    # read in SDSS (Jeremy's catalog) 
    tinker = Astrologs('tinkergroup', mlim='9.7') 
    
    R_mag   = tinker.data['M_r']
    ra      = tinker.data['ra'] 
    dec     = tinker.data['dec'] 

         
    Rlim = (R_mag < -20.) & (ra != -999.) & (dec != -999.)
    ra      = ra[Rlim] 
    dec     = dec[Rlim] 

    # match MPAJHU star forming AV to SDSS  
    match = spherematch(ra, dec, mpajhu_ra, mpajhu_dec, 0.000277778)
    m_sdss = match[0] 
    m_mpajhu = match[1] 
    print('%i matches out of %i, %i SDSS R < -20.' % (len(m_sdss), np.sum(Rlim), np.sum(R_mag < -20.))) 

    logms   = tinker.data['log.M_star'][Rlim]#[m_sdss]
    logssfr = tinker.data['log.ssfr'][Rlim]#[m_sdss]

    #Eq. 14 of Somerville+(1999) 
    # randomly sample cos(inclinatiion) from 0 - 1
    cos_i = np.random.uniform(0., 1., size=len(logms))
    sec_incl = 1./cos_i

    #[-0.14806716 -0.55586142  1.8306104  -0.44204573 -0.56640125  0.0114204 ]

    tauV = np.clip(-0.3 * (logms - 10.) - 0.6 * (logssfr + 10.) + 0.7, 1e-3, None)
    slab_AV = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111) 
    _ = sub.hist(mpajhu_av[m_mpajhu], range=(-3., 7), bins=51, density=True, 
            color='C0', linestyle='-', histtype='stepfilled', label='SDSS')
    _ = sub.hist(np.array(slab_AV), range=(-3., 7), bins=51, density=True, 
            color='k', linestyle='-', linewidth=2, histtype='step', 
            label=r'slab model')
    sub.legend(loc='upper right', handletextpad=0.3, fontsize=20) 
    sub.set_xlabel(r'$A_V$', fontsize=25) 
    sub.set_ylabel(r'$p(A_V)$', fontsize=25) 
    sub.set_xlim(-0.2, 5.) 
    sub.set_ylim(0., 1.) 
    
    ffig = os.path.join(fig_dir, 'slab_model.png')
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def slab_tnorm_comparison(): 
    ''' figure comparing the A_V distributions of the slab model, tnorm and
    observed SDSS. 
    '''
    from scipy.stats import truncnorm
    from pydl.pydlutils.spheregroup import spherematch
    # MPA-JHU Av from SED fitting 
    mpajhu_av, gal_type, mpajhu_ra, mpajhu_dec  = np.loadtxt(os.path.join(dat_dir, 'obs', 'SDSS_Av.txt'),
            unpack=True, usecols=[1, 2, -2, -1]) 
    starforming = (gal_type == 1) 
    mpajhu_av = mpajhu_av[starforming]
    mpajhu_ra = mpajhu_ra[starforming] 
    mpajhu_dec = mpajhu_dec[starforming]

    # read in SDSS (Jeremy's catalog) 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5')
    sdss = h5py.File(fsdss, 'r')
    
    R_mag   = sdss['mr_tinker'][...]
    
    Rlim = (R_mag < -20.) & (sdss['RA'][...] != -999.) & (sdss['DEC'][...] != -999.)
    ra      = sdss['RA'][...][Rlim] 
    dec     = sdss['DEC'][...][Rlim] 

    # match MPAJHU star forming AV to SDSS  
    match = spherematch(ra, dec, mpajhu_ra, mpajhu_dec, 0.000277778)
    m_sdss = match[0] 
    m_mpajhu = match[1] 
    print('%i matches out of %i, %i SDSS R < -20.' % (len(m_sdss), np.sum(Rlim), np.sum(R_mag < -20.))) 

    logms   = np.log10(sdss['ms_tinker'][...])[Rlim][m_sdss]
    logsfr  = sdss['sfr_tinker'][...][Rlim][m_sdss] + logms

    #Eq. 14 of Somerville+(1999) 
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=len(m_sdss))#np.sum(Rlim))
    sec_incl = 1./np.cos(incl) 

    tauV = np.clip(2. * (logms - 10.) + 1 * logsfr + 0.15, 1e-3, None) 
    slab_AV = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl))

    # tnorm AV 
    mu_Av   = np.clip(-1*(logms - 10.) + 1.75, 0., None) 
    sig_Av  = np.clip(-0.5 * (logms - 10.) + 0.75, 0.1, None) # can't be too narrow

    tnorm_AV = [truncnorm.rvs((0. - _mu_Av)/_sig_Av, np.inf, loc=_mu_Av,
            scale=_sig_Av)  for _mu_Av, _sig_Av in zip(mu_Av, sig_Av)]

    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111) 
    _ = sub.hist(mpajhu_av[m_mpajhu], range=(-3., 7), bins=51, density=True, 
            color='C0', linestyle='-', histtype='stepfilled', label='SDSS')
    _ = sub.hist(np.array(slab_AV), range=(-3., 7), bins=51, density=True, 
            color='k', linestyle='-', linewidth=2, histtype='step', 
            label=r'slab model')
    _ = sub.hist(np.array(tnorm_AV), range=(-3., 7), bins=51, density=True, 
            color='C1', linestyle='--', linewidth=2, histtype='step', 
            label=r'$\mathcal{N}_T$ model')
    sub.legend(loc='upper right', handletextpad=0.3, fontsize=20) 
    sub.set_xlabel(r'$A_V$', fontsize=25) 
    sub.set_ylabel(r'$p(A_V)$', fontsize=25) 
    sub.set_xlim(-0.2, 5.) 
    sub.set_ylim(0., 1.) 
    
    ffig = os.path.join(fig_dir, 'slab_tnorm.png')
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_corner(): 
    ''' example corner plot of DEM parameters
    '''
    import abcpmc
    # parameters of interest
    keep_cols = np.ones(len(param_lbls)).astype(bool) 
    #keep_cols[:-1] = True

    print('\t %s' % ','.join(param_lbls[keep_cols]))
    for i, sim in enumerate(sims):
        dat_dir = os.environ['GALPOPFM_DIR']
        abc_dir = os.path.join(dat_dir, 'abc', abc_run(sim.lower())) 
        T = nabc[sim.lower()]

        # read pool 
        theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
        rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
        w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
        
        if i == 0: 
            fig = DFM.corner(
                    theta_T[:,keep_cols], 
                    range=prior_range[keep_cols],
                    weights=w_T,# quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color=clrs[sim.lower()], 
                    labels=param_lbls[keep_cols], 
                    label_kwargs={'fontsize': 25}) 
        else: 
            DFM.corner(
                    theta_T[:,keep_cols], 
                    range=prior_range[keep_cols],
                    weights=w_T, #quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color=clrs[sim.lower()], 
                    labels=param_lbls[keep_cols], 
                    label_kwargs={'fontsize': 25}, 
                    fig = fig) 
        median = np.quantile(theta_T, [0.5], axis=0)[0]
        m1sig = median - np.quantile(theta_T, [0.16], axis=0)[0]
        p1sig = np.quantile(theta_T, [0.84], axis=0)[0] - median
        print('%s: %s' % (sim, '\t'.join(['%.2f +%.2f -%.2f' % (med, m1, p1) for
            med, m1, p1 in zip(median, m1sig, p1sig)])))
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, sim in enumerate(sims): 
        bkgd.fill_between([],[],[], color=clrs[sim.lower()], label=sim) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)

    ffig = os.path.join(fig_dir, 'abc.png')
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _ABC_corner_flexbump(): 
    ''' example corner plot of DEM parameters
    '''
    import abcpmc
    # update these values as I see fit
    sims = ['SIMBA', 'TNG', 'EAGLE']
    names = ['simba.slab_noll_msfr.L2.3d', 'tng.slab_noll_msfr.L2.3d', 'eagle.slab_noll_msfr.L2.3d']
    Ts = [5, 5, 5]
        
    prior_min = np.array([-5., -5., 0., -4., -4., -4., -4., 0., 1.]) 
    prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0, 0.0, 4., 4.]) 
    prior_range = [(_min, _max) for _min, _max in zip(prior_min, prior_max)]
        
    lbls = [r'$m_{\tau,1}$', r'$m_{\tau,2}$', r'$c_{\tau}$', 
            r'$m_{\delta,1}$', r'$m_{\delta,2}$', r'$c_\delta$',
            r'$m_{E}$', r'$c_E$',
            r'$f_{\rm neb}$'] 

    for i, name, T in zip(range(len(names)), names, Ts):
        dat_dir = os.environ['GALPOPFM_DIR']
        abc_dir = os.path.join(dat_dir, 'abc', name) 

        # read pool 
        theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
        rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
        w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
        
        # we have no constraints on m_E, c_E, and fneb so we ignore 
        keep_cols = np.zeros(len(lbls)).astype(bool) 
        keep_cols[:6] = True
            
        if i == 0: 
            fig = DFM.corner(
                    theta_T[:,keep_cols], 
                    range=np.array(prior_range)[keep_cols],
                    weights=w_T,# quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color='C%i' % i, 
                    labels=np.array(lbls)[keep_cols], 
                    label_kwargs={'fontsize': 25}) 
        else: 
            DFM.corner(
                    theta_T[:,keep_cols], 
                    range=np.array(prior_range)[keep_cols],
                    weights=w_T, #quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color='C%i' % i, 
                    labels=np.array(lbls)[keep_cols], 
                    label_kwargs={'fontsize': 25}, 
                    fig = fig) 
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, sim in enumerate(sims): 
        bkgd.fill_between([],[],[], color='C%i' % i, label=sim) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)

    ffig = os.path.join(fig_dir, 'abc.flexbump.png')
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_tnorm_corner(): 
    ''' example corner plot of DEM parameters
    '''
    import abcpmc
    # update these values as I see fit
    sims = ['SIMBA', 'TNG', 'EAGLE']
    clrs = ['C1', 'C0', 'C2']
    names = ['simba.tnorm_noll_msfr_fixbump.L2.3d', 
            'tng.tnorm_noll_msfr_fixbump.L2.3d', 
            'eagle.tnorm_noll_msfr_fixbump.L2.3d']
    Ts = [6, 6, 6]
    
    # priors 
    prior_min = np.array([-5., -5., 0., -5., -5., 0.1, -4., -4., -4., 1.]) 
    prior_max = np.array([5.0, 5.0, 6., 5.0, 5.0, 3., 4.0, 4.0, 4.0, 4.]) 
    prior_range = [(_min, _max) for _min, _max in zip(prior_min, prior_max)]

    lbls = [r'$m_{\mu,1}$', r'$m_{\mu,2}$', r'$c_{\mu}$', 
            r'$m_{\sigma,1}$', r'$m_{\sigma,2}$', r'$c_{\sigma}$', 
            r'$m_{\delta,1}$', r'$m_{\delta,2}$', r'$c_\delta$',
            r'$f_{\rm neb}$'] 

    for i, name, T in zip(range(len(names)), names, Ts):
        dat_dir = os.environ['GALPOPFM_DIR']
        abc_dir = os.path.join(dat_dir, 'abc', name) 

        # read pool 
        theta_T = np.loadtxt(os.path.join(abc_dir, 'theta.t%i.dat' % T)) 
        rho_T   = np.loadtxt(os.path.join(abc_dir, 'rho.t%i.dat' % T)) 
        w_T     = np.loadtxt(os.path.join(abc_dir, 'w.t%i.dat' % T)) 
        
        # we have no constraints on m_E, c_E, and fneb so we ignore 
        keep_cols = np.zeros(len(lbls)).astype(bool) 
        keep_cols[:9] = True
            
        if i == 0: 
            fig = DFM.corner(
                    theta_T[:,keep_cols], 
                    range=np.array(prior_range)[keep_cols],
                    weights=w_T,# quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color=clrs[i], 
                    labels=np.array(lbls)[keep_cols], 
                    label_kwargs={'fontsize': 25}) 
        else: 
            DFM.corner(
                    theta_T[:,keep_cols], 
                    range=np.array(prior_range)[keep_cols],
                    weights=w_T, #quantiles=[0.16, 0.5, 0.84], 
                    levels=[0.68, 0.95],
                    nbin=20, 
                    smooth=True, 
                    color=clrs[i], 
                    labels=np.array(lbls)[keep_cols], 
                    label_kwargs={'fontsize': 25}, 
                    fig = fig) 
    
    # legend
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, sim in enumerate(sims): 
        bkgd.fill_between([],[],[], color=clrs[i], label=sim) 
    bkgd.legend(loc='upper right', bbox_to_anchor=(0.875, 0.775), fontsize=25)

    ffig = os.path.join(fig_dir, 'abc_tnorm.png')
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
    ''' Figure the color-magnitude relation of simulations with DEM to SDSS
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    xs, sfr0s = [], [] 
    for sim in sims: 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 

        x_mod, _, sfr0 = _sim_observables(sim.lower(), theta_med)
        xs.append(x_mod) 
        sfr0s.append(sfr0) 
    #########################################################################
    # plotting 
    #########################################################################

    fig = plt.figure(figsize=(5*len(sims),10))
    for i, sim in enumerate(sims): 

        # R vs (G - R)
        sub = fig.add_subplot(2,len(sims),i+1)
        DFM.hist2d(x_obs[0], x_obs[1],range=[ranges[0], ranges[1]],
                levels=[0.68, 0.95], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
        DFM.hist2d(xs[i][0], xs[i][1], range=[ranges[0], ranges[1]],
                levels=[0.68, 0.95], bins=20, color=clrs[sim.lower()], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub)
        #sub.scatter(_x[0][_sfr0], _x[1][_sfr0], c='k', s=1)
        sub.text(0.95, 0.95, '%s + EDA' % sim, 
                ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: sub.set_ylabel(r'$G-R$', fontsize=20) 
        else: sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1., 1.5])


        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(sims),i+len(sims)+1)
        DFM.hist2d(x_obs[0], x_obs[2], range=[ranges[0], ranges[2]], 
                levels=[0.68, 0.95], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
        DFM.hist2d(xs[i][0], xs[i][2], range=[ranges[0], ranges[2]], 
                levels=[0.68, 0.95], bins=20, color=clrs[sim.lower()], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        #sub.scatter(_x[0][_sfr0], _x[2][_sfr0], c='k', s=1)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.1,
            fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_slope_AV(gal_type='all'): 
    ''' comparison of A_V-S relation 
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25 
    i3000 = 100
    i5500 = 225

    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111) 

    for i, sim in enumerate(['TNG', 'EAGLE']): 

        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, _sim, _sfr0 = _sim_observables(sim.lower(), theta_median)
        
        if gal_type == 'all': # all galaxies 
            subpop = np.ones(len(_sim['logmstar'])).astype(bool) 
        elif gal_type == 'starforming':
            # star-forming 
            subpop = (_sim['logsfr.inst'] - _sim['logmstar'] > -11)

        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst']
                )
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        A_V = A_lambda[:,i5500]
        S = A_lambda[:,i1500]/A_V
        

        DFM.hist2d(A_V[subpop], S[subpop], range=[(0., 1.4), (0., 14.4)], 
                levels=[0.68, 0.95], bins=10, color=clrs[sim.lower()],
                plot_datapoints=False, fill_contours=True, plot_density=False, 
                ax=sub)

    # MW
    sub.scatter([1.15], [2.8], c='k', marker='*', s=60) 
    sub.text(1.2, 2.8, 'MW', ha='left', va='bottom', fontsize=20) 
    # Calzetti 
    sub.plot([0.0, 1.4], [2.4, 2.4], c='k', ls='--')
    sub.text(0.125, 2.2, 'Calzetti+(2000)', ha='left', va='top', fontsize=15) 

    # Salim & Naryanan (2020) 
    _plt_obs = sub.fill_between(np.linspace(0., 1.4, 100), 
            10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424-0.12), 
            10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424+0.12), 
            color='k', alpha=0.25, linewidth=0)
    sub.set_xlabel(r'$A_V$', fontsize=25)
    sub.set_xlim(0.1, 1.4)
    sub.set_ylabel('$S = A_{1500}/A_V$', fontsize=25)
    sub.set_ylim(0., 14.4) 


    _plt_sims = [_plt_obs] 
    for sim in ['TNG', 'EAGLE']:   
        _plt_sim = sub.fill_between([], [], [], 
                color=clrs[sim.lower()], alpha=0.25, linewidth=0)
        _plt_sims.append(_plt_sim) 

    sub.legend(_plt_sims, ['Salim \& Narayanan(2020)', 'TNG', 'EAGLE'], 
        loc='upper right', handletextpad=0.1, fontsize=20) 


    ffig = os.path.join(fig_dir, 'abc_slope_AV.%s.png' % gal_type) 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_A_MsSFR(): 
    ''' A_1500 and A_V as a function of M_*-SFR heat map 
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25
    i5500 = 225

    # plotting 
    fig = plt.figure(figsize=(11,10))

    for i, sim in enumerate(['TNG', 'EAGLE']): 
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, _sim, _sfr0 = _sim_observables(sim.lower(), theta_median,
                no_Mr_cut=True)
        
        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst'])
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        cuts = (x_sim[0] > 20.)
    
        scs = [] 
        for ii, ilambda in enumerate([i1500, i5500]): 
            A = A_lambda[:,ilambda]

            sub = fig.add_subplot(2,2,2*ii+i+1)

            DFM.hist2d(_sim['logmstar'], _sim['logsfr.inst'], levels=[0.68, 0.95],	
                    range=[[9., 12.], [-4., 2.]], color='k', 	
                    contour_kwargs={'linewidths': 0.75, 'linestyles': 'dashed'}, 	
                    plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)

            scs.append(
                    sub.hexbin(_sim['logmstar'][cuts], _sim['logsfr.inst'][cuts], 
                        C=A[cuts], reduce_C_function=np.median, gridsize=15,
                        vmin=[2.0, 0.2][ii], vmax=[5., 1.4][ii], mincnt=10))
            
            if ii == 0: 
                sub.text(0.05, 0.95, sim, transform=sub.transAxes, ha='left', va='top', fontsize=25) 
            sub.set_xlim([9.5, 12.]) 
            if ii == 0: sub.set_xticklabels([])
            sub.set_ylim([-2.5, 2.]) 
            if i != 0: sub.set_yticklabels([]) 
            sub.set_xticks([10., 11., 12.]) 
            sub.set_yticks([-2., -1., 0., 1., 2.]) 
            if i == 1: 
                sub.text(0.95, 0.05, [r'$A_{1500}$', r'$A_V$'][ii],
                        transform=sub.transAxes, ha='right', va='bottom', fontsize=25) 

    fig.subplots_adjust(wspace=0.1, right=0.85)
    
    cbar_ax = fig.add_axes([0.9, 0.5, 0.025, 0.4])
    cbar = fig.colorbar(scs[0], cax=cbar_ax)
    cbar.ax.set_ylabel(r'median $A_{1500}$', labelpad=25, fontsize=25, rotation=270)
    
    cbar_ax = fig.add_axes([0.9, 0.05, 0.025, 0.4])
    cbar = fig.colorbar(scs[1], cax=cbar_ax)
    cbar.ax.set_ylabel(r'median $A_V$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_av_mssfr.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    #fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _ABC_stdA_MsSFR(): 
    ''' scatter in A_V as a function of M_*-SFR heat map 
    '''
    wave = np.linspace(1000, 10000, 451) 
    i5500 = 225

    # plotting 
    fig = plt.figure(figsize=(11,5))

    for i, sim in enumerate(['TNG', 'EAGLE']): 
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, _sim, _sfr0 = _sim_observables(sim.lower(), theta_median,
                no_Mr_cut=True)
        
        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst'])
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        cuts = (x_sim[0] > 20.)
    
        A_V = A_lambda[:,i5500]

        sub = fig.add_subplot(1,2,i+1)

        DFM.hist2d(_sim['logmstar'], _sim['logsfr.inst'], levels=[0.68, 0.95],	
                range=[[9., 12.], [-4., 2.]], color='k', 	
                contour_kwargs={'linewidths': 0.75, 'linestyles': 'dashed'}, 	
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)

        sc = sub.hexbin(_sim['logmstar'][cuts], _sim['logsfr.inst'][cuts], 
                C=A_V[cuts], reduce_C_function=np.std, gridsize=15,
                vmin=0., vmax=5., mincnt=10)
        
        sub.text(0.05, 0.95, sim, transform=sub.transAxes, ha='left', va='top', fontsize=25) 
        sub.set_xlim([9.5, 12.]) 
        sub.set_ylim([-2.5, 2.]) 
        if i != 0: sub.set_yticklabels([]) 
        sub.set_xticks([10., 11., 12.]) 
        sub.set_yticks([-2., -1., 0., 1., 2.]) 

    fig.subplots_adjust(wspace=0.1, right=0.85)
    
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.4])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$\sigma_{A_V}$', labelpad=25, fontsize=25, rotation=270)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, '_abc_stda_mssfr.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def _ABC_slope_AV(gal_type='all'): 
    ''' comparison of slope to A_V
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25 
    i3000 = 100
    i5500 = 225

    fig = plt.figure(figsize=(12,5))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)

    for i, sim in enumerate(['TNG', 'EAGLE']): 

        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, _sim, _sfr0 = _sim_observables(sim.lower(), theta_median)

        if gal_type == 'all': # all galaxies 
            subpop = np.ones(len(_sim['logmstar'])).astype(bool) 
        elif gal_type == 'starforming':
            # star-forming 
            subpop = (_sim['logsfr.inst'] - _sim['logmstar'] > -11)

        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst']
                )
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        A_V = A_lambda[:,i5500]
        S = A_lambda[:,i1500]/A_V
        
        delta_median = theta_median[3] * (_sim['logmstar'] - 10.) +\
                theta_median[4] * _sim['logsfr.inst'] + theta_median[5] 

        DFM.hist2d(A_V[subpop], S[subpop], range=[(0., 1.4), (0., 14.4)], 
                levels=[0.68, 0.95], bins=10, color=clrs[sim.lower()],
                plot_datapoints=False, fill_contours=True, plot_density=False, 
                ax=sub1)

        DFM.hist2d(A_V[subpop], delta_median[subpop], range=[(0., 1.4), (-1.5, 1.)],
                levels=[0.68, 0.95], bins=10, color=clrs[sim.lower()], 
                plot_datapoints=False, fill_contours=True, plot_density=False, 
                ax=sub2)

    # SMC
    #sub1.scatter([0.5], [4.8], c='b', s=60) 
    #sub1.text(0.55, 5., 'SMC', ha='left', va='bottom', fontsize=20) 
    # MW
    sub1.scatter([1.15], [2.8], c='k', marker='*', s=60) 
    sub1.text(1.2, 2.8, 'MW', ha='left', va='bottom', fontsize=20) 
    # Calzetti 
    sub1.plot([0.0, 1.4], [2.4, 2.4], c='k', ls='--')
    sub1.text(0.125, 2.2, 'Calzetti+(2000)', ha='left', va='top', fontsize=15) 
    # Inoue(2005) 
    sub1.plot([0.04624, 0.08447, 0.14629, 0.24109, 0.35660, 0.51096, 0.66340, 0.87693, 1.07223, 1.30417], 
            [13.85715, 8.97327, 6.44298, 4.74012, 3.71245, 3.03250, 2.57058, 2.15059, 1.92728, 1.68118], 
            c='k', ls=':', label='Inoue(2005)') 
    # Salim 2020 (0.12 dex scatter) 
    #sub1.plot(np.linspace(0., 1.4, 10), 
    #        10**(-0.68 * np.log10(np.linspace(0., 1.4, 10))+0.424), 
    #        c='k', ls='-.', label='Salim\&Narayanan(2020)')
    sub1.fill_between(np.linspace(0., 1.4, 100), 
            10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424-0.12), 
            10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424+0.12), 
            color='k', alpha=0.25, linewidth=0, label='Salim\&Narayanan(2020)')
    sub1.set_xlabel(r'$A_V$', fontsize=25)
    sub1.set_xlim(0.1, 1.4)
    sub1.set_ylabel('$S = A_{1500}/A_V$', fontsize=25)
    sub1.set_ylim(0., 14.4) 
    sub1.legend(loc='upper right', handletextpad=0.1, fontsize=18) 

    ## Wiit & Gordon (2000)
    #sub2.plot([0.01645, 0.63816, 1.77632, 2.83882],
    #        [-0.38591, -0.19720, 0.04641, 0.17912], 
    #        c='k', ls=':', label='Witt\&Gordon(2000)')
    # Chevallard+(2013)
    sub2.plot([0.10835, 0.21592, 0.32572, 0.53347, 1.08204, 1.39621], 
            [-0.69552, -0.40416, -0.20461, -0.00546, 0.19557, 0.25330], 
            c='k', ls='-.', label='Chevallard+(2013)')
    # Salmon+(2016)
    #sub2.plot([0.25000, 0.45395, 0.65461, 0.86513, 1.06250, 1.25987, 1.46711,
    #    1.68421, 1.88816, 2.08553, 2.28618, 2.49342, 2.70066], 
    #    [-0.44029, -0.33634, -0.22886, -0.15658, -0.11245, -0.06656, -0.04002,
    #        -0.00292, 0.02889, 0.04311, 0.06964, 0.11202, 0.14032], c='g',
    #    label='Salmon+(2016)') 
    # Salim+(2018)
    #sub2.plot([0.04749, 0.24662, 0.34430, 0.55215, 0.74590, 1.25348, 1.33918], 
    #        [-0.87232, -0.44691, -0.31288, -0.15964, -0.09486, 0.19159, 0.19779], 
    #        c='k', ls='-.', label='Salim+(2018)') 
    #sub2.scatter([0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800,
    #    0.900, 1.000, 1.100, 1.200, 1.300], [-0.499, -0.270, -0.111, -0.007,
    #        0.054, 0.112, 0.134, 0.144, 0.169, 0.196, 0.236, 0.282, 0.283],
    #        marker='^', s=40, label='Salim+(2018)') 
    sub2.fill_between( 
            [0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
                1.000, 1.100, 1.200, 1.300], 
            [-1.030, -0.830, -0.636, -0.519, -0.440, -0.381, -0.341, -0.286,
                -0.198, -0.103, -0.035, 0.054, 0.118], 
            [-0.499, -0.270, -0.111, -0.007, 0.054, 0.112, 0.134, 0.144, 0.169,
                0.196, 0.236, 0.282, 0.283], 
            facecolor='k', alpha=0.1, hatch='X', edgecolor='k', linewidth=0., label='Salim+(2018)') 
    #        facecolor='k', alpha=0.1, hatch='.', edgecolor='k', linewidth=0., label='Salim+(2018)') 
    # Trayford+(2020)
    #[-0.61368, -0.27968, -0.05030, 0.12274, 0.26761, 0.36419, 0.47686, 0.54125, 0.60563, 0.62173],
    sub2.plot([0.26801, 0.44502, 0.62433, 0.80357, 0.98036, 1.16435, 1.34110, 1.52263, 1.69932, 1.87595], 
            [-0.61368, -0.27968, -0.05030, 0.12274, 0.26761, 0.36419, 0.47686, 0.54125, 0.60563, 0.62173],
            '-^', c='k', label='Trayford+(2020)') 
    #    [-0.22334, -0.00201, 0.19517, 0.34004, 0.45272, 0.56942, 0.63380, 0.69014, 0.73038, 0.80684], 
    #    [-0.99598, -0.54125, -0.24748, -0.04225, 0.10262, 0.22334, 0.31187, 0.36016, 0.47284, 0.52918], 
    #    facecolor='k', alpha=0.1, hatch='X', edgecolor='k', linewidth=0., label='Trayford+(2020)') 
    sub2.set_xlabel(r'$A_V$', fontsize=25)
    sub2.set_xlim(0.1, 1.4)
    sub2.set_ylabel('$\delta$', fontsize=25)
    sub2.set_ylim(-1.6, 1.5) 

    # sim legends     
    _plt_sims = [] 
    for sim in ['TNG', 'EAGLE']:   
        _plt_sim = sub2.fill_between([], [], [], 
                color=clrs[sim.lower()], alpha=0.25, linewidth=0)
        _plt_sims.append(_plt_sim) 

    sim_legend = sub2.legend(_plt_sims, sims[1:], loc='lower right',
            handletextpad=0.1, prop={'size': 20})
    sub2.legend(loc='upper left', handletextpad=0.1, fontsize=18) 
    plt.gca().add_artist(sim_legend)

    fig.subplots_adjust(wspace=0.3)

    ffig = os.path.join(fig_dir, 'abc_slope_AV.%s.png' % gal_type) 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_slope_AV_subpop(): 
    ''' comparison of slope to A_V relation marking the different
    subpopulations 
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25 
    i3000 = 100
    i5500 = 225

    fig = plt.figure(figsize=(12,10))

    for i, sim in enumerate(['TNG', 'EAGLE']):
        sub1 = fig.add_subplot(2,2,2*i+1)
        sub2 = fig.add_subplot(2,2,2*i+2)
        sub1.text(0.05, 0.95, sim, transform=sub1.transAxes, fontsize=20, ha='left', va='top')

        # read sim 
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, _sim, _sfr0 = _sim_observables(sim.lower(), theta_median)

        # galaxies with low M* and high SFR 
        veryhighSFR = (_sim['logsfr.inst'] - _sim['logmstar'] > -9.75)
        highSFR     = ((_sim['logsfr.inst'] - _sim['logmstar'] < -9.75) & 
                (_sim['logsfr.inst'] - _sim['logmstar'] > -10.5))
        lowSFR      = ((_sim['logsfr.inst'] - _sim['logmstar'] < -10.5) & 
                (_sim['logsfr.inst'] - _sim['logmstar'] > -11.))
        verylowSFR  = ((_sim['logsfr.inst'] - _sim['logmstar'] < -11.) & 
                (_sim['logsfr.inst'] - _sim['logmstar'] > -12.))
        veryverylowSFR = (_sim['logsfr.inst'] - _sim['logmstar'] < -12.)
        lowmass = (_sim['logmstar'] < 10.5) 
        highmass = (_sim['logmstar'] >= 10.5) 

        subpops = [veryhighSFR, highSFR, lowSFR, verylowSFR, veryverylowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1', 'r', 'C4'][::-1]

        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst'])#,  mstar[subpop], sfr[subpop], nebular=False) 
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        A_V = A_lambda[:,i5500]
        S = A_lambda[:,i1500]/A_V
        
        delta_median = theta_median[3] * (_sim['logmstar'] - 10.) +\
                theta_median[4] * _sim['logsfr.inst'] + theta_median[5] 

        DFM.hist2d(A_V, S, range=[(0., 1.4), (0., 14.4)], 
                levels=[0.68, 0.95],
                bins=10, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub1)
        DFM.hist2d(A_V, delta_median, range=[(0., 1.4), (-1.5, 1.)], 
                levels=[0.68, 0.95],
                bins=10, color='k', #contour_kwargs={'linewidths': 0}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub2)
        
        for subpop, clr in zip(subpops, subclrs): 
            sub1.scatter(A_V[subpop & lowmass], S[subpop & lowmass], c=clr, s=3) 
            sub1.scatter(A_V[subpop & highmass], S[subpop & highmass], c=clr,
                    s=10, marker='o', edgecolors='k') 
            sub2.scatter(A_V[subpop & lowmass], delta_median[subpop & lowmass], c=clr, s=3) 
            sub2.scatter(A_V[subpop & highmass], delta_median[subpop & highmass],
                    c=clr, s=10, marker='o', edgecolors='k') 

        sub1.fill_between(np.linspace(0., 1.4, 100), 
                10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424-0.12), 
                10**(-0.68 * np.log10(np.linspace(0., 1.4, 100))+0.424+0.12), 
                color='k', alpha=0.25, linewidth=0, label='Salim\&Narayanan(2020)')

        sub2.fill_between( 
                [0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
                    1.000, 1.100, 1.200, 1.300], 
                [-1.030, -0.830, -0.636, -0.519, -0.440, -0.381, -0.341, -0.286,
                    -0.198, -0.103, -0.035, 0.054, 0.118], 
                [-0.499, -0.270, -0.111, -0.007, 0.054, 0.112, 0.134, 0.144, 0.169,
                    0.196, 0.236, 0.282, 0.283], 
                facecolor='k', alpha=0.1, hatch='X', edgecolor='k', linewidth=0., label='Salim+(2018)') 

        sub1.set_xlim(0.1, 1.4)
        sub1.set_ylabel('$S = A_{1500}/A_V$', fontsize=25)
        sub1.set_ylim(0., 14.4) 

        sub2.set_xlim(0.1, 1.4)
        sub2.set_ylabel('$\delta$', fontsize=25)
        sub2.set_ylim(-1.6, 1.5) 

    sub1.set_xlabel(r'$A_V$', fontsize=25)
    sub2.set_xlabel(r'$A_V$', fontsize=25)
    #sub1.legend(loc='upper right', handletextpad=0.1, fontsize=18) 

    fig.subplots_adjust(wspace=0.3)

    ffig = os.path.join(fig_dir, 'abc_slope_AV_subpop.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def _ABC_slope_AV_quiescent(): 
    ''' comparison of slope to A_V
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25 
    i3000 = 100
    i5500 = 225

    fig = plt.figure(figsize=(12,5))
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    for isim, sim, iabc in zip(range(len(sims))[1:], sims[1:], nabc[1:]): 
        # read sim 
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim), 'theta.t%i.dat' % iabc)) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, sfr0_sim, _sim = _sim_observables(sim.lower(), theta_median,
                zero_sfr_sample=True, return_sim=True)

        # get attenuation curve 
        _A_lambda = dem_attenuate(
                theta_median, 
                wave, 
                np.ones(len(wave)), 
                _sim['logmstar'], 
                _sim['logsfr.inst'], # mstar[subpop], sfr[subpop],
                nebular=False) 
        A_lambda = -2.5 * np.log10(_A_lambda)
        
        A_V = A_lambda[:,i5500]
        S = A_lambda[:,i1500]/A_V
        
        quiescent = (_sim['logsfr.inst'] - _sim['logmstar'] < -11.5) & ~sfr0_sim 
        print('%i quiescent galaxies' % np.sum(quiescent)) 
        
        delta_median = theta_median[3] * (_sim['logmstar'] - 10.) +\
                theta_median[4] * _sim['logsfr.inst'] + theta_median[5] 

        DFM.hist2d(A_V, S, levels=[0.68, 0.95],
                range=[(0., 2.), (0., 14.4)], bins=10, color=clrs[isim], #contour_kwargs={'linewidths': 0}, 
                plot_datapoints=False, fill_contours=True, plot_density=False, ax=sub1)
        sub1.scatter(A_V[quiescent], S[quiescent], c='r')

        A_V = A_lambda[:,i5500]

        DFM.hist2d(A_V, delta_median, levels=[0.68, 0.95],
                range=[(0., 2.), (-1.5, 1.)], bins=10, color=clrs[isim], #contour_kwargs={'linewidths': 0}, 
                plot_datapoints=False, fill_contours=True, plot_density=False, ax=sub2)

        sub2.scatter(A_V[quiescent], delta_median[quiescent], c='r')

    sub1.set_xlabel(r'$A_V$', fontsize=25)
    sub1.set_xlim(0.1, 2.)
    sub1.set_ylabel('$S = A_{1500}/A_V$', fontsize=25)
    sub1.set_ylim(0., 14.4) 

    sub2.set_xlabel(r'$A_V$', fontsize=25)
    sub2.set_xlim(0.1, 2.)
    sub2.set_ylabel('$\delta$', fontsize=25)
    sub2.set_ylim(-1.5, 1.) 

    # sim legends     
    _plt_sims = [] 
    for i in range(1,3): 
        _plt_sim = sub2.fill_between([], [], [], color=clrs[i], alpha=0.25,
                linewidth=0)
        _plt_sims.append(_plt_sim) 

    sim_legend = sub2.legend(_plt_sims, sims[1:], loc='lower right',
            handletextpad=0.1, prop={'size': 20})
    plt.gca().add_artist(sim_legend)

    fig.subplots_adjust(wspace=0.3)

    ffig = os.path.join(fig_dir, '_abc_slope_AV_quiescent.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def ABC_slope_MSFR(): 
    ''' comparison of slope on the M*-SFR plane  
    '''
    wave = np.linspace(1000, 10000, 451) 
    i1500 = 25 
    i3000 = 100
    i5500 = 225

    fig = plt.figure(figsize=(10,5))
    for isim, sim, iabc in zip(range(len(sims))[1:], sims[1:], nabc[1:]): 
        sub = fig.add_subplot(1,2,isim)
        # read sim 
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim), 'theta.t%i.dat' % iabc)) 
        theta_median = np.median(theta_T, axis=0) 

        x_sim, sfr0_sim, _sim = _sim_observables(sim.lower(), theta_median,
                zero_sfr_sample=True, return_sim=True)

        delta_median = theta_median[3] * (_sim['logmstar'] - 10.) +\
                theta_median[4] * _sim['logsfr.inst'] + theta_median[5] 
        sc = sub.scatter(_sim['logmstar'][~sfr0_sim],
                _sim['logsfr.inst'][~sfr0_sim]-_sim['logmstar'][~sfr0_sim], 
                c=delta_median[~sfr0_sim], vmin=-1.5, vmax=1.)

        sub.set_xlim([9., 12.]) 
        sub.set_ylim([-14., -8.]) 
        if isim != 1: sub.set_yticklabels([]) 
        sub.set_xticklabels([9., '', 10., '', 11.]) 
        sub.text(0.05, 0.95, sim, transform=sub.transAxes, fontsize=20, ha='left', va='top')

    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( sSFR $[yr^{-1}]$ )', labelpad=15, fontsize=25) 

    fig.subplots_adjust(wspace=0.1, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sc, cax=cbar_ax)

    ffig = os.path.join(fig_dir, 'abc_slope_msfr.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def ABC_attenuation(): 
    ''' comparison of attenuation curves of DEM models to standard attenuation curves in
    the literature.

    todo: 
    * compile the following attenuation curves: 
        * Cardelli+(1989) MW
        * Wild+(2011)
        * Kriek & Conroy (2013)
        * Reddy+(2015)    
    '''
    def _salim2018(_lam, logm, logsfr): 
        # Salim(2018) table 1 and Eq. 10
        logssfr = logsfr - logm 

        lam = _lam/10000. 
        if logssfr < -11:  # quiescent
            RV = 2.61
            B = 2.21
            a0 = -3.72
            a1 = 2.20
            a2 = -0.062
            a3 = 0.0080
        elif logm > 9.5 and logm < 10.5: 
            RV  = 2.99 
            B   = 1.73 
            a0  = -4.13
            a1  = 2.56
            a2  = -0.153    
            a3  = 0.0105
        elif logm > 10.5: 
            RV  = 3.47
            B   = 1.09
            a0  = -4.66
            a1  = 3.03
            a2  = -0.271
            a3  = 0.0147

        Dl = B * lam**2 * 0.035**2 / ((lam**2 - 0.2175**2)**2 + lam**2 * 0.035**2)
        kl = a0 + a1/lam + a2 / lam**2 + a3/lam**3 + Dl + RV
        return kl / RV

    def _calzetti(lam): 
        return dustFM.calzetti_absorption(lam)
    
    # Battisti+(2017) Eq. 9
    def _battisti2017(_lam): 
        lam = np.atleast_1d(_lam/1e4)
        x = 1./lam 
        lowlam = (lam < 0.63) 
        highlam = (lam >= 0.63) 
        kl = np.zeros(len(lam))
        kl[lowlam] = 2.40 * (-2.488 + 1.803 * x[lowlam] - 0.261 * x[lowlam]**2 + 0.0145 *
                x[lowlam]**3) + 3.67
        kl[highlam] = 2.30 * (-1.996 + 1.135 * x[highlam] - 0.0124 * x[highlam]**2) + 3.67
        return kl  

    # read Narayanan+(2018) attenuation curves
    fnara = os.path.join(dat_dir, 'obs', 'narayanan_median_Alambda.dat.txt')
    _wave_n2018, av_n2018 = np.loadtxt(fnara, skiprows=1, unpack=True, usecols=[0, 1]) 
    wave_n2018 = 1e4/_wave_n2018

    ## read SMC from Pei(1992)
    #fsmc = os.path.join(dat_dir, 'obs', 'pei1992_smc.txt') 
    #_1_lam, E_ratio = np.loadtxt(fsmc, skiprows=1, unpack=True, usecols=[0, 1]) 
    #wave_smc = 1e4/_1_lam
    #RV_smc = 2.93
    #Asmc = (E_ratio + RV_smc)/(1+RV_smc)
    ## normalize at 3000 
    #Asmc3000 = np.interp([3000.], wave_smc, Asmc)[0]
    #Asmc /= Asmc3000

    wave = np.linspace(1000, 10000, 2251) 
    i3000 = 500

    
    theta_meds, sim_seds = [], [] 
    for sim in ['TNG', 'EAGLE']:  
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 
        theta_meds.append(theta_median) 

        # get sims posterior
        _, _sim_sed, _ = _sim_observables(sim.lower(), theta_median)
        sim_seds.append(_sim_sed) 

    fig = plt.figure(figsize=(11,8))
    
    # SF or Q  
    for isfq, _sfq in enumerate(['star-forming', 'quiescent']): 
        # low or high mass 
        for im, _m in enumerate(['low mass', 'high mass']): 
            sub = fig.add_subplot(2,2, 2 * im + isfq + 1) 

            for i, sim in enumerate(['TNG', 'EAGLE']):  
                # get abc posterior
                theta_median = theta_meds[i]
                _sim_sed = sim_seds[i] 
    
                mstar   = _sim_sed['logmstar']
                sfr     = _sim_sed['logsfr.inst']
                ssfr    = sfr - mstar
                assert sfr.min() != -999

                # M* and SFR 
                if _m == 'low mass': 
                    mlim = (mstar < 11)
                elif _m == 'high mass': 
                    mlim = (mstar >= 11)

                if _sfq == 'star-forming': 
                    ssfrlim = (ssfr > -11) 
                    #sfrlim = (sfr > 0.5) 
                elif _sfq == 'quiescent': 
                    ssfrlim = (ssfr < -11) 
                    #sfrlim = (sfr < -0.5) 

                # subpopulation sample cut 
                subpop = mlim & ssfrlim 
    
                # get attenuation curve 
                _A_lambda = dem_attenuate(
                        theta_median, 
                        wave, 
                        np.ones(len(wave)), 
                        mstar[subpop], 
                        sfr[subpop])#, nebular=False) 
                A_lambda = -2.5 * np.log10(_A_lambda)
                # normalize to 3000A 
                A_lambda /= A_lambda[:,i3000][:,None]

                Al_1m, Al_med, Al_1p = np.quantile(A_lambda, [0.16, 0.5, 0.84], axis=0) 

                sub.fill_between(wave, Al_1m, Al_1p, color=clrs[sim.lower()],
                        alpha=0.25, linewidth=0, label=sim) 
                sub.plot(wave, Al_med, c=clrs[sim.lower()])
                
                # calzetti
                A_calzetti = _calzetti(wave) 
                A_salim = _salim2018(wave, [10.25, 11.][im], [1., -1][isfq])
                A_battisti = _battisti2017(wave)
                if isfq == 0 and im == 0: 
                    calz,   = sub.plot(wave, A_calzetti/A_calzetti[i3000], c='k', ls='--')
                    #smc,    = sub.plot(wave_smc, Asmc, c='r') 
                    b2017,  = sub.plot(wave, A_battisti/A_battisti[i3000],
                            c='k', ls=':')
                if isfq == 0: 
                    n2018, = sub.plot(wave_n2018, av_n2018, c='k', ls='-.') 
                    sal, = sub.plot(wave, A_salim/A_salim[i3000], c='k', 
                            lw=3, ls=(0, (1, 5))) #ls=(0, (3, 5, 1, 5, 1, 5)))

                sub.set_xlim(1.2e3, 1e4)
                if im == 0: sub.set_xticklabels([]) 
                sub.set_ylim(0., 4.) 
                if isfq == 1: sub.set_yticklabels([]) 

                if im == 0 and isfq == 0: 
                    sub.set_title(r'Star-forming ($\log {\rm SSFR} > -11$)', fontsize=20)
                #if im == 1 and isfq == 0: 
                    sub.legend(
                            [calz, b2017, n2018, sal], 
                            ['Calzetti+(2001)', 'Battisti+(2017)',
                                'Narayanan+(2018)', 'Salim+(2018)'], 
                            loc='upper right', handletextpad=0.2, fontsize=20) 
                if im == 0 and isfq == 1: 
                    sub.set_title(r'Quiescent ($\log {\rm SSFR} < -11$)', fontsize=20)
                    sub.legend(loc='upper right', handletextpad=0.2, fontsize=20) 
                    sub.text(1.01, 0.5, r'$\log M_*/M_\odot < 11$', 
                            transform=sub.transAxes, ha='left', va='center', rotation=270, fontsize=20)
                if im == 1 and isfq == 1:
                    sub.text(1.01, 0.5, r'$\log M_*/M_\odot > 11$', 
                            transform=sub.transAxes, ha='left', va='center', rotation=270, fontsize=20)
                    #sub.legend([sal], ['Salim+(2018)'], 
                    #        loc='upper right', handletextpad=0.2, fontsize=20) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'Wavelength [$\AA$]', labelpad=5, fontsize=20) 
    bkgd.set_ylabel(r'$A(\lambda)/A(3000\AA)$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_attenuation.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_attenuation_unnormalized(): 
    ''' comparison of unnormalized attenuation curves for different subpopulations 
    '''
    wave = np.linspace(1000, 10000, 2251) 

    theta_meds, sim_seds = [], [] 
    for sim in ['TNG', 'EAGLE']:  
        # get abc posterior
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_median = np.median(theta_T, axis=0) 
        theta_meds.append(theta_median) 

        # get sims posterior
        _, _sim_sed, _ = _sim_observables(sim.lower(), theta_median)
        sim_seds.append(_sim_sed) 

    fig = plt.figure(figsize=(11,8))
    # SF or Q  
    for isfq, _sfq in enumerate(['star-forming', 'quiescent']): 
        # low or high mass 
        for im, _m in enumerate(['low mass', 'high mass']): 
            sub = fig.add_subplot(2,2, 2 * im + isfq + 1) 

            for i, sim in enumerate(['TNG', 'EAGLE']):  
                # get abc posterior
                theta_median = theta_meds[i]
                _sim_sed = sim_seds[i] 
    
                mstar   = _sim_sed['logmstar']
                sfr     = _sim_sed['logsfr.inst']
                ssfr    = sfr - mstar
                assert sfr.min() != -999

                # M* and SFR 
                if _m == 'low mass': 
                    mlim = (mstar < 11)
                elif _m == 'high mass': 
                    mlim = (mstar >= 11)

                if _sfq == 'star-forming': 
                    ssfrlim = (ssfr > -11) 
                    #sfrlim = (sfr > 0.5) 
                elif _sfq == 'quiescent': 
                    ssfrlim = (ssfr < -11) 
                    #sfrlim = (sfr < -0.5) 

                # subpopulation sample cut 
                subpop = mlim & ssfrlim 
    
                # get attenuation curve 
                _A_lambda = dem_attenuate(
                        theta_median, 
                        wave, 
                        np.ones(len(wave)), 
                        mstar[subpop], 
                        sfr[subpop])#, nebular=False) 
                A_lambda = -2.5 * np.log10(_A_lambda)

                Al_1m, Al_med, Al_1p = np.quantile(A_lambda, [0.16, 0.5, 0.84], axis=0) 

                sub.fill_between(wave, Al_1m, Al_1p, color=clrs[sim.lower()],
                        alpha=0.25, linewidth=0, label=sim) 
                sub.plot(wave, Al_med, c=clrs[sim.lower()])

                sub.set_xlim(1.2e3, 1e4)
                if im == 0: sub.set_xticklabels([]) 
                sub.set_ylim(0., 8.) 
                if isfq == 1: sub.set_yticklabels([]) 

                if im == 0 and isfq == 0: 
                    sub.set_title(r'Star-forming ($\log {\rm SSFR} > -11$)', fontsize=20)
                if im == 0 and isfq == 1: 
                    sub.set_title(r'Quiescent ($\log {\rm SSFR} < -11$)', fontsize=20)
                    sub.legend(loc='upper right', handletextpad=0.2, fontsize=20) 
                    sub.text(1.01, 0.5, r'$\log M_*/M_\odot < 11$', 
                            transform=sub.transAxes, ha='left', va='center',
                            rotation=270, fontsize=20)
                if im == 1 and isfq == 1:
                    sub.text(1.01, 0.5, r'$\log M_*/M_\odot > 11$', 
                            transform=sub.transAxes, ha='left', va='center',
                            rotation=270, fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'Wavelength [$\AA$]', labelpad=5, fontsize=20) 
    bkgd.set_ylabel(r'$A(\lambda)$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_attenuation_unormalized.png') 
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
    
    obs_lims = [(20, 22.5), (0.2, 1.5), (-0.5, 4)]
    obs_lbls = [r'$M_r$ luminosity', '$G - R$', '$FUV - NUV$']
    yobs_lbls = [r'central luminosity function, $\Phi^{\rm cen}_{M_r}$', '$p(G - R)$', '$p(FUV - NUV)$']

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
                    range=ranges[i], bins=20, color='C1', linewidth=2, histtype='step') 
            _ = sub.hist(x_tng[i][x_tng[0] > 20], 
                    weights=np.repeat(1./vol_tng, len(x_tng[i])),
                    range=ranges[i], bins=20, color='C0', linewidth=2, histtype='step') 
            _ = sub.hist(x_obs[i],
                    weights=np.repeat(1./vol_sdss, len(x_obs[i])), 
                    range=ranges[i], bins=20, color='k',
                    linestyle='--', linewidth=2, histtype='step') 
    
        sub.set_xlabel(obs_lbls[i], fontsize=20) 
        sub.set_xlim(obs_lims[i]) 
        sub.set_ylabel(yobs_lbls[i], fontsize=20)
        
    fig.subplots_adjust(wspace=0.6)
    ffig = os.path.join(fig_dir, 'abc_tnorm_observables.1d.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_Lir(): 
    ''' compare L_IR predicted by the ABC posterior dust attenuation 
    '''
    cinA    = 2.9979e18 # A/s
    lsun    = 3.839e33 # erg/s
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################

    def get_seds(sim, theta): 
        sed = dustInfer._read_sed(sim) 

        zerosfr = (sed['logsfr.inst'] == -999)

        logsfr_min = sed['logsfr.inst'][~zerosfr].min() # minimum SFR
        sed['logsfr.inst'][zerosfr] = logsfr_min

        cuts = (sed['logmstar'] > 9.4) 

        sed_nodust  = sed['sed_noneb'][cuts,:]
        sed_dust    = dustFM.Attenuate(
                theta, 
                sed['wave'], 
                sed['sed_noneb'][cuts,:], 
                sed['sed_onlyneb'][cuts,:], 
                sed['logmstar'][cuts],
                sed['logsfr.inst'][cuts],
                dem=dem) 

        R_mag = measureObs.AbsMag_sed(sed['wave'], sed_dust, band='r_sdss') 

        return sed['wave'], sed_nodust, sed_dust, R_mag 

    L_irs, M_rs = [], [] 
    for sim in ['TNG', 'EAGLE']: 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 

        wave, sed_nodust, sed_dust, M_r = get_seds(sim.lower(), theta_med)

        L_nodust    = measureObs.tsum(wave, sed_nodust * cinA / (wave**2)) # Lsun
        L_dust      = measureObs.tsum(wave, sed_dust * cinA / (wave**2)) # Lsun
    
        # L_ir based on energy balance assumption of da Cunha+(2008) 
        L_ir = L_nodust - L_dust 
        notnan = ~np.isnan(np.log10(L_ir))

        L_irs.append(L_ir[notnan]) 
        M_rs.append(-1.*M_r[notnan]) 
    
    #########################################################################
    # plotting 
    #########################################################################
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    for i, _M_r, _L_ir, sim in zip(range(2), M_rs, L_irs, ['TNG', 'EAGLE']): 
        # R vs log L_IR  
        DFM.hist2d(_M_r, np.log10(_L_ir), range=[(20, 23), (8, 12)],
                levels=[0.68, 0.95], bins=20, color=clrs[sim.lower()], #contour_kwargs={'linewidths': 1}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
    
    _plt1 = sub.fill_between([], [], [], color=clrs['tng'], alpha=0.25, edgecolor='None')
    _plt2 = sub.fill_between([], [], [], color=clrs['eagle'], alpha=0.25, edgecolor='None')
    sub.legend([_plt1, _plt2], ['TNG', 'EAGLE'], loc='lower right',
            handletextpad=0.2, fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'IR dust emission $\log(~L_{\rm IR}$ [$L_\odot$] )', fontsize=25) 
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=25) 
    sub.set_ylim(9, 12) 
    
    ffig = os.path.join(fig_dir, 'abc_Lir.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def ABC_color_distribution(): 
    ''' (g-r) color distribution 
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dgr = gr_edges[1] - gr_edges[0]
    nbin = int(1.3/dgr)

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 
    
    sdss_mstar = sdss.data['log.M_star'][mr_complete]

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    xs, sfr0s, simseds = [], [], []
    for sim in sims: 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 

        x_mod, simsed, sfr0 = _sim_observables(sim.lower(), theta_med)
        xs.append(x_mod) 
        sfr0s.append(sfr0) 
        simseds.append(simsed)
    #########################################################################
    # plotting 
    #########################################################################

    fig = plt.figure(figsize=(10,10))
    for ii, mr_range in enumerate([[21.4, 21.9], [20, 20.5]]): 

        mr_lim_obs = ((x_obs[0] > mr_range[0]) & (x_obs[0] < mr_range[1]))
        sub = fig.add_subplot(2,1,ii+1)
        sub.hist(x_obs[1][mr_lim_obs], range=[-0.1, 1.2], bins=nbin,
                density=True, histtype='step', color='k')

        for i, sim in enumerate(sims): 
            if sim.lower() == 'simba': continue 
            mr_lim_sim = ((xs[i][0] > mr_range[0]) & (xs[i][0] < mr_range[1]))

            sub.hist(xs[i][1][mr_lim_sim], range=[-0.1, 1.2], bins=nbin, 
                    density=True, histtype='stepfilled', alpha=0.5, color=clrs[sim.lower()])
        sub.set_xlim(-0.1, 1.2) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$(G-R)$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_color_distribution.Mr_bin.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()


    fig = plt.figure(figsize=(10,10))
    for ii, ms_range in enumerate([(11.2, 11.5), (10.5, 10.8)]):
        ms_lim_obs = ((sdss_mstar > ms_range[0]) & (sdss_mstar < ms_range[1]))

        sub = fig.add_subplot(2,1,ii+1)
        sub.hist(x_obs[1][ms_lim_obs], range=[-0.1, 1.2], bins=nbin,
                density=True, histtype='step', color='k')

        for i, sim in enumerate(sims): 
            if sim.lower() == 'simba': continue 
            ms_lim_sim = ((simseds[i]['logmstar'] > ms_range[0]) & (simseds[i]['logmstar']< ms_range[1]))

            sub.hist(xs[i][1][ms_lim_sim], range=[-0.1, 1.2], bins=nbin, 
                    density=True, histtype='stepfilled', alpha=0.5, color=clrs[sim.lower()])
        sub.set_xlim(-0.1, 1.2) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$(G-R)$ color', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'abc_color_distribution.Ms_bin.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def galpop_attenuation(sim): 
    ''' determine why quiescent galaxies have high attenuation 
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations 
    #########################################################################
    fig = plt.figure(figsize=(10,10))

    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run(sim), 'theta.t%i.dat' % _nabc[sim])) 
    theta_med = np.median(theta_T, axis=0) 

    x, sfr0, sim_sed = _sim_observables(sim, theta_med,
            zero_sfr_sample=True, no_Mr_cut=True, return_sim=True)

    x_nodust, _     = _sim_observables(sim, np.array([0. for i in range(6)]),
            zero_sfr_sample=True, no_Mr_cut=True)

    # simple SSFR cut 
    quiescent = ((sim_sed['logsfr.inst'] - sim_sed['logmstar']) < -11.) & ~sfr0
    starforming =((sim_sed['logsfr.inst'] - sim_sed['logmstar']) >= -11.) & ~sfr0 
    quiescent_sfr0 = sfr0 
    print("%i star-forming galaxies" % np.sum(starforming))
    print("%i quiescent galaxies" % np.sum(quiescent)) 
    
    mr_lim = (x[0] > 20) 
    print("%i star-forming galaxies Mr < -20 w/ EDP" % 
            np.sum(starforming & mr_lim))
    print("%i quiescent galaxies Mr < -20 w/ EDP" % 
            np.sum(quiescent & mr_lim)) 

    # R vs (G - R)
    sub = fig.add_subplot(221)
    DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
            range=[ranges[0], ranges[1]], bins=20, color='k', 
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    DFM.hist2d(x[0], x[1], levels=[0.68, 0.95],
            range=[ranges[0], ranges[1]], bins=20, color=_clrs[sim], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    sub.scatter(x[0][starforming], x[1][starforming], c='C0', s=1)
    sub.scatter(x[0][quiescent_sfr0], x[1][quiescent_sfr0], c='r', s=1)
    sub.scatter(x[0][quiescent], x[1][quiescent], c='C1', s=1)
    sub.text(0.95, 0.95, 'w/ EDP', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylabel(r'$G-R$', fontsize=20) 
    sub.set_ylim(ranges[1]) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    
    sub = fig.add_subplot(222)
    DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
            range=[ranges[0], ranges[1]], bins=20, color='k', 
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    DFM.hist2d(x_nodust[0], x_nodust[1], levels=[0.68, 0.95],
            range=[ranges[0], ranges[1]], bins=20, color=_clrs[sim], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    sub.scatter(x_nodust[0][starforming], x_nodust[1][starforming], c='C0', s=1)
    sub.scatter(x_nodust[0][quiescent_sfr0], x_nodust[1][quiescent_sfr0], c='r', s=1)
    sub.scatter(x_nodust[0][quiescent], x_nodust[1][quiescent], c='C1', s=1)
    sub.text(0.95, 0.95, 'w/o EDP', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([])
    sub.set_ylim(ranges[1]) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_yticklabels([]) 
    
    # R vs FUV-NUV
    sub = fig.add_subplot(223)
    DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
            range=[ranges[0], ranges[2]], bins=20, color='k', 
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    DFM.hist2d(x[0], x[2], levels=[0.68, 0.95],
            range=[ranges[0], ranges[2]], bins=20, color=_clrs[sim], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    sub.scatter(x[0][starforming], x[2][starforming], c='C0', s=1) 
    sub.scatter(x[0][quiescent_sfr0], x[2][quiescent_sfr0], c='r', s=1)
    sub.scatter(x[0][quiescent], x[2][quiescent], c='C1', s=1) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(ranges[2]) 
    
    sub = fig.add_subplot(224)
    DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
            range=[ranges[0], ranges[2]], bins=20, color='k', 
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
    DFM.hist2d(x_nodust[0], x_nodust[2], levels=[0.68, 0.95],
            range=[ranges[0], ranges[2]], bins=20, color=_clrs[sim], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub) 
    sub.scatter(x_nodust[0][starforming], x_nodust[2][starforming], c='C0', s=1) 
    sub.scatter(x_nodust[0][quiescent_sfr0], x_nodust[2][quiescent_sfr0], c='r', s=1)
    sub.scatter(x_nodust[0][quiescent], x_nodust[2][quiescent], c='C1', s=1) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_yticklabels([]) 
    sub.set_ylim(ranges[2]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.1,
            fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'galpop_attenuation.%s.png' % sim) 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def quiescent_attenuation(): 
    ''' determine why quiescent galaxies have high attenuation 
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in simulations 
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('simba'), 'theta.t%i.dat' % nabc[0])) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('tng'), 'theta.t%i.dat' % nabc[1])) 
    theta_tng = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('eagle'), 'theta.t%i.dat' % nabc[2])) 
    theta_eagle = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba, sim_simba = _sim_observables('simba', theta_simba,
            zero_sfr_sample=True, no_Mr_cut=True, return_sim=True)
    x_tng, sfr0_tng, sim_tng = _sim_observables('tng', theta_tng,
            zero_sfr_sample=True, no_Mr_cut=True, return_sim=True)
    x_eagle, sfr0_eagle, sim_eagle = _sim_observables('eagle', theta_eagle,
            zero_sfr_sample=True, no_Mr_cut=True, return_sim=True)

    x_simba_nodust, _   = _sim_observables('simba', np.array([0. for i in range(6)]), 
            zero_sfr_sample=True, no_Mr_cut=True)
    x_tng_nodust, _     = _sim_observables('tng', np.array([0. for i in range(6)]),
            zero_sfr_sample=True, no_Mr_cut=True)
    x_eagle_nodust, _   = _sim_observables('eagle', np.array([0. for i in range(6)]),
            zero_sfr_sample=True, no_Mr_cut=True)

    # identify quiescent galaxies 
    xs      = [x_simba, x_tng, x_eagle]
    xs_nodust = [x_simba_nodust, x_tng_nodust, x_eagle_nodust]
    sims    = [sim_simba, sim_tng, sim_eagle]
    sfr0s   = [sfr0_simba, sfr0_tng, sfr0_eagle]
    xs_q_nodust = [] 

    for _x, _x_nodust, _sfr0, _sim in zip(xs, xs_nodust, sfr0s, sims): 
        # simple SSFR cut 
        quiescent = ((_sim['logsfr.inst'] - _sim['logmstar']) < -11.) & ~_sfr0
        print("%i quiescent galaxies" % np.sum(quiescent)) 
        print(_x[:,quiescent][:,:5])
        print(_x_nodust[:,quiescent][:,:5])
       
        # no attenuation for quiescent galaxies
        x = _x.copy()
        x[:,quiescent] = _x_nodust[:,quiescent] 
        xs_q_nodust.append(x) 

    #########################################################################
    # plotting 
    #########################################################################
    names   = ['SIMBA + EDP', 'TNG + EDP', 'EAGLE + EDP']

    fig = plt.figure(figsize=(5*len(xs),20))

    for i, _x, _x_nodust, name, clr in zip(range(len(xs)), xs, xs_nodust, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(4,len(xs),i+1)
        DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1., 1.5])
        
        sub = fig.add_subplot(4,len(xs),len(xs)+i+1)
        DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x_nodust[0], _x_nodust[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=20) 
            sub.text(0.95, 0.95, 'no EDP for quiescent', 
                    ha='right', va='top', transform=sub.transAxes, fontsize=25)
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1., 1.5])

        # R vs FUV-NUV
        sub = fig.add_subplot(4,len(xs),2*len(xs)+i+1)
        DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
        
        sub = fig.add_subplot(4,len(xs),3*len(xs)+i+1)
        DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x_nodust[0], _x_nodust[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
            sub.text(0.95, 0.95, 'no EDP for quiescent', 
                    ha='right', va='top', transform=sub.transAxes, fontsize=25)
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.1,
            fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'quiescent_attenuation.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def simba_starbursts(): 
    ''' examine the impact of low M* starburst galaxies in SIMBA 
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    #########################################################################
    # read in posteriors  
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('simba'), 'theta.t%i.dat' % nabc['simba'])) 
    theta_simba = np.median(theta_T, axis=0) 

    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('tng'), 'theta.t%i.dat' % nabc['tng'])) 
    theta_eagle = np.median(theta_T, axis=0) 
    #########################################################################
    # run FM for different scenarios 
    #########################################################################
    # simba with simba posterior 
    x_simba_nodust, _, _ = _sim_observables('simba', np.zeros(6),
            no_Mr_cut=True)
    # simba with simba posterior 
    x_simba, _, sfr0_simba = _sim_observables('simba', theta_simba,
            no_Mr_cut=True)
    # simba with eagle posterior 
    x_simba_eagle, _, sfr0_simba_eagle = _sim_observables('simba', theta_eagle,
            no_Mr_cut=True)

    # simba with eagle posterior excluding starbursts 
    _x_simba_eagle, sim_simba, _ = _sim_observables('simba',
            theta_eagle, no_Mr_cut=True)
    
    starburst = (sim_simba['logsfr.inst']  > -0.5 + 0.7 * (sim_simba['logmstar'] - 9))
    print('%i starburst galaxies' % np.sum(starburst))
    # remove starbursts
    x_simba_eagle_nosb = _x_simba_eagle[:,~starburst]

    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_simba_nodust, x_simba, x_simba_eagle, x_simba_eagle_nosb]
    names   = ['SIMBA + no dust', 'SIMBA + EDP', 'SIMBA+TNG EDP', 'SIMBA+TNG EDP\nno SB']

    fig = plt.figure(figsize=(5*(len(xs)+1),10))

    # SFR-M* relation 
    sub = fig.add_subplot(2, len(xs)+1, 1) 
    DFM.hist2d(sim_simba['logmstar'], sim_simba['logsfr.inst'],
            levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
            contour_kwargs={'linewidths': 0.5}, 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(sim_simba['logmstar'][starburst],
            sim_simba['logsfr.inst'][starburst],
            c='r', s=2) 

    sub.set_xlabel(r'log( $M_*$ [$M_\odot$] )', labelpad=5, fontsize=25)
    sub.set_xlim(9., 12.5)
    sub.set_xticks([9., 10., 11., 12.]) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
    sub.set_ylim([-3., 2.]) 

    for i, _x, name in zip(range(len(xs)), xs, names): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs)+1,i+2)
        DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='C0', 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        if i in [0, 1, 2]: 
            sub.scatter(_x[0][starburst], _x[1][starburst], c='r', s=2)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$G-R$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[1]) 
        sub.set_yticks([0., 0.5, 1., 1.5])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs)+1,i+len(xs)+3)
        DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='C0', 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        if i in [0, 1, 2]: 
            sub.scatter(_x[0][starburst], _x[2][starburst], c='r', s=2)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(ranges[2]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.1,
            fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'simba_starburst.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def sfr0_galaxies(): 
    ''' fraction of galaxies with SFR=0  
    '''
    for sim in sims: 
        _, sim_sed, sfr0 = _sim_observables(sim.lower(), np.zeros(6))
        mass_cut = np.ones(len(sim_sed['logmstar'])).astype(bool) #(sim_sed['logmstar'] > 10.) 
        print('%.1f of %s galaxies' %
                ((np.sum(mass_cut & sfr0)/np.sum(mass_cut))*100., sim))
        print('min SFR = %f' % (10**sim_sed['logsfr.inst'][~sfr0].min()))
    return None 


def _observables_noise(): 
    ''' comparison of color-magnitude relation of simulations + DEM with and
    without the noise model 
    '''
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]
    #########################################################################
    # read in simulations + DEM 
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('simba'), 'theta.t%i.dat' % nabc[0])) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('tng'), 'theta.t%i.dat' % nabc[1])) 
    theta_tng = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('eagle'), 'theta.t%i.dat' % nabc[2])) 
    theta_eagle = np.median(theta_T, axis=0) 
    
    # with noise model 
    x_noisy_simba, sfr0_simba = _sim_observables('simba', theta_simba, noise=True, 
            zero_sfr_sample=True)
    x_noisy_tng, sfr0_tng = _sim_observables('tng', theta_tng, noise=True, 
            zero_sfr_sample=True)
    x_noisy_eagle, sfr0_eagle = _sim_observables('eagle', theta_eagle, noise=True, 
            zero_sfr_sample=True)
    
    # without noise model
    x_simba, sfr0_simba = _sim_observables('simba', theta_simba, noise=False, 
            zero_sfr_sample=True)
    x_tng, sfr0_tng = _sim_observables('tng', theta_tng, noise=False, 
            zero_sfr_sample=True)
    x_eagle, sfr0_eagle = _sim_observables('eagle', theta_eagle, noise=False,  
            zero_sfr_sample=True)
    #########################################################################
    # plotting 
    #########################################################################
    xs_noisy = [x_noisy_simba, x_noisy_tng, x_noisy_eagle]
    xs      = [x_simba, x_tng, x_eagle]
    names   = ['SIMBA + DEM', 'TNG + DEM', 'EAGLE + DEM']

    fig = plt.figure(figsize=(5*len(xs),10))

    #for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
    for i, _x_noisy, _x, name, clr in zip(range(len(xs)), xs_noisy, xs, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x_noisy[0], _x_noisy[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
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
        sub.set_yticks([0., 0.5, 1., 1.5])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x_noisy[0], _x_noisy[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
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
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['no noise model'], loc='lower right', handletextpad=0.1,
            fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, '_observables_noise.png') 
    fig.savefig(ffig, bbox_inches='tight') 
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


def _sdsses(): 
    ''' which group catalog should I use?  
    '''
    # read in the different SDSS group catalogs 

    fig = plt.figure(figsize=(18,6))

    for i, mlim in enumerate(['9.7', '10.1', '10.5']): 
        tinker = Astrologs('tinkergroup', mlim=mlim) 

        sub = fig.add_subplot(1,3,i+1)
        R_mag = tinker.data['M_r']
        logms = tinker.data['log.M_star'] 
        print('%i of %i are centrals' % (np.sum(tinker.data['iscen']),
            len(tinker.data['iscen'])))

        sub.scatter(logms, R_mag, c='k', s=1, label='$M_{*, lim}=%s$' % mlim)
        
        sub.set_xlabel(r'$\log(\,M_*$ [$M_\odot$]$)$', fontsize=25) 
        sub.set_xlim(9.6, 12.)

        sub.set_ylabel(r'$M_r$', fontsize=25) 
        sub.set_ylim(-17., -23.4) 

    ffig = os.path.join(fig_dir, '_sdsses.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def _observables_sfr0(): 
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
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    x_simba, sfr0_simba  = _sim_observables('simba', np.array([0. for i in range(7)]), 
            zero_sfr_sample=False)
    x_tng, sfr0_tng      = _sim_observables('tng', np.array([0. for i in range(7)]),
            zero_sfr_sample=False)
    x_eag, sfr0_eag      = _sim_observables('eagle', np.array([0. for i in range(7)]),
            zero_sfr_sample=False)
    print('--- fraction of galaxies w/ 0 SFR ---') 
    print('simba %.2f' % (np.sum(sfr0_simba)/len(sfr0_simba)))
    print('tng %.2f' % (np.sum(sfr0_tng)/len(sfr0_tng)))
    print('eagle %.2f' % (np.sum(sfr0_eag)/len(sfr0_eag)))
    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_simba, x_tng, x_eag]
    names   = ['SIMBA (no dust)', 'TNG (no dust)', 'EAGLE (no dust)']
    clrs    = ['C1', 'C0', 'C2']
    sfr0s   = [sfr0_simba, sfr0_tng, sfr0_eag] 

    fig = plt.figure(figsize=(5*len(xs),10))

    for i, _x, _sfr0, name, clr in zip(range(len(xs)), xs, sfr0s, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(_x[0][~_sfr0], _x[1][~_sfr0], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
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
        sub.set_yticks([0., 0.5, 1., 1.5])

        # R vs FUV-NUV
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        DFM.hist2d(_x[0][~_sfr0], _x[2][~_sfr0], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color=clrs[i], 
                contour_kwargs={'linewidths': 0.5}, 
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

    _plth0, = sub.plot([], [], c='k', ls='--')
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'$M_r$ luminosity', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, '_observables_sfr0.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _SIMBA_oddities(): 
    ''' SIMBA has a number of differences compared to TNG and EAGLE. This
    script is to examine some of the oddities: 
    * luminous blue galaxies 
    '''
    # read ABC posterior 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'simba.slab_noll_msfr_fixbump.L2.3d', 'theta.t8.dat')) 
    theta_simba = np.median(theta_T, axis=0) 

    # run through DEM  
    _sim_sed = dustInfer._read_sed('simba') 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    downsample = np.ones(len(_sim_sed['logmstar'])).astype(bool)
    f_downsample = 1.#0.1

    cens    = _sim_sed['censat'].astype(bool) 
    mlim    = (_sim_sed['logmstar'] > 9.4) 
    zerosfr = (_sim_sed['logsfr.inst'] == -999)
    
    # sample cut centrals, mass limit, non 0 SFR
    cuts = cens & mlim & ~zerosfr & downsample 

    sim_sed = {} 
    sim_sed['sim']          = 'simba' 
    sim_sed['logmstar']     = _sim_sed['logmstar'][cuts].copy()
    sim_sed['logsfr.inst']  = _sim_sed['logsfr.inst'][cuts].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
    sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 
    
    # get observables R, G-R, FUV-NUV
    x_simba = dustInfer.sumstat_model(theta_simba, 
            sed=sim_sed,
            dem='slab_noll_msfr_fixbump',
            f_downsample=f_downsample, 
            statistic='2d',
            extra_data=None, 
            return_datavector=True)
    
    # galaxies with blue color but high Mr 
    blue_lum = (x_simba[0] > 21) & (x_simba[1] < 0.75) 
    
    # get observables with no DEM  
    x_nodust = dustInfer.sumstat_model(
            np.array([0. for i in range(7)]),
            sed=sim_sed,
            dem='slab_noll_msfr_fixbump',
            f_downsample=f_downsample, 
            statistic='2d',
            extra_data=None, 
            return_datavector=True)

    fig = plt.figure(figsize=(15,5))
    # plot R vs (G - R)
    sub = fig.add_subplot(131)
    DFM.hist2d(x_simba[0], x_simba[1], levels=[0.68, 0.95],
            range=[(20., 23.), (-0.05, 1.7)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(x_simba[0][blue_lum], x_simba[1][blue_lum], c='k', s=1)
    
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_ylabel(r'$G-R$', fontsize=20) 
    sub.set_ylim((-0.05, 1.7)) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_title('SIMBA + DEM', fontsize=20) 
    
    # plot (G-R)-Mr relation with no dust  
    sub = fig.add_subplot(132)
    DFM.hist2d(x_nodust[0], x_nodust[1], levels=[0.68, 0.95],
            range=[(20., 23.), (-0.05, 1.7)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(x_nodust[0][blue_lum], x_nodust[1][blue_lum], c='k', s=1)
    
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_ylim((-0.05, 1.7)) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_yticklabels([]) 
    sub.set_title('SIMBA + no dust ', fontsize=20) 
    
    # plot where they lie on the M*-SFR relation 
    sub = fig.add_subplot(133)
    DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
            range=[(9.0, 12.), (-3., 2.)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(sim_sed['logmstar'][blue_lum],
            sim_sed['logsfr.inst'][blue_lum], c='k', s=1)

    sub.set_xlabel(r'$\log M_*$', fontsize=20) 
    sub.set_xlim(9.0, 12) 
    sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
    sub.set_ylim((-3., 2.)) 

    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, '_simba_oddities.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()

    # what happens if we force m_\tau,SFR < 0 like the other simulations? 
    # get observables R, G-R, FUV-NUV
    theta_modified = theta_simba.copy()
    theta_modified[1] = -1.
    x_modified = dustInfer.sumstat_model(
            theta_modified,
            sed=sim_sed,
            dem='slab_noll_msfr_fixbump',
            f_downsample=f_downsample, 
            statistic='2d',
            extra_data=None, 
            return_datavector=True)
    
    fig = plt.figure(figsize=(20,5))

    blue_w_nodust = (x_nodust[0] > 20.2) & (x_nodust[1] < 0.15) 
    # plot (G-R)-Mr relation with no dust  
    sub = fig.add_subplot(141)
    DFM.hist2d(x_nodust[0], x_nodust[1], levels=[0.68, 0.95],
            range=[(20., 23.), (-0.05, 1.7)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(x_nodust[0][blue_lum], x_nodust[1][blue_lum], c='k', s=1)
    sub.scatter(x_nodust[0][blue_w_nodust], x_nodust[1][blue_w_nodust], c='C0', s=2)
    
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_ylabel(r'$G-R$', fontsize=20) 
    sub.set_ylim((-0.05, 1.7)) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_title('SIMBA + no dust ', fontsize=20) 
    
    # plot R vs (G - R)
    sub = fig.add_subplot(142)
    DFM.hist2d(x_simba[0], x_simba[1], levels=[0.68, 0.95],
            range=[(20., 23.), (-0.05, 1.7)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(x_simba[0][blue_lum], x_simba[1][blue_lum], c='k', s=1)
    sub.scatter(x_simba[0][blue_w_nodust], x_simba[1][blue_w_nodust], c='C0', s=2)
    
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_ylim((-0.05, 1.7)) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_title('SIMBA + DEM', fontsize=20) 
    
    # plot color-magnitude relation if we change m_tau,SFR 
    sub = fig.add_subplot(143)
    DFM.hist2d(x_modified[0], x_modified[1], levels=[0.68, 0.95],
            range=[(20., 23.), (-0.05, 1.7)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(x_modified[0][blue_lum], x_modified[1][blue_lum], c='k', s=1)
    sub.scatter(x_modified[0][blue_w_nodust], x_modified[1][blue_w_nodust], c='C0', s=2)
    
    sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_ylim((-0.05, 1.7)) 
    sub.set_yticks([0., 0.5, 1., 1.5])
    sub.set_yticklabels([]) 
    sub.set_title(r'SIMBA w/ $m_{\tau, {\rm SFR}} = -1$', fontsize=20) 
    
    # plot where they lie on the M*-SFR relation 
    sub = fig.add_subplot(144)
    DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
            range=[(9.0, 12.), (-3., 2.)], bins=20, color='C1', 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
    sub.scatter(sim_sed['logmstar'][blue_lum],
            sim_sed['logsfr.inst'][blue_lum], c='k', s=1)
    sub.scatter(sim_sed['logmstar'][blue_w_nodust],
            sim_sed['logsfr.inst'][blue_w_nodust], c='C0', s=2)

    sub.set_xlabel(r'$\log M_*$', fontsize=20) 
    sub.set_xlim(9.0, 12) 
    sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
    sub.set_ylim((-3., 2.)) 

    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, '_simba_oddities1.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None


def _simba_close_examination():
    ''' closer examination of simba. Reproducing some of Romeel's figures 
    '''
    # read in SDSS measurements
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(statistic='2d',
            return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    sdss = Catalog('tinker') 
    sdss_M_fuv, sdss_M_nuv, _, sdss_M_g, sdss_M_r, _, _ = sdss.data['NSA_ABSMAG'].T
    mr_complete = (sdss_M_r < -20.) 

    x_obs = [-1.*sdss_M_r[mr_complete], 
            sdss_M_g[mr_complete] - sdss_M_r[mr_complete], 
            sdss_M_fuv[mr_complete] - sdss_M_nuv[mr_complete]] 

    # read in simulations without dust attenuation
    nontheta = np.zeros(6) 
    x_simba, simba, sfr0_simba  = _sim_observables('simba', nontheta)

    logssfr = simba['logsfr.inst'] - simba['logmstar']

    # R vs (G - R)
    fig = plt.figure(figsize=(12,5))
    sub = fig.add_subplot(111)
    DFM.hist2d(x_obs[0], x_obs[1], range=[ranges[0], ranges[1]],
            levels=[0.68, 0.95], bins=20, color='k',
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub)
    sc = sub.scatter(x_simba[0], x_simba[1], c=logssfr, s=1, vmin=-12, vmax=-9,
            cmap='jet_r', rasterized=True)
    sub.text(0.95, 0.95, 'SIMBA (no dust)', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlabel('$r$', fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'$g-r$', fontsize=25) 
    sub.set_ylim(0., 0.9) 
    fig.colorbar(sc)
    ffig = os.path.join(fig_dir, '_simba_close_exam.optical_colormag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()

    # R vs FUV-NUV
    fig = plt.figure(figsize=(12,5))
    sub = fig.add_subplot(111)
    DFM.hist2d(x_obs[0], x_obs[2], range=[ranges[0], ranges[2]],
            levels=[0.68, 0.95], bins=20, color='k',
            contour_kwargs={'linestyles': 'dashed'}, 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub)
    sc = sub.scatter(x_simba[0], x_simba[2], c=logssfr, s=1, vmin=-12, vmax=-9,
            cmap='jet_r', rasterized=True)
    sub.text(0.95, 0.95, 'SIMBA (no dust)', ha='right', va='top', transform=sub.transAxes, fontsize=25)
    sub.set_xlabel('$r$', fontsize=25)
    sub.set_xlim(20., 23) 
    sub.set_xticks([20., 21., 22., 23]) 
    sub.set_xticklabels([-20, -21, -22, -23]) 
    sub.set_ylabel(r'$FUV-NUV$', fontsize=25) 
    sub.set_ylim(-1., 4.) 
    fig.colorbar(sc)
    ffig = os.path.join(fig_dir, '_simba_close_exam.uv_colormag.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def subpops_nodust(): 
    ''' Where in color-magnitude space without dust attenuation do
    subpopulations lie? 
    '''
    fig = plt.figure(figsize=(15,15))
    for i, sim in enumerate(sims): 

        x_sim, sim_sed, _ = _sim_observables(sim.lower(), np.zeros(6),
                no_Mr_cut=True)
        
        # galaxies with low M* and high SFR 
        veryhighSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -9.75)
        highSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -9.75) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -10.5))
        lowSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -10.5)

        lowmass = (sim_sed['logmstar'] < 10.5)
        highmass = (sim_sed['logmstar'] > 10.5) 

        subpops = [veryhighSFR, highSFR, lowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1'][::-1]

        # plot where they lie on the M*-SFR relation 
        sub = fig.add_subplot(3,3,3*i+1)
        DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
                range=[(9.0, 12.), (-3., 2.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(sim_sed['logmstar'][subpop & lowmass],
                    sim_sed['logsfr.inst'][subpop & lowmass], c=subclr, s=2)
            sub.scatter(sim_sed['logmstar'][subpop & highmass],
                    sim_sed['logsfr.inst'][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')
        sub.set_xlabel(r'$\log M_*$', fontsize=20) 
        sub.set_xlim(9.0, 12) 
        sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
        sub.set_ylim((-3., 2.)) 
        sub.text(0.05, 0.95, sims[i], 
            transform=sub.transAxes, ha='left', va='top', fontsize=20)

        # plot R vs (G - R)
        sub = fig.add_subplot(3,3,3*i+2)
        DFM.hist2d(x_sim[0], x_sim[1], levels=[0.68, 0.95],
                range=[(20., 23.), (-0.05, 1.7)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(x_sim[0][subpop & lowmass], x_sim[1][subpop & lowmass], c=subclr, s=2)
            sub.scatter(x_sim[0][subpop & highmass], x_sim[1][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')
        
        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_ylabel(r'$G-R$', fontsize=20) 
        sub.set_ylim((-0.05, 1.7)) 
        sub.set_yticks([0., 0.5, 1., 1.5])
        
        # plot R vs (G - R)
        sub = fig.add_subplot(3,3,3*i+3)
        DFM.hist2d(x_sim[0], x_sim[2], levels=[0.68, 0.95],
                range=[(20., 23.), (-1., 4.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(x_sim[0][subpop & lowmass], x_sim[2][subpop & lowmass], c=subclr, s=2)
            sub.scatter(x_sim[0][subpop & highmass], 
                    x_sim[2][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')

        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        sub.set_ylim(-1, 4.) 
        
        
    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, 'subpops_in_colormag_nodust.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None


def subpops(): 
    ''' Where in color-magnitude space do the low M* high SFR galaxies lie? 
    '''
    fig = plt.figure(figsize=(15,15))
    for i, sim in enumerate(sims): 
        # read ABC posterior 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 

        # run forward model  
        x_sim, sim_sed, sfr0 = _sim_observables(sim.lower(), theta_med,
                no_Mr_cut=True)
        maglim = (x_sim[0] > 20.) 

        # galaxies with low M* and high SFR 
        veryhighSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -9.75)
        highSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -9.75) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -10.5))
        lowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -10.5) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -11.))
        verylowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -11.) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -12.))
        veryverylowSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -12.)

        subpops = [veryhighSFR, highSFR, lowSFR, verylowSFR, veryverylowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1', 'r', 'C4'][::-1]

        # plot where they lie on the M*-SFR relation 
        sub = fig.add_subplot(3,3,3*i+1)
        DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
                range=[(9.0, 12.), (-3., 2.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(sim_sed['logmstar'][subpop & ~maglim],
                    sim_sed['logsfr.inst'][subpop & ~maglim],
                    c=subclr, s=3)
            sub.scatter(sim_sed['logmstar'][subpop & maglim],
                    sim_sed['logsfr.inst'][subpop & maglim], 
                    c=subclr, s=10, marker='o', edgecolors='k')

        sub.set_xlabel(r'$\log M_*$', fontsize=20) 
        sub.set_xlim(9.0, 12) 
        sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
        sub.set_ylim((-3., 2.)) 
        sub.text(0.05, 0.95, sims[i], 
            transform=sub.transAxes, ha='left', va='top', fontsize=20)

        # plot R vs (G - R)
        sub = fig.add_subplot(3,3,3*i+2)
        DFM.hist2d(x_sim[0], x_sim[1], levels=[0.68, 0.95],
                range=[(20., 23.), (-0.05, 1.7)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(x_sim[0][subpop], x_sim[1][subpop], c=subclr, s=2)
        
        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_ylabel(r'$G-R$', fontsize=20) 
        sub.set_ylim((-0.05, 1.7)) 
        sub.set_yticks([0., 0.5, 1., 1.5])
        
        # plot R vs (FUV-NUV)
        sub = fig.add_subplot(3,3,3*i+3)
        DFM.hist2d(x_sim[0], x_sim[2], levels=[0.68, 0.95],
                range=[(20., 23.), (-1, 4)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(x_sim[0][subpop], x_sim[2][subpop], c=subclr, s=2)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        sub.set_ylim(-1, 4) 

    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, '_subpops.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None


def subpops_optical_color(): 
    ''' Where in color-magnitude space do the low M* high SFR galaxies lie? 
    '''
    fig = plt.figure(figsize=(15,15))
    for i, sim in enumerate(sims): 
        # read ABC posterior 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 

        # run forward model on no dust  
        x_nodust, sim_sed, _ = _sim_observables(sim.lower(), np.zeros(6),
                no_Mr_cut=True)

        # run forward model on ABC posterior 
        x_w_dust, _, _ = _sim_observables(sim.lower(), theta_med,
                no_Mr_cut=True)

        # galaxies with low M* and high SFR 
        veryhighSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -9.75)
        highSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -9.75) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -10.5))
        lowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -10.5) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -11.))
        verylowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -11.) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -12.))
        veryverylowSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -12.)

        lowmass = (sim_sed['logmstar'] < 10.5) 
        highmass = (sim_sed['logmstar'] >= 10.5) 

        subpops = [veryhighSFR, highSFR, lowSFR, verylowSFR, veryverylowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1', 'r', 'C4'][::-1]

        # plot where they lie on the M*-SFR relation 
        sub = fig.add_subplot(3,3,3*i+1)
        DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
                range=[(9.0, 12.), (-3., 2.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    sim_sed['logmstar'][subpop & lowmass], 
                    sim_sed['logsfr.inst'][subpop & lowmass],
                    c=subclr, s=3)
            sub.scatter(
                    sim_sed['logmstar'][subpop & highmass], 
                    sim_sed['logsfr.inst'][subpop & highmass],
                    c=subclr, s=10, marker='o', edgecolors='k')

        sub.set_xlabel(r'$\log M_*$', fontsize=20) 
        sub.set_xlim(9.0, 12) 
        sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
        sub.set_ylim((-3., 2.)) 
        sub.text(0.05, 0.95, sims[i], 
            transform=sub.transAxes, ha='left', va='top', fontsize=20)

        # plot R vs (G - R)
        sub = fig.add_subplot(3,3,3*i+2)
        DFM.hist2d(x_nodust[0], x_nodust[1], range=[(20., 23.), (-0.05, 1.7)],
                levels=[0.68, 0.95], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    x_nodust[0][subpop & lowmass], 
                    x_nodust[1][subpop & lowmass], 
                    c=subclr, s=2)
            sub.scatter(
                    x_nodust[0][subpop & highmass], 
                    x_nodust[1][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')

        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_ylabel(r'$G-R$', fontsize=20) 
        sub.set_ylim((-0.05, 1.7)) 
        sub.set_yticks([0., 0.5, 1., 1.5])
        
        # plot R vs (FUV-NUV)
        sub = fig.add_subplot(3,3,3*i+3)
        DFM.hist2d(x_w_dust[0], x_w_dust[1], range=[(20., 23.), (-0.05, 1.7)],
                levels=[0.68, 0.95], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    x_w_dust[0][subpop & lowmass], 
                    x_w_dust[1][subpop & lowmass], 
                    c=subclr, s=2)
            sub.scatter(
                    x_w_dust[0][subpop & highmass], 
                    x_w_dust[1][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')
        
        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_ylabel(r'$G-R$', fontsize=20) 
        sub.set_ylim((-0.05, 1.7)) 
        sub.set_yticks([0., 0.5, 1., 1.5])

    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, 'subpops_optical_color.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None


def subpops_uv_color(): 
    ''' Where in color-magnitude space do the low M* high SFR galaxies lie? 
    '''
    fig = plt.figure(figsize=(15,15))
    for i, sim in enumerate(sims): 
        # read ABC posterior 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            abc_run(sim.lower()), 'theta.t%i.dat' % nabc[sim.lower()])) 
        theta_med = np.median(theta_T, axis=0) 
        
        # run forward model on no dust  
        x_nodust, sim_sed, _ = _sim_observables(sim.lower(), np.zeros(6),
                no_Mr_cut=True)

        # run forward model on ABC posterior 
        x_w_dust, _, _ = _sim_observables(sim.lower(), theta_med,
                no_Mr_cut=True)

        # galaxies with low M* and high SFR 
        veryhighSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -9.75)
        highSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -9.75) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -10.5))
        lowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -10.5) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -11.))
        verylowSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -11.) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -12.))
        veryverylowSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -12.)

        lowmass = (sim_sed['logmstar'] < 10.5) 
        highmass = (sim_sed['logmstar'] >= 10.5) 

        subpops = [veryhighSFR, highSFR, lowSFR, verylowSFR, veryverylowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1', 'r', 'C4'][::-1]

        # plot where they lie on the M*-SFR relation 
        sub = fig.add_subplot(3,3,3*i+1)
        DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
                range=[(9.0, 12.), (-3., 2.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    sim_sed['logmstar'][subpop & lowmass], 
                    sim_sed['logsfr.inst'][subpop & lowmass],
                    c=subclr, s=3)
            sub.scatter(
                    sim_sed['logmstar'][subpop & highmass], 
                    sim_sed['logsfr.inst'][subpop & highmass],
                    c=subclr, s=10, marker='o', edgecolors='k')

        sub.set_xlabel(r'$\log M_*$', fontsize=20) 
        sub.set_xlim(9.0, 12) 
        sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
        sub.set_ylim((-3., 2.)) 
        sub.text(0.05, 0.95, sims[i], 
            transform=sub.transAxes, ha='left', va='top', fontsize=20)

        # plot R vs (G - R)
        sub = fig.add_subplot(3,3,3*i+2)
        DFM.hist2d(x_nodust[0], x_nodust[2], range=[(20., 23.), (-1, 4)],
                levels=[0.68, 0.95], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    x_nodust[0][subpop & lowmass], 
                    x_nodust[2][subpop & lowmass], 
                    c=subclr, s=2)
            sub.scatter(
                    x_nodust[0][subpop & highmass], 
                    x_nodust[2][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        sub.set_ylim(-1, 4) 
        
        # plot R vs (FUV-NUV)
        sub = fig.add_subplot(3,3,3*i+3)
        DFM.hist2d(x_w_dust[0], x_w_dust[2], range=[(20., 23.), (-1, 4)],
                levels=[0.68, 0.95], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(
                    x_w_dust[0][subpop & lowmass], 
                    x_w_dust[2][subpop & lowmass], 
                    c=subclr, s=2)
            sub.scatter(
                    x_w_dust[0][subpop & highmass], 
                    x_w_dust[2][subpop & highmass], 
                    c=subclr, s=10, marker='o', edgecolors='k')
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        sub.set_ylim(-1, 4) 

    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, 'subpops_uv_color.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None


def _extra_luminous(): 
    ''' Where in M*-SFR are the most luminous galaxies lie? DEM current
    produces exceed luminous galaxies. 
    '''
    sims = ['simba', 'tng', 'eagle']
    iabc = [8, 6, 6] 
    clrs = ['C1', 'C0', 'C2'] 

    fig = plt.figure(figsize=(10,15))
    for i in range(len(sims)): 
        # read ABC posterior 
        theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
            '%s.slab_noll_msfr_fixbump.L2.3d' % sims[i], 
            'theta.t%i.dat' % iabc[i])) 
        theta_sim = np.median(theta_T, axis=0) 

        # run through DEM  
        _sim_sed = dustInfer._read_sed(sims[i]) 
        wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
        downsample = np.ones(len(_sim_sed['logmstar'])).astype(bool)
        f_downsample = 1.#0.1

        cens    = _sim_sed['censat'].astype(bool) 
        mlim    = (_sim_sed['logmstar'] > 9.4) 
        zerosfr = (_sim_sed['logsfr.inst'] == -999)
    
        # sample cut centrals, mass limit, non 0 SFR
        cuts = cens & mlim & ~zerosfr & downsample 

        sim_sed = {} 
        sim_sed['sim']          = sims[i] 
        sim_sed['logmstar']     = _sim_sed['logmstar'][cuts].copy()
        sim_sed['logsfr.inst']  = _sim_sed['logsfr.inst'][cuts].copy() 
        sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
        sim_sed['sed_noneb']    = _sim_sed['sed_noneb'][cuts,:][:,wlim].copy() 
        sim_sed['sed_onlyneb']  = _sim_sed['sed_onlyneb'][cuts,:][:,wlim].copy() 
    
        # get observables R, G-R, FUV-NUV
        x_sim = dustInfer.sumstat_model(
                theta_sim, 
                sed=sim_sed,
                dem='slab_noll_msfr_fixbump',
                f_downsample=f_downsample, 
                statistic='2d',
                extra_data=None, 
                return_datavector=True)
        
        # galaxies with low M* and high SFR 
        luminous = x_sim[0] > 21.

        subpops = [luminous]
        subclrs = ['C0']

        # plot where they lie on the M*-SFR relation 
        sub = fig.add_subplot(3,2,2*i+1)
        DFM.hist2d(sim_sed['logmstar'], sim_sed['logsfr.inst'], levels=[0.68, 0.95],
                range=[(9.0, 12.), (-3., 2.)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(sim_sed['logmstar'][subpop],
                    sim_sed['logsfr.inst'][subpop], c=subclr, s=2)
        sub.set_xlabel(r'$\log M_*$', fontsize=20) 
        sub.set_xlim(9.0, 12) 
        sub.set_ylabel(r'$\log {\rm SFR}$', fontsize=20) 
        sub.set_ylim((-3., 2.)) 
        sub.text(0.05, 0.95, sims[i], 
            transform=sub.transAxes, ha='left', va='top', fontsize=20)

        # plot R vs (G - R)
        sub = fig.add_subplot(3,2,2*i+2)
        DFM.hist2d(x_sim[0], x_sim[1], levels=[0.68, 0.95],
                range=[(20., 23.), (-0.05, 1.7)], bins=20, color='k', 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        for subpop, subclr in zip(subpops, subclrs): 
            sub.scatter(x_sim[0][subpop], x_sim[1][subpop], c=subclr, s=2)
        
        sub.set_xlabel(r'$M_r$ luminosity', fontsize=20) 
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_ylabel(r'$G-R$', fontsize=20) 
        sub.set_ylim((-0.05, 1.7)) 
        sub.set_yticks([0., 0.5, 1., 1.5])
        
    fig.subplots_adjust(wspace=0.3)
    ffig = os.path.join(fig_dir, '_extra_luminous.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None


def _abc_color_Ms(): 
    ''' examine g-r color in bins of M* to compare with Trayford+(2015)
    '''
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    r_edges, gr_edges, fn_edges, _ = dustInfer.sumstat_obs(name='sdss',
            statistic='2d', return_bins=True)
    dr  = r_edges[1] - r_edges[0]
    dgr = gr_edges[1] - gr_edges[0]
    dfn = fn_edges[1] - fn_edges[0]
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    mr_complete = (sdss['mr_tinker'][...] < -20.)

    x_obs = [-1.*sdss['mr_tinker'][...][mr_complete], 
            sdss['mg_tinker'][...][mr_complete] - sdss['mr_tinker'][...][mr_complete], 
            sdss['ABSMAG'][...][:,0][mr_complete] - sdss['ABSMAG'][...][:,1][mr_complete]] 
    sfr0_obs = np.zeros(len(x_obs[0])).astype(bool)
    sdss_ms = np.log10(sdss['ms_tinker'][...][mr_complete])
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'simba.slab_noll_msfr_fixbump.L2.3d', 'theta.t%i.dat' % simba_abc)) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'tng.slab_noll_msfr_fixbump.L2.3d', 'theta.t%i.dat' % tng_abc)) 
    theta_tng = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'eagle.slab_noll_msfr_fixbump.L2.3d', 'theta.t%i.dat' % eagle_abc)) 
    theta_eagle = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba, simba  = _sim_observables('simba', theta_simba,
            zero_sfr_sample=True, return_sim=True)
    x_tng, sfr0_tng, tng        = _sim_observables('tng', theta_tng,
            zero_sfr_sample=True, return_sim=True)
    x_eagle, sfr0_eagle, eagle  = _sim_observables('eagle', theta_eagle,
            zero_sfr_sample=True, return_sim=True)
    #########################################################################
    # plotting 
    #########################################################################
    Ms_bins = [(11., 11.5), (10.5, 11.), (10., 10.5)]
    obs_lbls = ['$G - R$']

    fig = plt.figure(figsize=(8,10))
    for i in range(len(Ms_bins)):
        simba_msbin = (Ms_bins[i][0] < simba['logmstar']) & (Ms_bins[i][1] >= simba['logmstar'])
        tng_msbin   = (Ms_bins[i][0] < tng['logmstar']) & (Ms_bins[i][1] >= tng['logmstar'])
        eagle_msbin = (Ms_bins[i][0] < eagle['logmstar']) & (Ms_bins[i][1] >= eagle['logmstar'])
        obs_msbin   = (Ms_bins[i][0] < sdss_ms) & (Ms_bins[i][1] >= sdss_ms)
    
        sub = fig.add_subplot(3,1,i+1)

        _ = sub.hist(x_obs[1][obs_msbin], density=True, #weights=np.repeat(1./vol_sdss, np.sum(obs_msbin)), 
                range=(0., 2.), bins=20, color='k', alpha=0.25, 
                histtype='stepfilled') 

        _ = sub.hist(x_simba[1][simba_msbin], density=True, #weights=np.repeat(1./vol_simba, np.sum(simba_msbin)),
                range=(0., 2.), bins=20, color='C1', linewidth=2.2, histtype='step') 
        _ = sub.hist(x_tng[1][tng_msbin], density=True, # weights=np.repeat(1./vol_tng, np.sum(tng_msbin)),
                range=(0., 2.), bins=20, color='C0', linewidth=2, histtype='step') 
        _ = sub.hist(x_eagle[1][eagle_msbin], density=True, # weights=np.repeat(1./vol_eagle, np.sum(eagle_msbin)),
                range=(0., 2.), bins=20, color='C2', linewidth=1.8, histtype='step') 

        sub.set_xlim(0., 2.) 
                
        sub.text(0.05, 0.95, '$%.1f < M_* > %.1f$' % (Ms_bins[i][0], Ms_bins[i][1]), 
                transform=sub.transAxes, fontsize=20, ha='left', va='top')
    sub.set_xlabel(r'$G - R$', fontsize=20) 

    _plth0 = sub.fill_between([], [], [], color='k', alpha=0.25, edgecolor='None')
    _plth1, = sub.plot([], [], c='C1')
    _plth2, = sub.plot([], [], c='C0')
    _plth3, = sub.plot([], [], c='C2')

    names   = ['SIMBA + DEM', 'TNG + DEM', 'EAGLE + DEM', 'SDSS']
    sub.legend([_plth1, _plth2, _plth3, _plth0], names, loc='upper right',
            handletextpad=0.2, fontsize=14) 

    bkgd = fig.add_subplot(111, frameon=False)
    #bkgd.set_ylabel(r'number density $({\rm Mpc}/h)^{-3}$', labelpad=25, fontsize=25) 
    bkgd.set_ylabel(r'$p(G-R)$', labelpad=25, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    ffig = os.path.join(fig_dir, '_abc_observables.ms_bin.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close()
    return None 


def _profile_tng(): 
    ''' abc for tng takes very long to converge 
    '''
    import time 
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        abc_run('eagle'), 'theta.t%i.dat' % nabc[2])) 
    theta_mid = np.median(theta_T, axis=0) 
    
    
    for sim in ['simba', 'eagle', 'tng']: 
        t0 = time.time() 
        x_mod, _ = _sim_observables(sim, theta_mid, zero_sfr_sample=True)
        print('%s takes %.1f sec' % (sim, (time.time()-t0)))
        print(x_mod.shape) 
    return None 


if __name__=="__main__": 

    #SMF_MsSFR()

    #DEM()

    #Observables()

    # ABC posteriors 
    #ABC_corner() 
    
    # color magnitude relation for ABC posterior
    #ABC_Observables()
    
    # color distriution in Mr bins 
    #ABC_color_distribution()
    
    # slope-AV relation for ABC posterior
    #ABC_slope_AV(gal_type='all')
    #ABC_slope_AV(gal_type='starforming')
    #ABC_slope_AV_subpop()

    #ABC_A_MsSFR()
    _ABC_stdA_MsSFR()

    # amplitude normalized attenuation curves
    #ABC_attenuation()
    #ABC_attenuation_unnormalized()
    
    # dust IR emission luminosity 
    #ABC_Lir()

    # subpopulations in color magnitude space 
    #subpops_nodust()
    #subpops()
    #subpops_optical_color()
    #subpops_uv_color()

    # sfr=0 galaxies 
    #sfr0_galaxies()

    # examine starburst galaxies in simba 
    #simba_starbursts()
    #_simba_close_examination()

    # examining what happens if quiiescent galaxies don't have attenuation
    #for sim in ['simba', 'tng', 'eagle']: 
    #    galpop_attenuation(sim)
    #quiescent_attenuation()


    # testing the noise model  
    #_observables_noise()

    
    # tnorm Av model  
    #ABC_tnorm_corner()
    #ABC_tnorm_Observables()
    #slab_model()
    #slab_tnorm_comparison()
    
    #_observables_sfr0()
    #_SIMBA_oddities()
    #_extra_luminous()
    #_abc_color_Ms()
