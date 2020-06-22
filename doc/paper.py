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
vol_simba   = 100.**3 # (Mpc/h)^3
vol_tng     = 75.**3 # (Mpc/h)^3
vol_eagle   = 67.77**3 # (Mpc/h)^3  Lbox 100 Mpc h = 0.6777

simba_abc = 8
tng_abc = 9 
eagle_abc = 11 

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

    feag = os.path.join(dat_dir, 'sed', 'eagle.hdf5') 
    eag = h5py.File(feag, 'r')
    cen_eag = eag['censat'][...].astype(bool)
    #########################################################################
    # calculate SMFs
    #########################################################################
    logms_bin = np.linspace(8., 13., 21)
    dlogms = logms_bin[1:] - logms_bin[:-1]

    Ngal_simba, _   = np.histogram(simba['logmstar'][...][cen_simba], bins=logms_bin)
    Ngal_tng, _     = np.histogram(tng['logmstar'][...][cen_tng], bins=logms_bin)
    Ngal_eag, _     = np.histogram(eag['logmstar'][...][cen_eag], bins=logms_bin)

    phi_simba   = Ngal_simba.astype(float) / vol_simba / dlogms
    phi_tng     = Ngal_tng.astype(float) / vol_tng / dlogms
    phi_eag     = Ngal_eag.astype(float) / vol_eagle / dlogms
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
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_eag, c='C2', 
            label='EAGLE')
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


def M_SFR(): 
    ''' M_* - SFR relation of the simulations to highlight their differences
    '''
    #########################################################################
    # read in SDSS 
    #########################################################################
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    sdss_ms = np.log10(sdss['ms_tinker'][...]) 
    sdss_sfr = sdss['sfr_tinker'][...] + sdss_ms
    #########################################################################
    # read simulations 
    #########################################################################
    fsimba = os.path.join(dat_dir, 'sed', 'simba.hdf5')
    simba = h5py.File(fsimba, 'r')
    cen_simba = simba['censat'][...].astype(bool)
    simba_ms  = simba['logmstar'][...][cen_simba]
    simba_sfr = simba['logsfr.inst'][...][cen_simba]

    ftng = os.path.join(dat_dir, 'sed', 'tng.hdf5')
    tng = h5py.File(ftng, 'r')
    cen_tng = tng['censat'][...].astype(bool)
    tng_ms  = tng['logmstar'][...][cen_tng]
    tng_sfr = tng['logsfr.inst'][...][cen_tng]

    feag = os.path.join(dat_dir, 'sed', 'eagle.hdf5') 
    eag = h5py.File(feag, 'r')
    cen_eag = eag['censat'][...].astype(bool)
    eag_ms  = eag['logmstar'][...][cen_eag]
    eag_sfr = eag['logsfr.inst'][...][cen_eag]
    #########################################################################
    # plot M*-SFR relations  
    #########################################################################
    names   = ['SIMBA', 'TNG', 'EAGLE']
    clrs    = ['C1', 'C0', 'C2']
    ms      = [simba_ms, tng_ms, eag_ms]
    sfrs    = [simba_sfr, tng_sfr, eag_sfr]

    fig = plt.figure(figsize=(15,5))
    for i in range(len(names)): 
        sub = fig.add_subplot(1,3,i+1)
        DFM.hist2d(sdss_ms, sdss_sfr, levels=[0.68, 0.95],
                range=[[7.8, 12.], [-4., 2.]], color='k', 
                contour_kwargs={'linewidths': 0.75, 'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(ms[i], sfrs[i], color=clrs[i], 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                contour_kwargs={'linewidths': 0.5}, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub)
        sub.text(0.05, 0.95, names[i], transform=sub.transAxes, ha='left', va='top', fontsize=25) 
        sub.set_xlim([9., 12.]) 
        sub.set_ylim([-3., 2.]) 
        if i != 0: sub.set_yticklabels([]) 
        sub.set_xticklabels([9., '', 10., '', 11.]) 
    
    _plth0, = sub.plot([], [], c='k', ls='--')
    sub.legend([_plth0], ['SDSS'], loc='lower right', handletextpad=0.2,
            fontsize=20) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(fig_dir, 'm_sfr.png') 
    fig.savefig(ffig, bbox_inches='tight') 

    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def SMF_MsSFR(): 
    ''' figure comparing the stellar mass fucntions of SDSS and the simulations
    and M*-SFR relation 
    '''
    #########################################################################
    # read in SDSS 
    #########################################################################
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.phi_logMstar.dat') 
    logms_low, logms_high, phi_sdss, err_phi_sdss = np.loadtxt(fsdss, unpack=True)

    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    sdss_ms = np.log10(sdss['ms_tinker'][...]) 
    sdss_sfr = sdss['sfr_tinker'][...] + sdss_ms
    #########################################################################
    # read simulations 
    #########################################################################
    fsimba = os.path.join(dat_dir, 'sed', 'simba.hdf5')
    simba = h5py.File(fsimba, 'r')
    cen_simba = simba['censat'][...].astype(bool)
    simba_ms  = simba['logmstar'][...][cen_simba]
    simba_sfr = simba['logsfr.inst'][...][cen_simba]

    ftng = os.path.join(dat_dir, 'sed', 'tng.hdf5')
    tng = h5py.File(ftng, 'r')
    cen_tng = tng['censat'][...].astype(bool)
    tng_ms  = tng['logmstar'][...][cen_tng]
    tng_sfr = tng['logsfr.inst'][...][cen_tng]

    feag = os.path.join(dat_dir, 'sed', 'eagle.hdf5') 
    eag = h5py.File(feag, 'r')
    cen_eag = eag['censat'][...].astype(bool)
    eag_ms  = eag['logmstar'][...][cen_eag]
    eag_sfr = eag['logsfr.inst'][...][cen_eag]
    #########################################################################
    # calculate SMFs
    #########################################################################
    logms_bin = np.linspace(8., 13., 21)
    dlogms = logms_bin[1:] - logms_bin[:-1]

    Ngal_simba, _   = np.histogram(simba['logmstar'][...][cen_simba], bins=logms_bin)
    Ngal_tng, _     = np.histogram(tng['logmstar'][...][cen_tng], bins=logms_bin)
    Ngal_eag, _     = np.histogram(eag['logmstar'][...][cen_eag], bins=logms_bin)

    phi_simba   = Ngal_simba.astype(float) / vol_simba / dlogms
    phi_tng     = Ngal_tng.astype(float) / vol_tng / dlogms
    phi_eag     = Ngal_eag.astype(float) / vol_eagle / dlogms
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
    sub.errorbar(0.5*(logms_low + logms_high), phi_sdss, yerr=err_phi_sdss,
            fmt='.k', label='SDSS Centrals')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_simba, c='C0',
            label='SIMBA')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_tng, c='C1', 
            label='TNG')
    sub.plot(0.5*(logms_bin[1:] + logms_bin[:-1]), phi_eag, c='C2', 
            label='EAGLE')
    sub.legend(loc='lower left', handletextpad=0.3, fontsize=20)
    sub.set_xlabel(r'log( $M_*$ [$M_\odot$] )', labelpad=5, fontsize=25)
    sub.set_xlim(9.7, 12.5)
    sub.set_xticks([10., 11., 12.]) 
    sub.set_ylabel(r'central SMF ($\Phi^{\rm cen}_{M_*}$)', fontsize=24)
    sub.set_yscale("log")
    sub.set_ylim(5e-6, 3e-2) 
    
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
        sub.set_xlim([9., 12.]) 
        sub.set_ylim([-3., 2.]) 
        if i != 0: sub.set_yticklabels([]) 
        else: sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
        if i == 1: sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
        sub.set_xticklabels([9., '', 10., '', 11.]) 
    
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

    def _cardelli89_mw(_lam): 
        x = 10000./_lam
        y = x - 1.82
        return None  

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

    wave = np.linspace(1000, 10000, 1000) 

    fig = plt.figure(figsize=(11,8))
    
    logSFR_sf = 0.5
    logSFR_q = -2.
    M_low = 10.
    M_high = 11.

    # low mass SFing galaxies
    sub = fig.add_subplot(221) 
    sub.plot(wave, _dem(wave, M_low, logSFR_sf), c='k')
    sub.plot(wave, _salim2018(wave, M_low, logSFR_sf), c='C0')
    sub.plot(wave, _calzetti(wave), c='C1') 
    #sub.text(0.95, 0.95, r'Star-Forming', ha='right',
    #        va='top', transform=sub.transAxes, fontsize=20)
    #sub.text(0.05, 0.95, r'$M_*\sim 10^{%.f}M_\odot$' % M_low, ha='left',
    #        va='top', transform=sub.transAxes, fontsize=20)

    sub.set_xlim(1.2e3, 1e4)
    sub.set_xticklabels([]) 
    sub.set_ylim(0., 7.) 
    sub.set_title('Star-Forming', fontsize=20)

    # low mass Quiescent galaxies
    logSFR = -2 
    sub = fig.add_subplot(222) 
    _plt_dem, = sub.plot(wave, _dem(wave, M_low, logSFR_q), c='k')
    _plt_salim, = sub.plot(wave, _salim2018(wave, M_low, logSFR_q), c='C0')
    _plt_cal, = sub.plot(wave, _calzetti(wave), c='C1') 

    sub.legend([_plt_dem, _plt_cal, _plt_salim], 
            ['DEM', 'Calzetti+(2001)', 'Salim+(2018)'], loc='lower right',
            bbox_to_anchor=(1., 0.5), handletextpad=0.25, fontsize=16) 
    #sub.text(0.95, 0.95, r'Quiescent', ha='right',
    #        va='top', transform=sub.transAxes, fontsize=20)
    sub.set_xlim(1.2e3, 1e4)
    sub.set_xticklabels([]) 
    sub.set_yticklabels([]) 
    sub.set_ylim(0., 7.) 
    sub.set_title('Quiescent', fontsize=20)
    sub.text(1.01, 0.5, r'$M_*\sim 10^{%.f}M_\odot$' % M_low, 
            transform=sub.transAxes, ha='left', va='center', rotation=270, fontsize=20)
    
    sub = fig.add_subplot(223) 
    sub.plot(wave, _dem(wave, M_high, logSFR_sf), c='k')
    sub.plot(wave, _salim2018(wave, M_high, logSFR_sf), c='C0')
    sub.plot(wave, _calzetti(wave), c='C1') 
    #sub.text(0.05, 0.95, r'$M_*\sim 10^{%.f}M_\odot$' % M_high, ha='left',
    #        va='top', transform=sub.transAxes, fontsize=20)

    sub.set_xlim(1.2e3, 1e4)
    sub.set_ylim(0., 7.) 

    # Quiescent galaxies
    logSFR = -2 
    sub = fig.add_subplot(224) 
    sub.plot(wave, _dem(wave, M_high, logSFR_q), c='k')
    sub.plot(wave, _salim2018(wave, M_high, logSFR_q), c='C0')
    sub.plot(wave, _calzetti(wave), c='C1') 
    sub.set_xlim(1.2e3, 1e4)
    sub.set_yticklabels([]) 
    sub.set_ylim(0., 7.) 
    sub.text(1.01, 0.5, r'$M_*\sim 10^{%.f}M_\odot$' % M_high,
            transform=sub.transAxes, ha='left', va='center', rotation=270, fontsize=20)

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_xlabel(r'Wavelength [$\AA$]', labelpad=5, fontsize=20) 
    bkgd.set_ylabel(r'$A(\lambda)$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

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
    ranges = [(r_edges[0], r_edges[-1]), (-0.05, 1.7), (-1., 4.)]

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
    xs      = [x_obs, x_simba, x_tng, x_eag]
    names   = ['SDSS', 'SIMBA (no dust)', 'TNG (no dust)', 'EAGLE (no dust)']
    clrs    = ['k', 'C1', 'C0', 'C2']
    sfr0s   = [sfr0_obs, sfr0_simba, sfr0_tng, sfr0_eag] 

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
        sub.set_yticks([0., 0.5, 1., 1.5])

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


def _sim_observables(sim, theta, model='slab', fixbump=True,
        zero_sfr_sample=False, return_sim=False): 
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
                dem='%s_noll_msfr%s' % (model, ['', '_fixbump'][fixbump]), 
                f_downsample=f_downsample, statistic='2d',
                return_datavector=True)
        _zerosfr = sim_sed['logsfr.inst'] == -999
    else: 
        zerosfr_obs = dustInfer._observable_zeroSFR(
                _sim_sed['wave'][wlim], 
                _sim_sed['sed_neb'][cens & mlim & zerosfr & downsample,:][:,wlim])

        x_mod = dustInfer.sumstat_model(theta, sed=sim_sed,
                dem='%s_noll_msfr%s' % (model, ['', '_fixbump'][fixbump]), 
                f_downsample=f_downsample, statistic='2d',
                extra_data=zerosfr_obs, return_datavector=True)
        _zerosfr = np.zeros(x_mod.shape[1]).astype(bool)
        _zerosfr[np.sum(cuts):] = True
    mr_cut = x_mod[0] > 20
    if not return_sim: 
        return x_mod[:,mr_cut], _zerosfr[mr_cut]
    else: 
        _simsed = {} 
        if not zero_sfr_sample: 
            _simsed['logmstar'] = sim_sed['logmstar'][mr_cut]
            _simsed['logsfr.inst'] = sim_sed['logsfr.inst'][mr_cut]
        else: 
            _simsed['logmstar'] = np.concatenate([sim_sed['logmstar'],
                _sim_sed['logmstar'][cens & mlim & zerosfr & downsample]])[mr_cut]
            _simsed['logsfr.inst'] = np.concatenate([sim_sed['logsfr.inst'], 
                _sim_sed['logsfr.inst'][cens & mlim & zerosfr & downsample]])[mr_cut]
        return x_mod[:,mr_cut], _zerosfr[mr_cut], _simsed


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
    # update these values as I see fit
    sims = ['SIMBA', 'TNG', 'EAGLE']
    clrs = ['C1', 'C0', 'C2']
    names = ['simba.slab_noll_msfr_fixbump.L2.3d',
            'tng.slab_noll_msfr_fixbump.L2.3d', 
            'eagle.slab_noll_msfr_fixbump.L2.3d']
    Ts = [8, 8, 11]
        
    prior_min = np.array([-5., -5., 0., -4., -4., -4., 1.]) 
    prior_max = np.array([5.0, 5.0, 6., 4.0, 4.0, 4.0, 4.]) 
    prior_range = [(_min, _max) for _min, _max in zip(prior_min, prior_max)]
        
    lbls = [r'$m_{\tau,1}$', r'$m_{\tau,2}$', r'$c_{\tau}$', 
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
        keep_cols[:6] = True
            
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
    
    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, sim in enumerate(sims): 
        bkgd.fill_between([],[],[], color=clrs[i], label=sim) 
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
        'simba.slab_noll_msfr_fixbump.L2.3d', 'theta.t8.dat')) 
    theta_simba = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'tng.slab_noll_msfr_fixbump.L2.3d', 'theta.t8.dat')) 
    theta_tng = np.median(theta_T, axis=0) 
    theta_T = np.loadtxt(os.path.join(os.environ['GALPOPFM_DIR'], 'abc',
        'eagle.slab_noll_msfr_fixbump.L2.3d', 'theta.t11.dat')) 
    theta_eagle = np.median(theta_T, axis=0) 

    x_simba, sfr0_simba = _sim_observables('simba', theta_simba,
            zero_sfr_sample=True)
    x_tng, sfr0_tng = _sim_observables('tng', theta_tng,
            zero_sfr_sample=True)
    x_eagle, sfr0_eagle = _sim_observables('eagle', theta_eagle,
            zero_sfr_sample=True)
    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_simba, x_tng, x_eagle]
    names   = ['SIMBA + DEM', 'TNG + DEM', 'EAGLE + DEM']
    clrs    = ['C1', 'C0', 'C2']

    fig = plt.figure(figsize=(5*len(xs),10))

    #for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
    for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
        # R vs (G - R)
        sub = fig.add_subplot(2,len(xs),i+1)
        DFM.hist2d(x_obs[0], x_obs[1], levels=[0.68, 0.95],
                range=[ranges[0], ranges[1]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[1], levels=[0.68, 0.95],
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
        DFM.hist2d(x_obs[0], x_obs[2], levels=[0.68, 0.95],
                range=[ranges[0], ranges[2]], bins=20, color='k', 
                contour_kwargs={'linestyles': 'dashed'}, 
                plot_datapoints=False, fill_contours=False, plot_density=False, ax=sub)
        DFM.hist2d(_x[0], _x[2], levels=[0.68, 0.95],
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

    obs_lims = [(20, 22.5), (0.2, 1.5), (-0.5, 5)]
    obs_lbls = [r'$M_r$ luminosity', '$G - R$', '$FUV - NUV$']
    yobs_lbls = [r'central luminosity function, $\Phi^{\rm cen}_{M_r}$', '$p(G - R)$', '$p(FUV - NUV)$']

    fig = plt.figure(figsize=(16,4))
    for i in range(3):
        sub = fig.add_subplot(1,3,i+1)
        
        if i == 0: 
            mr_bin = np.linspace(20, 23, 7) 
            dmr = mr_bin[1:] - mr_bin[:-1]

            Ngal_simba, _ = np.histogram(x_simba[i], bins=mr_bin)
            Ngal_tng, _ = np.histogram(x_tng[i], bins=mr_bin)
            Ngal_eagle, _ = np.histogram(x_eagle[i], bins=mr_bin)

            phi_simba   = Ngal_simba.astype(float) / vol_simba / dmr
            phi_tng     = Ngal_tng.astype(float) / vol_tng / dmr
            phi_eagle   = Ngal_eagle.astype(float) / vol_eagle / dmr

            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_simba, c='C1')
            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_tng, c='C0')
            sub.plot(0.5*(mr_bin[1:] + mr_bin[:-1]), phi_eagle, c='C2')

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
                    range=ranges[i], bins=20, color='C1', linewidth=2.2, histtype='step') 
            _ = sub.hist(x_tng[i][x_tng[0] > 20], 
                    weights=np.repeat(1./vol_tng, len(x_tng[i])),
                    range=ranges[i], bins=20, color='C0', linewidth=2, histtype='step') 
            _ = sub.hist(x_eagle[i], 
                    weights=np.repeat(1./vol_eagle, len(x_eagle[i])),
                    range=ranges[i], bins=20, color='C2', linewidth=1.8, histtype='step') 
            _ = sub.hist(x_obs[i],
                    weights=np.repeat(1./vol_sdss, len(x_obs[i])), 
                    range=ranges[i], bins=20, color='k',
                    linestyle='--', linewidth=2, histtype='step') 
    
        sub.set_xlabel(obs_lbls[i], fontsize=20) 
        sub.set_xlim(obs_lims[i]) 
        sub.set_ylabel(yobs_lbls[i], fontsize=20)

    _plth0, = sub.plot([], [], c='k', ls='--')
    _plth1, = sub.plot([], [], c='C1')
    _plth2, = sub.plot([], [], c='C0')
    _plth3, = sub.plot([], [], c='C2')

    names   = ['SDSS', 'SIMBA + DEM', 'TNG + DEM', 'EAGLE + DEM']
    sub.legend([_plth0, _plth1, _plth2, _plth3], names, loc='upper right',
            handletextpad=0.2, fontsize=14) 
    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(fig_dir, 'abc_observables.1d.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    
    Mr_bins = [(20., 21.), (21., 22.5)]
    obs_lims = [(0.0, 2.), (-0.5, 4.5)]
    obs_lbls = ['$G - R$', '$FUV - NUV$']
    obs_ylims = [(0., 1.5e-3), (0., 1.2e-3)]

    fig = plt.figure(figsize=(12,8))
    for i in range(len(Mr_bins)):
        
        simba_mrbin = (Mr_bins[i][0] < x_simba[0]) & (Mr_bins[i][1] >= x_simba[0])
        tng_mrbin   = (Mr_bins[i][0] < x_tng[0]) & (Mr_bins[i][1] >= x_tng[0])
        eagle_mrbin = (Mr_bins[i][0] < x_eagle[0]) & (Mr_bins[i][1] >= x_eagle[0])
        obs_mrbin   = (Mr_bins[i][0] < x_obs[0]) & (Mr_bins[i][1] >= x_obs[0])
        
        for j in range(2): 
            sub = fig.add_subplot(2,len(Mr_bins),2*i+j+1)

            _ = sub.hist(x_obs[j+1][obs_mrbin],
                    weights=np.repeat(1./vol_sdss, np.sum(obs_mrbin)), 
                    range=ranges[j+1], bins=20, color='k', alpha=0.25, 
                    histtype='stepfilled') 

            _ = sub.hist(x_simba[j+1][simba_mrbin], 
                    weights=np.repeat(1./vol_simba, np.sum(simba_mrbin)),
                    range=ranges[j+1], bins=20, color='C1', linewidth=2.2, histtype='step') 
            _ = sub.hist(x_tng[j+1][tng_mrbin], 
                    weights=np.repeat(1./vol_tng, np.sum(tng_mrbin)),
                    range=ranges[j+1], bins=20, color='C0', linewidth=2, histtype='step') 
            _ = sub.hist(x_eagle[j+1][eagle_mrbin], 
                    weights=np.repeat(1./vol_eagle, np.sum(eagle_mrbin)),
                    range=ranges[j+1], bins=20, color='C2', linewidth=1.8, histtype='step') 

            if i == 1:
                sub.set_xlabel(obs_lbls[j], fontsize=20) 
            else: 
                sub.set_xticklabels([])
            sub.set_xlim(obs_lims[j]) 
            sub.set_ylim(obs_ylims[j]) 
            if j == 0: 
                sub.text(0.05, 0.95, '$-%.1f > M_r > -%.1f$' % (Mr_bins[i][0], Mr_bins[i][1]), 
                        transform=sub.transAxes, fontsize=20, ha='left', va='top')

            if i == 0 and j == 1: 
                _plth0 = sub.fill_between([], [], [], color='k', alpha=0.25, edgecolor='None')
                _plth1, = sub.plot([], [], c='C1')
                _plth2, = sub.plot([], [], c='C0')
                _plth3, = sub.plot([], [], c='C2')

                names   = ['SIMBA + DEM', 'TNG + DEM', 'EAGLE + DEM', 'SDSS']
                sub.legend([_plth1, _plth2, _plth3, _plth0], names, loc='upper right',
                        handletextpad=0.2, fontsize=14) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.set_ylabel(r'number density $({\rm Mpc}/h)^{-3}$', labelpad=25, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.subplots_adjust(wspace=0.4)
    ffig = os.path.join(fig_dir, 'abc_observables.mr_bin.png') 
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


def fig_tex(ffig, pdf=False):
    ''' given filename of figure return a latex friendly file name
    '''
    path, ffig_base = os.path.split(ffig)
    ext = ffig_base.rsplit('.', 1)[-1]
    ffig_name = ffig_base.rsplit('.', 1)[0]

    _ffig_name = ffig_name.replace('.', '_')
    if pdf: ext = 'pdf'
    return os.path.join(path, '.'.join([_ffig_name, ext]))


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


def _subpops(): 
    ''' Where in color-magnitude space do the low M* high SFR galaxies lie? 
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
        veryhighSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -9.75)
        highSFR = ((sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -9.75) & 
                (sim_sed['logsfr.inst'] - sim_sed['logmstar'] > -10.5))
        lowSFR = (sim_sed['logsfr.inst'] - sim_sed['logmstar'] < -10.5)

        subpops = [veryhighSFR, highSFR, lowSFR][::-1]
        subclrs = ['C0', 'C2', 'C1'][::-1]

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
    ffig = os.path.join(fig_dir, '_subpops.png') 
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


if __name__=="__main__": 
    #SDSS()
    #SMFs() 
    #M_SFR()
    #SMF_MsSFR()
    #DEM()
    #Observables()
    #ABC_corner() 
    #_ABC_corner_flexbump() 
    #_ABC_Observables()
    #ABC_Observables()
    
    #ABC_tnorm_corner()
    #ABC_tnorm_Observables()
    #slab_tnorm_comparison()

    #_SIMBA_oddities()
    #_subpops()
    #_extra_luminous()
    _abc_color_Ms()
