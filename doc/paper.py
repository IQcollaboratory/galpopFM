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
    vol_simba = 100.**3 # (Mpc/h)^3

    ftng = os.path.join(dat_dir, 'sed', 'tng.hdf5')
    tng = h5py.File(ftng, 'r')
    cen_tng = tng['censat'][...].astype(bool)
    vol_tng = 75.**3 # (Mpc/h)^3
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
    # don't touch these values!
    nbins = [8, 10, 10]
    ranges = [(20, 24), (-1, 1.), (-1, 4.)]
    dRmag   = 0.5 
    dbalmer = 0.2
    dfuvnuv = 0.5
    #########################################################################
    # read in SDSS measurements
    #########################################################################
    Rmag_edges, balmer_edges, fuvnuv_edges, x_obs, x_obs_err = np.load(os.path.join(dat_dir, 'obs',
                'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_Balmer_FUVNUV.npy'),
                allow_pickle=True)
    #########################################################################
    # read in simulations without dust attenuation
    #########################################################################
    R_mag_simba, balmer_ratio_simba, FUV_NUV_simba =\
            _sim_observables('simba', downsample=False) 
    R_mag_tng, balmer_ratio_tng, FUV_NUV_tng =\
            _sim_observables('tng', downsample=False) 

    Nbins_simba, _ = np.histogramdd(
            np.array([-1.*R_mag_simba, balmer_ratio_simba, FUV_NUV_simba]).T, 
            bins=nbins, range=ranges)
    Nbins_tng, _ = np.histogramdd(
            np.array([-1.*R_mag_tng, balmer_ratio_tng, FUV_NUV_tng]).T, 
            bins=nbins, range=ranges)
    
    # volume of simulation 
    vol_simba = 100.**3
    vol_tng = 75.**3

    x_simba = Nbins_simba.astype(float) / vol_simba / dRmag / dbalmer / dfuvnuv
    x_tng = Nbins_tng.astype(float) / vol_tng / dRmag / dbalmer / dfuvnuv
    #########################################################################
    # plotting 
    #########################################################################
    xs      = [x_obs, x_simba, x_tng]
    names   = ['SDSS', 'SIMBA', 'TNG']
    clrs    = ['Greys', 'Oranges', 'Blues'] 

    fig = plt.figure(figsize=(5*len(xs),10))

    # Rmag - Balmer
    for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
        sub = fig.add_subplot(2,len(xs),i+1)
        sub.pcolormesh(Rmag_edges, balmer_edges, dfuvnuv * np.sum(_x, axis=2).T,
                vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap=clr)
        sub.text(0.95, 0.95, name, ha='right', va='top', transform=sub.transAxes, fontsize=25)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([])
        if i == 0: 
            sub.set_ylabel(r'$\log (H_\alpha/H_\beta)/(H_\alpha/H_\beta)_I$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(-1., 1.) 

    # Rmag - FUV-NUV
    for i, _x, name, clr in zip(range(len(xs)), xs, names, clrs): 
        sub = fig.add_subplot(2,len(xs),i+len(xs)+1)
        h = sub.pcolormesh(Rmag_edges, fuvnuv_edges, dbalmer * np.sum(_x, axis=1).T,
                vmin=1e-5, vmax=1e-2, norm=mpl.colors.LogNorm(), cmap=clr)
        sub.set_xlim(20., 23) 
        sub.set_xticks([20., 21., 22., 23]) 
        sub.set_xticklabels([-20, -21, -22, -23]) 
        sub.set_yticks([-0.8, -0.4, 0., 0.4, 0.8]) 
        if i == 0: 
            sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
        else: 
            sub.set_yticklabels([]) 
        sub.set_ylim(-1., 4.) 

    fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.02, 0.7])
    fig.colorbar(h, cax=cbar_ax)

    ffig = os.path.join(fig_dir, 'observables.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    fig.savefig(fig_tex(ffig, pdf=True), bbox_inches='tight') 
    plt.close()
    return None 


def _sim_observables(sim, downsample=True): 
    ''' read specified simulations and return data vector 
    '''
    # read simulations 
    _sim_sed = dustInfer._read_sed(sim) 
    wlim = (_sim_sed['wave'] > 1e3) & (_sim_sed['wave'] < 8e3) 
    if downsample: 
        _downsample = np.zeros(len(_sim_sed['logmstar'])).astype(bool)
        _downsample[::10] = True
        f_downsample = 0.1
    else: 
        _downsample = np.ones(len(_sim_sed['logmstar'])).astype(bool)
        f_downsample = 1.
    
    cens = _sim_sed['censat'].astype(bool) & (_sim_sed['logmstar'] > 9.4) & _downsample

    sim_sed = {} 
    sim_sed['sim']          = sim
    sim_sed['logmstar']     = _sim_sed['logmstar'][cens].copy()
    sim_sed['logsfr.100']   = _sim_sed['logsfr.100'][cens].copy() 
    sim_sed['wave']         = _sim_sed['wave'][wlim].copy()
    sim_sed['sed']    = _sim_sed['sed_neb'][cens,:][:,wlim].copy() 
    
    # observational measurements 
    F_mag = measureObs.AbsMag_sed(sim_sed['wave'], sim_sed['sed'], band='galex_fuv') 
    N_mag = measureObs.AbsMag_sed(sim_sed['wave'], sim_sed['sed'], band='galex_nuv') 
    R_mag = measureObs.AbsMag_sed(sim_sed['wave'], sim_sed['sed'], band='r_sdss') 
    FUV_NUV = F_mag - N_mag 
    
    # balmer measurements 
    Ha_dust, Hb_dust = measureObs.L_em(['halpha', 'hbeta'], sim_sed['wave'], sim_sed['sed']) 
    balmer_ratio = Ha_dust/Hb_dust

    HaHb_I = 2.86 # intrinsic balmer ratio 
    
    return R_mag, np.log10(balmer_ratio/HaHb_I), FUV_NUV


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
    DEM()
    #Observables()
