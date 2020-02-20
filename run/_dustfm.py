'''

script to check galpopfm.dustfm

'''
import os 
import h5py 
import numpy as np 
# -- galpopfm --
from galpopfm import dustfm as dustFM
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


def fm_fuv(name): 
    ''' check A_FUV measurements from the forward modeled SEDs and compare
    with Fig.4 of Salim+(2018) 

    notes
    -----
    * uhhh match is not great! 
    '''
    dat_dir = os.environ['GALPOPFM_DIR']
    if name == 'simba': 
        fhdf5 = os.path.join(dat_dir, 'sed', 'simba.hdf5') 
    else: 
        raise NotImplementedError

    f = h5py.File(fhdf5, 'r') 
    sed = {} 
    sed['wave']         = f['wave'][...] 
    sed['sed_neb']      = f['sed_neb'][...]
    sed['sed_noneb']    = f['sed_noneb'][...]
    sed['sed_onlyneb']  = sed['sed_neb'] - sed['sed_noneb'] # only nebular emissoins 
    sed['mstar']        = f['mstar'][...] 
    f.close() 
    
    theta = np.array([0.1, 0.2, 1./0.44]) 
    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            np.log10(sed['mstar']),
            dem='slab_calzetti') 

    # observational measurements 
    F_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_fuv') 
    N_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_nuv') 
    R_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 
    # noise model somewhere here
    a_fuv = measureObs.A_FUV(F_mag, N_mag, R_mag) 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.log10(sed['mstar']), a_fuv, c='k', s=1)
    sub.set_xlabel('$\log\,M_*$', fontsize=20) 
    sub.set_xlim(8., 12.) 
    sub.set_ylabel(r'$A_{\rm FUV}$', fontsize=20) 
    sub.set_ylim(-0.05, 4.) 
    ffig = fhdf5.replace('.hdf5', '.fuv_afuv.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def fm_colormag(name): 
    ''' check the color magnitude diagram of forward modeled 
    '''
    dat_dir = os.environ['GALPOPFM_DIR']
    if name == 'simba': 
        fhdf5 = os.path.join(dat_dir, 'sed', 'simba.hdf5') 
    else: 
        raise NotImplementedError

    f = h5py.File(fhdf5, 'r') 
    sed = {} 
    sed['wave']         = f['wave'][...] 
    sed['sed_neb']      = f['sed_neb'][...]
    sed['sed_noneb']    = f['sed_noneb'][...]
    sed['sed_onlyneb']  = sed['sed_neb'] - sed['sed_noneb'] # only nebular emissoins 
    sed['mstar']        = f['mstar'][...] 
    f.close() 
    
    sed_dusty = sed['sed_neb'] 
    import time
    t0 = time.time() 
    theta = np.array([0.1, 0.2, 1./0.44]) 
    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            np.log10(sed['mstar']), 
            dem='slab_calzetti') 
    print('dust attenuation takes %.2f sec' % (time.time()-t0))

    t0 = time.time() 
    # observational measurements 
    g_mag = measureObs.AbsMag_sed(sed['wave'], sed['sed_neb'], band='g_sdss') 
    print('magnitude calculation takes %.2f sec' % (time.time()-t0))
    r_mag = measureObs.AbsMag_sed(sed['wave'], sed['sed_neb'], band='r_sdss') 
    # dusty magnitudes
    g_mag_dusty = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='g_sdss') 
    r_mag_dusty = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.scatter(r_mag, g_mag - r_mag, c='k', s=0.1, label='No dust')
    sub.scatter(r_mag_dusty, g_mag_dusty - r_mag_dusty, c='C1', s=0.1, label='w/ dust')
    sub.legend(loc='upper left', fontsize=15) 
    sub.set_xlabel('$M_r$', fontsize=20) 
    sub.set_xlim(-15., -24.) 
    sub.set_ylabel('$M_g - M_r$', fontsize=20) 
    sub.set_ylim(-0.2, 1.2) 
    ffig = fhdf5.replace('.hdf5', '.gr_r.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def fm_AV(name): 
    ''' examine slab dust model V-band attenuation A_V(Mstar) which is used in
    forward model and compare it to A_V from SDSS.
    '''
    # read data 
    dat_dir = os.environ['GALPOPFM_DIR']
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    # get SDSS attenuation A_V 
    AV_sdss     = sdss['AV'][...] 
    mstar_sdss  = sdss['MASS'][...] * 0.7**2 
    
    cut_sdss = (AV_sdss > 0.) 
    
    # read simulation 
    if name == 'simba': 
        fhdf5 = os.path.join(dat_dir, 'sed', 'simba.hdf5') 
    else: 
        raise NotImplementedError
    f = h5py.File(fhdf5, 'r') 
    logmstar_sim   = f['logmstar'][...] 
    f.close() 

    tauv = np.clip((logmstar_sim - 10.) + 1.5, 0., None) 

    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar_sim.shape[0])
    sec_incl = 1./np.cos(incl) 
    T_V = (1.0 - np.exp(-tauv * sec_incl)) / (tauv * sec_incl) #Eq. 14 of Somerville+(1999) 
    AV_sim = -2.5 * np.log10(T_V)

    import corner as DFM 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    DFM.hist2d(np.log10(mstar_sdss)[cut_sdss], AV_sdss[cut_sdss], color='k', 
            levels=[0.68, 0.95], range=[[8.5, 11.], [0., 4.]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(logmstar_sim, AV_sim, color='C1', 
            levels=[0.68, 0.95], range=[[8.5, 11.], [0., 4.]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 

    #marr = np.linspace(8, 12, 20) 
    #_tauv = (marr-10.) + 1.5
    #sub.plot(marr, 1.086 * _tauv, c='C0', ls='--') 
    sub.set_xlabel(r'$\log\,M_*$', fontsize=20) 
    sub.set_ylabel(r'$A_V$', fontsize=20) 
    ffig = fhdf5.replace('.hdf5', '.AV_mstar.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def obs_tauV(): 
    ''' examine optical depth tauV(Mstar) for SDSS data assuming face on (theta=0) slab dust model
    tau_V = A_V / 1.086
    '''
    # read data 
    dat_dir = os.environ['GALPOPFM_DIR']
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    # get attenuation A_V 
    AV_sdss     = sdss['AV'][...] 
    mstar_sdss  = sdss['MASS'][...] * 0.7**2 

    # there should be some cut here 
    cut = AV_sdss > 0. 
    AV_sdss     = AV_sdss[cut]
    mstar_sdss  = mstar_sdss[cut]
    
    # get tauV 
    tauV = AV_sdss/1.086     
    sensible_tau = (tauV > 0.) & (tauV < 10.) 

    import corner as DFM 
    fig = plt.figure()
    sub = fig.add_subplot(111) 
    DFM.hist2d(np.log10(mstar_sdss)[sensible_tau], tauV[sensible_tau], color='k', 
            levels=[0.68, 0.95], range=[[8.5, 11.], [0., 4.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    # overplot tauV(msta) within our prior 
    marr = np.linspace(8, 12, 20) 
    _tauv = (marr-10.) + 1.5
    sub.plot(marr, _tauv, c='C1', ls='--') 

    sub.set_xlabel(r'$\log\,M_*$', fontsize=20) 
    sub.set_ylabel(r'$\tau_V = A_V/1.086$', fontsize=20) 
    fig.savefig(os.path.join(dat_dir, 'obs', 'sdss.tauV_mstar.png'), bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    #fm_fuv('simba')
    #fm_colormag('simba')
    fm_AV('simba') 
    #obs_tauV()
