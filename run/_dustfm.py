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


if __name__=="__main__": 
    #fm_fuv('simba')
    fm_colormag('simba')
