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
    ''' 
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
    flux_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            sed['mstar'], 
            dem='slab_calzetti') 

    # observational measurements 
    #A_V = measureObs.A_V(
    f_mag = measureObs.mag(sed['wave'], flux_dusty, band='galex_fuv') 
    n_mag = measureObs.mag(sed['wave'], flux_dusty, band='galex_nuv') 
    r_mag = measureObs.mag(sed['wave'], flux_dusty, band='r_sdss') 
    # noise model somewhere here
    a_fuv = measureObs.A_FUV(f_mag, n_mag, r_mag) 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.scatter(f_mag, a_fuv, c='k', s=1)
    ffig = fhdf5.replace('.hdf5', '.fuv_afuv.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


if __name__=="__main__": 
    fm_fuv('simba')
