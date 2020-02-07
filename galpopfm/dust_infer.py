'''



'''
import h5py 
import numpy as np 
from . import dustfm as dustFM
from . import measure_obs as measureObs

dat_dir = os.environ['GALPOPFM_DIR']


def dust_abc(name, dem='slab_calzetti'):
    '''
    '''
    # read in observations 


    # read SED for sims 
    sim_sed = _read_sed(name) 
    
    def rho(tt): 
        return distance_metric(tt, sim_sed, obs, dem=dem) 

    # abc here 


def distance_metric(theta, sed, obs, dem='slab_calzetti'): 
    '''
    '''
    flux_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['spec_noneb'], 
            sed['sed_onlyneb'], 
            sed['mstar'], 
            dem=dem) 

    # observational measurements 
    #A_V = measureObs.A_V(
    f_mag = measureObs.mag(sed['wave'], flux_dusty, band='galex_fuv') 
    n_mag = measureObs.mag(sed['wave'], flux_dusty, band='galex_nuv') 
    r_mag = measureObs.mag(sed['wave'], flux_dusty, band='r_sdss') 
    # noise model somewhere here
    a_fuv = measureObs.A_FUV(f_mag, n_mag, r_mag) 
    #a_v   = measureObs.A_V() 

    # calculate the distance 
    
    return 


def _read_sed(name): 
    ''' read in sed files 
    '''
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
    return sed
