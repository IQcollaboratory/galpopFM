'''



'''
import h5py 
import numpy as np 
from . import dustfm as dustFM
from . import measure_obs as measureObs

def dust_abc(name, dem='slab_calzetti'):
    '''
    '''
    # read in observations 


    # read SED for sims 
    sim_sed = _read_sed(name) 
    
    def rho(tt): 
        distance_metric(tt, sim_sed, obs, dem=dem) 


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
    # noise model here

    # observational measurements  
    A_V = measureObs.A_V(
    A_FUV = measureObs.A_FUV(
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
