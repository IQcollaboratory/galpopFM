'''

module for interfacing with astrologs 

'''
import numpy as np 
from astrologs.astrologs import Astrologs 


def Catalog(name): 
    if name == 'nsa': 
        return _nsa(name) 
    elif name in ['simba', 'tng', 'eagle']: 
        return _sim_sed(name)
    else: 
        raise ValueError 


def _nsa(name): 
    ''' read in the NSA catalog and impose selection critera 
    '''
    nsa = Astrologs('nsa') 

    # selection criteria 
    cut_redshift = (nsa.data['redshift'] > 0.01) & (nsa.data['redshift'] < 0.055) # redshift range 
    # lazy way to simplify the comoving volume calculation
    cut_footprint = (
            (nsa.data['ra'] > 130.) & (nsa.data['ra'] < 235) & 
            (nsa.data['dec'] > 0) & (nsa.data['dec'] < 56)) 
    # conservative absolute magnitude cut for completeness (this sample is
    # roughly M_* complete avove 10^9.7 Msun 
    cut_absmag = (nsa.data['M_r'] < -19.5)
    cuts = (cut_redshift & cut_footprint & cut_absmag)

    # impose selection cut on the sample 
    nsa.select(cuts) 

    nsa.cosmic_volume = 2558519.7 # (Mpc/h)^3

    return nsa 


def _sim_sed(name): 
    ''' read in the simulated seds using astrolog while correcting the NaNs for
    log SFR 
    '''
    simsed = Astrologs('simsed', sim=name) 
    
    if 'logsfr.100' in simsed.data.keys(): 
        isnan = (~np.isfinite(simsed.data['logsfr.100']))
        simsed.data['logsfr.100'][isnan] = -999.

    isnan = (~np.isfinite(simsed.data['logsfr.inst']))
    simsed.data['logsfr.inst'][isnan] = -999.
    
    vol = {'simba': 100.**3, 'tng': 75.**3, 'eagle': 67.77**3}  
    simsed.cosmic_volume = vol[name] 
    return simsed
