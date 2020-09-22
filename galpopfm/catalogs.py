'''

module for interfacing with astrologs 

'''
import numpy as np 
from astrologs.astrologs import Astrologs 


def Catalog(name): 
    if name == 'tinker': 
        return _tinkergroup()
    elif name == 'nsa': 
        return _nsa() 
    elif name in ['simba', 'tng', 'eagle']: 
        return _sim_sed(name)
    else: 
        raise ValueError 


def _nsa(): 
    ''' read in the NSA catalog and impose selection critera 
    '''
    # read nsa catalog over the SDSS VAGC footprint 
    nsa = Astrologs('nsa', vagc_footprint=True) 

    # selection criteria 
    cut_redshift = (nsa.data['redshift'] > 0.01) & (nsa.data['redshift'] < 0.055) # redshift range 
    # conservative absolute magnitude cut for completeness (this sample is
    # roughly M_* complete avove 10^9.7 Msun 
    cut_absmag = (nsa.data['M_r'] < -19.5)
    cuts = (cut_redshift & cut_absmag)

    # impose selection cut on the sample 
    nsa.select(cuts) 
    
    # some extra meta data 
    nsa.footprint = 7818.28 # deg^2

    # from astropy.cosmology import Planck as cosmo 
    # ((cosmo.comoving_volume(0.055).value - cosmo.comoving_volume(0.01).value) * (7818.28/41252.96) * cosmo.h**3)
    nsa.cosmic_volume = 3401908.8 # (Mpc/h)^3

    return nsa 


def _tinkergroup(): 
    ''' read Jeremy's group catalog with M* limit of 10^9.7 Msun
    '''
    # read Jeremy's group catalog with M* = 10^9.7 Msun limit, cross matched
    # with NSA
    tinker = Astrologs('tinkergroup', mlim='9.7', cross_nsa=True) 
    
    # some extra meta data 
    tinker.footprint = 7818.28 # deg^2
    # from astropy.cosmology import Planck15 as cosmo
    # ((cosmo.comoving_volume(0.0334).value - cosmo.comoving_volume(0.0107).value) * (7818.28/41252.96) * cosmo.h**3))
    tinker.cosmic_volume = 751113.0 # (Mpc/h)^3
    return tinker


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
