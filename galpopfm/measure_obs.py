'''

module for observational measurements 


'''
import numpy as np 


def mag(wave, spec, band='r_sdss'):
    ''' calculate the magnitude of band 

    :param wave:
        wavelength in angstrom 
    :param spec: 
        flux
    :param band: 
        
    :return _mag: 
        magnitude of specifid band  
    '''
    # read throughput 
    through = throughput(band) 

    wmin_through = through[0,:].min() 
    wmax_through = through[0,:].max() 

    if (wave.min() > wmin_through) or (wave.max() < wmax_through): return 0. 
    
    wlim = (wave >= wmin_through) & (wave <= wmax_through) # throughput wavelength limit 

    #interp_band_r = interp.interp1d(SDSS_r_through[:,0], SDSS_r_through[:,1])

    through_wave = np.interp(wave[wlim], through[0,:], through[1,:]) 

    trans_r = through_wave * 1./tsum(wave[wlim], through_wave / wave[bandw_r])
    
    _mag = -2.5 * np.log10(tsum(wave[wlim], spec[wlim] * trans_r / wave[wlim])) - 48.60 - 2.5 * mag2cgs
    return _mag


def throughput(band): 
    ''' throughput of specified band  

    :param band: 
        string specifying the band 
    :return through: 
        2 x Nwave array that specifies [wavelength, throughput] 
    '''
    if band not in ['r_sdss']: raise NotImplementedError

    # something here. 

    return through 


def A_FUV(fmag, nmag, rmag):
    ''' Calculate attenuation of FUV 
    '''
    fmag = np.atleast_1d(fmag) 
    nmag = np.atleast_1d(nmag) 
    rmag = np.atleast_1d(rmag) 

    f_n  = fmag - nmag # F-N 
    n_r  = nmag - rmag   # N-r
    
    afuv = np.zeros(fmag.shape) 
    afuv[n_r >= 4.] = 3.37 
    afuv[(n_r >= 4.) & (f_n < 0.95)] = 3.32 * f_n + 0.22

    afuv[n_r < 4.] = 2.96
    afuv[(n_r < 4.) & (f_n < 0.90)] = 2.99 * f_n +0.27
    return afuv
