'''

module for observational measurements 


'''
import numpy as np 

# some constants 
lsun    = 3.839e33
pc2cm   = 3.08568e18
mag2cgs = np.log10(lsun/4.0/np.pi/(pc2cm**2)/100.0)


def mag(wave, spec, band='r_sdss'):
    ''' calculate the magnitude of a single photometric band 

    :param wave:
        wavelength in angstrom 
    :param spec: 
        flux
    :param band: 
        string specifying the photometric band  
    :return _mag: 
        magnitude of specifid band  
    '''
    # read throughput 
    through = throughput(band) 

    wmin_through = through[0,:].min() 
    wmax_through = through[0,:].max() 

    if (wave.min() > wmin_through) or (wave.max() < wmax_through): return 0. 
    
    wlim = (wave >= wmin_through) & (wave <= wmax_through) # throughput wavelength limit 

    through_wave = np.interp(wave[wlim], through[0,:], through[1,:]) 

    trans = through_wave * 1./np.trapz(through_wave / wave[wlim], x=wave[wlim])
    
    _mag = -2.5 * np.log10(np.trapz(spec[wlim] * trans / wave[wlim], x=wave[wlim])) - 48.60 - 2.5 * mag2cgs
    return _mag


def throughput(band): 
    ''' throughput of specified band  

    :param band: 
        string specifying the band 
    :return through: 
        2 x Nwave array that specifies [wavelength (Ang), throughput] 
    '''
    if band not in ['V_johnson', 'r_sdss']: raise NotImplementedError

    band_dict = {
            'V_johnson': 'johnson_v',
            'galex_fuv': 'galex_fuv', 
            'galex_nuv': 'galex_nuv', 
            'u_sdss': 'sdss_u', 
            'g_sdss': 'sdss_g', 
            'r_sdss': 'sdss_r', 
            'i_sdss': 'sdss_i', 
            'z_sdss': 'sdss_z' 
            }
    fband = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%.dat' % band_dict[band])
    through = np.loadtxt(fband, skiprows=1, unpack=True usecols=[0,1]) 
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
