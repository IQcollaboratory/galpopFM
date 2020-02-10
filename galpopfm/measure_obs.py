'''

module for observational measurements 


'''
import os 
import numpy as np 

# some constants 
lsun    = 3.839e33
pc2cm   = 3.08568e18
mag2cgs = np.log10(lsun/4.0/np.pi/(pc2cm**2)/100.0)

H0  = 2.2685455e-18 # 1/s (70. km/s/Mpc) 
c   = 2.9979e10 # cm/s

def mag(wave, spec, redshift=0.05, band='r_sdss'):
    ''' **THIS DOES NOT WORK YET!**
    **THIS DOES NOT WORK YET!**
    **THIS DOES NOT WORK YET!**

    calculate the magnitude of a single photometric band 

    :param wave:
        wavelength in angstrom 
    :param spec: 
        flux
    :param redshift: 
        
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

    trans = through_wave / tsum(wave[wlim], through_wave / wave[wlim]) # transmission 
    
    _mag = -2.5 * np.log10(tsum(wave[wlim], spec[:,wlim] * trans / wave[wlim])) #- 48.60 - 2.5 * mag2cgs
    #       mags(i) = TSUM(spec_lambda,tspec*bands(:,i)/spec_lambda)
    #       mags(i) = -2.5*LOG10(mags(i)) - 48.60 - 2.5*mag2cgs 

    return _mag


def AbsMag_sed(wave, sed, band='r_sdss'): 
    ''' calculate rest-frame absolute magnitude given SED for the specified band 

    :param wave: 
        wavelength in angstroms
    :param sed: 
        sed fluxes in Lsun/Hz 
    :param band: 
        specified band (default: 'r_sdss') 
    '''
    # read throughput 
    through = throughput(band) 

    wmin_through = through[0,:].min() 
    wmax_through = through[0,:].max() 

    if (wave.min() > wmin_through) or (wave.max() < wmax_through): return 0. 
    
    wlim = (wave >= wmin_through) & (wave <= wmax_through) # throughput wavelength limit 

    through_wave = np.interp(wave[wlim], through[0,:], through[1,:]) 

    trans = through_wave / tsum(wave[wlim], through_wave / wave[wlim]) # transmission 
    
    _sum = tsum(wave[wlim], sed[:,wlim] * trans / wave[wlim])

    _mag = -2.5 * np.log10(tsum(wave[wlim], sed[:,wlim] * trans / wave[wlim])) - 48.60 - 2.5 * mag2cgs
    return _mag  


def throughput(band): 
    ''' throughput of specified band  

    :param band: 
        string specifying the band 
    :return through: 
        2 x Nwave array that specifies [wavelength (Ang), throughput] 

    notes
    -----
    * taks ~7 seconds 
    '''
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
    assert band in band_dict.keys() 

    fband = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', '%s.dat' % band_dict[band])
    through = np.loadtxt(fband, skiprows=1, unpack=True, usecols=[0,1]) 
    return through 


def A_FUV(fmag, nmag, rmag):
    ''' Calculate attenuation of FUV A_FUV based on Salim+2007 Eq.(5)  

    :param fmag: 
        rest-frame (absolute) FUV magnitudee  
    :param nmag: 
        rest-frame (absolute) NUV magnitudee  
    :param rmag: 
        rest-frame (absolute) r-band magnitudee  
    '''
    fmag = np.atleast_1d(fmag) 
    nmag = np.atleast_1d(nmag) 
    rmag = np.atleast_1d(rmag) 

    f_n  = fmag - nmag # F-N 
    n_r  = nmag - rmag   # N-r
    
    afuv = np.zeros(fmag.shape) 
    afuv[n_r >= 4.] = 3.37 
    afuv[(n_r >= 4.) & (f_n < 0.95)] = 3.32 * f_n[(n_r >= 4.) & (f_n < 0.95)] + 0.22

    afuv[n_r < 4.] = 2.96
    afuv[(n_r < 4.) & (f_n < 0.90)] = 2.99 * f_n[(n_r < 4.) & (f_n < 0.90)] +0.27
    return afuv


def tsum(xin, yin): 
    ''' simple trapezoidal integration of tabulated function (xin,yin)
    used by FSPS conroy 

    source
    ------
    * https://github.com/cconroy20/alf/blob/master/src/tsum.f90
    '''
    nn = len(xin) 
    yin = np.atleast_2d(yin)
    tsum = np.sum((xin[1:] - xin[:-1]) * (yin[:,1:] + yin[:,:-1])/2., axis=1)
    return tsum


def flux_convert(wave_rest, sed, redshift=0.05): 
    ''' convert sed fluxes that come out of FSPS in units of Lun/Hz to 
    erg / s / cm^2 / Ang. 
    
    :param wave_rest: 
        rest-frame wavelenght in angstrom
    :param sed: 

    :param redshift: 
        redshift of galaxy 
    :return flux:
        flux in units of erg/s/cm^2/A
    '''
    wave_obs = wave_rest * (1. + redshift)
    flux = sed / (4.*np.pi * (redshift * c/H0)**2) / wave_rest**2 * c * 3.828e+41 # erg/s/cm^2/A
    return wave_obs, flux 
