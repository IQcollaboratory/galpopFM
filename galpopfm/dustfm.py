'''

foward modeling dust with empirical prescriptions for assigning
attenuation curves for forward modeled galaxies 

'''
import numpy as np 


def Attenuate(theta, lam, spec_noneb, spec_neb, mstar, dem='slab_calzetti'): 
    ''' 
    '''
    nspec = spec_noneb.shape[0] 
    assert spec_neb.shape[0] == nspec
    nwave = lam.shape[0] 
    assert spec_neb.shape[1] == nwave
    assert spec_noneb.shape[1] == nwave

    if dem == 'slab_calzetti': 
        mdust = DEM_slabcalzetti
    elif dem == 'slab_noll_m': 
        mdust = DEM_slab_noll 
    elif dem == 'slab_noll_msfr': 
        mdust = DEM_slab_noll_m
    else: 
        raise NotImplementedError

    # apply attenuation curve to spectra without nebular emissoin
    spec_noneb_dusty    = np.zeros(spec_noneb.shape) 
    spec_neb_dusty      = np.zeros(spec_neb.shape) 
    for i in range(nspec):  
        spec_noneb_dusty[i,:]   = mdust(theta, lam, spec_noneb[i,:], mstar[i], nebular=False) 
        spec_neb_dusty[i,:]     = mdust(theta, lam, spec_neb[i,:], mstar[i], nebular=True)

    spec_dusty = spec_noneb_dusty + spec_neb_dusty 
    return spec_dusty 


def DEM_slab_noll_msfr(theta, lam, flux_i, logmstar, logsfr, nebular=True): 
    ''' Dust empirical model that combines the slab model with Noll+(2009)

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = m_tau1 (log M* - 10.) + m_tau2 logSFR + c_tau
    delta   = m_delta1  (log M* - 10.) + m_delta2 logSFR + c_delta         -2.2 < delta < 0.4
    E_b     = m_E delta + c_E

    :param theta: 
        8 free parameter of the slab + Noll+(2009) model
        theta[0]: m_tau1
        theta[1]: m_tau2
        theta[2]: c_tau
        theta[3]: m_delta1
        theta[4]: m_delta2
        theta[5]: c_delta
        theta[6]: m_E
        theta[7]: c_E
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.
    '''
    logmstar = np.atleast_1d(logmstar) 
    logsfr = np.atleast_1d(logsfr) 

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1] * logsfr + theta[2], 0., None) 

    delta = theta[3] * (logmstar - 10.) + theta[4] * logsfr + theta[5] 

    E_b = theta[6] * delta + theta[7] 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    
    dlam = 350. # width of bump from Noll+(2009)
    lam0 = 2175. # wavelength of bump 
    k_V_calzetti = 4.87789
    
    # bump 
    D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)

    A_lambda = A_V * (calzetti_absorption(lam) + D_bump) / k_V_calzetti * \
            (lam / 5500.)**delta 

    T_lam = 10.0**(-0.4 * A_lambda * factor)

    return flux_i * T_lam 


def DEM_slab_noll_m(theta, lam, flux_i, logmstar, nebular=True): 
    ''' Dust empirical model that combines the slab model with Noll+(2009)

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = m_tau (log M* - 10.) + c_tau
    delta   = m_delta  (log M* - 10.) + c_delta         -2.2 < delta < 0.4
    E_b     = m_E delta + c_E

    :param theta: 
        6 free parameter of the slab + Noll+(2009) model
        theta[0]: m_tau
        theta[1]: c_tau
        theta[2]: m_delta
        theta[3]: c_delta
        theta[4]: m_E
        theta[5]: c_E
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.
    '''
    logmstar = np.atleast_1d(logmstar) 

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1], 0., None) 

    delta = theta[2] * (logmstar - 10.) + theta[3] 

    E_b = theta[4] * delta + theta[5] 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    
    dlam = 350. # width of bump from Noll+(2009)
    lam0 = 2175. # wavelength of bump 
    k_V_calzetti = 4.87789
    
    # bump 
    D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)

    A_lambda = A_V * (calzetti_absorption(lam) + D_bump) / k_V_calzetti * \
            (lam / 5500.)**delta 

    T_lam = 10.0**(-0.4 * A_lambda * factor)

    return flux_i * T_lam 


def DEM_slabcalzetti(theta, lam, flux_i, logmstar, nebular=True): 
    ''' Dust Empirical Model that uses the slab model with tauV(theta, mstar)
    parameterization with inclinations randomly sampled 

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                (k'(lambda) / k_V)
    
    :param theta: 
        parameter of the DEM model that specifies the M* dep. V-band optical depth (slope, offset) 
        as well as the nebular flux attenuatoin fraction
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.

    notes
    -----
    *    slab model to apply dust attenuation Eq.14 (Somerville+1999) 
    '''
    logmstar = np.atleast_1d(logmstar) 

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1], 0., None) 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 
    #cosis = 1.0 - np.cos(np.random.uniform(low=0, high=0.5*np.pi, size=mstar.shape[0]))
    
    if not nebular: factor = 1.
    else: factor = theta[2] 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    # minimum attenuation from Romeel's paper (which one?) 
    A_V = np.clip(A_V, 0.1, None) 

    T_lam = 10.0**(A_V * -0.4 * calzetti_absorption(lam) * factor)
    return flux_i * T_lam


def slab_salim(lam, tauV, mag, sec_incl, mstar, factor=1.):
    ''' slab model to apply dust attenuation based on Rachel's code

    :param lam: 
        wavelength in angstrom 
    :param tauV: 
        attenuation in V (to be parameterized as a function of stellar mass) 
        in Rachel's model this is parametrized by Mgas, gas metallicity and galaxy size, which all depend
        on stellar mass.
    :param mag: 
        flux 
    :param sec_incl: 
        sec(inclination), which should be randomly sampled 
    :param factor: 
        scale factor for the attenuation for the nebular emission 
    '''
    #cosi_ave = np.sqrt(3.0)/2.0
    #fraction of light transmitted in V-band (slab)
    ##use a slab model, with average (constant) inclination
    #T_V = (1.0-np.exp(-tauV/cosi_ave))/(tauV/cosi_ave)
    ##use a slab model, with randomly assigned inclination
    T_V = (1.0 - np.exp(-tauV/incl)) / (tauV/incl)

    AV = -2.5 * np.log10(T_V)

    if tauV == 0 or AV == 0:
        AV = 0.1 # this can be a free parameters (0.1 is from Romeel's paper as the minimum attenuation) 
        newmag = mag
    else:
        T_lam = 10.0**(AV * -0.4 * salim_absorption(lam, mstar) * factor)
        newmag = mag * T_lam
    return newmag


def salim_absorption(lam, mstar):
    ''' Modified Calzetti(2000) attenuation curve with a variable slope and optinal UV bump 
     see Eq.3 of Salim+(2018). 

    delta is the exponent of a power-law curve centered at the 
    V-band with which the Calzetti curve is multiplied:

    delta = -0.38 + 0.29*(np.log10(Mstar) - 10)

    B is the amplitude of the UV bump as described by a Drude profile D_lam (Fitzpatrick & Massa 1986)

    Both delta and B should be determined through fitting functions to galaxy parameters, but 
    for B this is not provided really:

    The avarage B is 1.3. Provided in the table: 


    :param lam: 
        wavelength in angstroms. (np.ndarray) 
    :param mstar: 
        stellar mass of galaxies. (np.ndarray) 

    notes
    -----
    * A_V_young/A_V_old = 2.27 or EBV_old = 0.44 EBV_young 
    '''
    lam = np.atleast_1d(lam) / 1.e4
    
    # Salim+(2018) Table 1: Functional Fits of Dust Attenuation Curves 
    B = 1.3 # average 
    #B = 1.57 # average SF galaxie 
    #B = 2.62 # SF galaxies 8.5 < log M* < 9.5
    #B = 1.73 # SF galaxies 9.5 < log M* < 10.5
    #B = 1.09 # SF galaxies 10.5 < log M* < 11.5 
    #B = 2.27 # average high z
    #B = 2.74 # high z log M* < 10.
    #B = 2.11 # high z log M* > 10. 
    #B = 2.21 # Q galaxies 

    delta = -0.38 + 0.29 * (mstar - 10.) # Salim+(2018) Eq.6 
    #delta = -0.45 + 0.19 * (mstar - 10.) # Salim+(2018) Eq. 7 for high redshift 
    
    # calzetti+(2000) attenuation curve
    k_Cal = calzetti_absorption(lam)

    R_V_Cal = 4.05
    R_V_mod = R_V_Cal/((R_V_Cal + 1) * (4400./5500.)**delta - R_V_Cal)

    D_lam = B * lam**2 * (0.35)**2/((lam**2 - 0.2175**2)**2 + lam**2 * 0.35**2)

    k_mod = k_Cal * R_V_mod/R_V_Cal * (lam / 5500.)**delta + D_lam
    return k_mod


def salim_absorption_poly(lam):
    ''' polynomial fit to Salim+(2018)'s modified Calzetti(2000) attenuation curve.
    See Eq.8-10 in Salim+(2018) 

    :param lam: 
        wavelength in angstroms. (np.ndarray) 
    :param mstar: 
        stellar mass of galaxies. (np.ndarray) 

    notes
    -----
    * A_V_young/A_V_old = 2.27 or EBV_old = 0.44 EBV_young 
    '''
    lam = np.atleast_1d(lam) / 1.e4
    
    # Salim+(2018) Table 1: Functional Fits of Dust Attenuation Curves 
    #B = 2.62 # SF galaxies 8.5 < log M* < 9.5
    #B = 1.73 # SF galaxies 9.5 < log M* < 10.5
    #B = 1.09 # SF galaxies 10.5 < log M* < 11.5 
    #B = 2.27 # average high z
    #B = 2.74 # high z log M* < 10.
    #B = 2.11 # high z log M* > 10. 
    #B = 2.21 # Q galaxies 
    
    # average SF galaxies values
    B   = 1.57 
    R_V = 3.15 
    a0  = -4.30 
    a1  = 2.71
    a2  = -0.191
    a3  = 0.0121

    D_lam = B * lam**2 * (0.35)**2/((lam**2 - 0.2175**2)**2 + lam**2 * 0.35**2)
    
    k_mod = np.zeros(lam.shape) 
    wlim = [(lam >= 0.09) & (lam < 2.2)]
    k_mod[wlim] = a0 + a1/lam[wlim] + a2/(lam[wlim]**2) + a3/(lam[wlim]**3) + D_lam[wlim] + R_V
    return k_mod


def calzetti_absorption(lam):
    ''' calzetti dust absorption attenuation curve normalized to 1 at V

    :param lam: 
        wavelength in Angstroms 

    notes
    -----
    * different versions from different papers...
        - k_lam = np.where(lam <= 0.63, 4.88+2.656*(-2.156+1.509/lam-0.198/(lam*lam)+0.011/(lam*lam*lam)), 1.73 + ((1.86 - 0.48/lam)/lam - 0.1)/lam)
    '''
    lam = np.atleast_1d(lam) / 1.e4 
    
    #R_V = 4.88 # (Calzetti 1997b)
    R_V = 4.05 # (Calzetti+2000) 
        
    # Calzetti+(2000) Eq.4
    k_lam = np.zeros(lam.shape)
    wlim = (lam >= 0.12) & (lam < 0.63)
    k_lam[wlim] = R_V + 2.659 * (-2.156 + 1.509/lam[wlim] - 0.198/(lam[wlim]**2) + 0.011/(lam[wlim]**3))
    wlim = (lam >= 0.63) & (lam <= 2.2)
    k_lam[wlim] = R_V + 2.659 * (-1.857 + 1.040/lam[wlim])

    k_V_calzetti = 4.87789
    return k_lam/k_V_calzetti
