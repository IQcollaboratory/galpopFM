'''

foward modeling dust with empirical prescriptions for assigning
attenuation curves for forward modeled galaxies 

'''
import numpy as np 


def Attenuate(theta, lam, spec_noneb, spec_neb, logmstar, logsfr, dem='slab_calzetti'): 
    ''' DEM attenuation wrapper. Imposes the following attenuation 

    Fo = Fi^star * 10^{-0.4 A(lambda)} + Fi^neb * 10^{-0.4 A_neb(lambda)}

    A_neb = f_neb x A(lambda) 

    :param theta: 
        8 free parameter of the slab + Noll+(2009) model
        theta: m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E f_nebular
    :param lam: 
        wavelength in angstrom
    :param spec_noneb: 
        sed without nebular emission lines 
    :param spec_neb: 
        sed with emission lines 
    :param logmstar: 
        log M* of galaxies 
    :param logsfr: 
        log SFR of galaxies
    :param dem: 
        string specifying the name of DEM 
    '''
    nspec = spec_noneb.shape[0] 
    assert spec_neb.shape[0] == nspec
    nwave = lam.shape[0] 
    assert spec_neb.shape[1] == nwave
    assert spec_noneb.shape[1] == nwave
    assert logmstar.shape[0] == nspec
    assert logsfr.shape[0] == nspec

    if dem == 'slab_calzetti': 
        mdust = DEM_slabcalzetti
    elif dem == 'slab_noll_m': 
        mdust = DEM_slab_noll_m
    elif dem == 'slab_noll_msfr': 
        mdust = DEM_slab_noll_msfr
    elif dem == 'slab_noll_simple': 
        mdust = DEM_slab_noll_simple
    else: 
        raise NotImplementedError

    # apply attenuation curve to spectra without nebular emissoin
    spec_noneb_dusty    = np.zeros(spec_noneb.shape) 
    spec_neb_dusty      = np.zeros(spec_neb.shape) 
    for i in range(nspec):  
        spec_noneb_dusty[i,:]   = mdust(theta, lam, spec_noneb[i,:],
                logmstar[i], logsfr[i], nebular=False) 
        spec_neb_dusty[i,:]     = mdust(theta, lam, spec_neb[i,:], 
                logmstar[i], logsfr[i], nebular=True)

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
        9 free parameter of the slab + Noll+(2009) model
        theta: m_tau1 m_tau2 c_tau m_delta1 m_delta2 c_delta m_E c_E f_nebular
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param logmstar: 
        log M* of galaxies 
    :param logsfr: 
        log SFR of galaxies
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.
    '''
    assert theta.shape[0] == 9, print(theta) 

    if logsfr == -999.: # if SFR = 0 no attenuation
        return flux_i 

    logmstar = np.atleast_1d(logmstar) 
    logsfr = np.atleast_1d(logsfr) 

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1] * logsfr + theta[2],
            1e-3, None) 

    delta = theta[3] * (logmstar - 10.) + theta[4] * logsfr + theta[5] 

    E_b = theta[6] * delta + theta[7] 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    assert np.isfinite(A_V), print(tauV, _slab, logmstar, logsfr) 
    
    dlam = 350. # width of bump from Noll+(2009)
    lam0 = 2175. # wavelength of bump 
    k_V_calzetti = 4.87789
    
    # bump 
    D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)
    
    # calzetti is already normalized to k_V
    A_lambda = A_V * (calzetti_absorption(lam) + D_bump / k_V_calzetti) * \
            (lam / 5500.)**delta 

    if not nebular: factor = 1.
    else: factor = theta[8] 

    T_lam = 10.0**(-0.4 * A_lambda * factor)

    return flux_i * T_lam 


def DEM_slab_noll_m(theta, lam, flux_i, logmstar, logsfr, nebular=True): 
    ''' Dust empirical model that combines the slab model with Noll+(2009)

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = m_tau (log M* - 10.) + c_tau
    delta   = m_delta  (log M* - 10.) + c_delta         -2.2 < delta < 0.4
    E_b     = m_E delta + c_E

    :param theta: 
        7 free parameter of the slab + Noll+(2009) model
        theta: m_tau c_tau m_delta c_delta m_E c_E f_nebular
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param logmstar: 
        logmstar of galaxy 
    :param logsfr: 
        logSFR of galaxy (not used in this DEM 
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.
    '''
    assert theta.shape[0] == 7, print(theta)
    logmstar = np.atleast_1d(logmstar) 

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1], 1e-3, None) 

    delta = theta[2] * (logmstar - 10.) + theta[3] 

    E_b = theta[4] * delta + theta[5] 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl))
    assert np.isfinite(A_V), print(tauV, _slab, logmstar, logsfr) 
    
    dlam = 350. # width of bump from Noll+(2009)
    lam0 = 2175. # wavelength of bump 
    k_V_calzetti = 4.87789
    
    # bump 
    D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)

    A_lambda = A_V * (calzetti_absorption(lam) + D_bump / k_V_calzetti) * \
            (lam / 5500.)**delta 

    if not nebular: factor = 1.
    else: factor = theta[6] 

    T_lam = 10.0**(-0.4 * A_lambda * factor)

    return flux_i * T_lam 


def DEM_slabcalzetti(theta, lam, flux_i, logmstar, logsfr, nebular=True): 
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

    tauV = np.clip(theta[0] * (logmstar - 10.) + theta[1], 1e-3, None) 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 
    #cosis = 1.0 - np.cos(np.random.uniform(low=0, high=0.5*np.pi, size=mstar.shape[0]))
    
    if not nebular: factor = 1.
    else: factor = theta[2] 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    assert np.isfinite(A_V), print(tauV, _slab, logmstar, logsfr) 
    # minimum attenuation from Romeel's paper (which one?) 
    A_V = np.clip(A_V, 0.1, None) 

    T_lam = 10.0**(A_V * -0.4 * calzetti_absorption(lam) * factor)
    return flux_i * T_lam


def DEM_slab_noll_simple(theta, lam, flux_i, logmstar, logsfr, nebular=True): 
    ''' simplified version of the Dust empirical model that combines the slab
    model with Noll+(2009). This is to better understand the distance metrics

    A(lambda) = -2.5 log10( (1 - exp(-tauV sec(i))) / (tauV sec(i)) ) x 
                    (k'(lambda) + D(lambda, E_b))/k_V x 
                    (lambda / lambda_V)^delta

    tauV    = c_tau
    delta   = c_delta         -2.2 < delta < 0.4
    E_b     = constant 

    :param theta: 
        2 free parameter of the simplified slab + Noll+(2009) model
        theta: c_tau c_delta f_nebular
    :param lam: 
        wavelength in angstrom
    :param flux_i: 
        intrinsic flux of sed (units don't matter) 
    :param logmstar: 
        log M* of galaxies 
    :param logsfr: 
        log SFR of galaxies
    :param nebular: 
        if True nebular flux has an attenuation that is scaled from the
        continuum attenuation.
    '''
    assert theta.shape[0] == 2, print(theta) 

    logmstar = np.atleast_1d(logmstar) 
    logsfr = np.atleast_1d(logsfr) 

    tauV = np.clip(theta[0], 1e-3, None) 
    delta = theta[1] 
    E_b = 0.85 
    
    # randomly sample the inclinatiion angle from 0 - pi/2 
    incl = np.random.uniform(0., 0.5*np.pi, size=logmstar.shape[0])
    sec_incl = 1./np.cos(incl) 

    #Eq. 14 of Somerville+(1999) 
    A_V = -2.5 * np.log10((1.0 - np.exp(-tauV * sec_incl)) / (tauV * sec_incl)) 
    assert np.isfinite(A_V), print(tauV, _slab, logmstar, logsfr) 
    
    dlam = 350. # width of bump from Noll+(2009)
    lam0 = 2175. # wavelength of bump 
    k_V_calzetti = 4.87789
    
    # bump 
    D_bump = E_b * (lam * dlam)**2 / ((lam**2 - lam0**2)**2 + (lam * dlam)**2)
    
    # calzetti is already normalized to k_V
    A_lambda = A_V * (calzetti_absorption(lam) + D_bump / k_V_calzetti) * \
            (lam / 5500.)**delta 

    T_lam = 10.0**(-0.4 * A_lambda)

    return flux_i * T_lam 


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
