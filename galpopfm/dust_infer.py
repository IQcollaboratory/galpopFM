'''



'''
import os 
import sys 
import h5py 
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
# -- abcpmc -- 
import abcpmc
from abcpmc import mpi_util
# -- galpopfm --
from . import dustfm as dustFM
from . import measure_obs as measureObs

dat_dir = os.environ['GALPOPFM_DIR']


def dust_abc(name, T, eps0=[0.1, 1.], N_p=100, prior_range=None, dem='slab_calzetti', abc_dir=None, nthread=1):
    '''
    '''
    # read in observations 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    F_mag_sdss = sdss['ABSMAG'][...][:,0]
    N_mag_sdss = sdss['ABSMAG'][...][:,1]
    R_mag_sdss = sdss['ABSMAG'][...][:,4]
    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]
    
    x_obs = sumstat_obs(F_mag_sdss, N_mag_sdss, R_mag_sdss, Haflux_sdss, Hbflux_sdss, sdss['Z'][...])
    
    # read SED for sims 
    sim_sed = _read_sed(name) 

    # pass through the minimal amount of memory 
    wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 1e4) 
    
    # save as global variable that can be accessed by multiprocess 
    global shared_sim_sed
    shared_sim_sed = {} 
    shared_sim_sed['mstar'] = sim_sed['mstar'].copy()
    shared_sim_sed['wave']  = sim_sed['wave'][wlim].copy()
    shared_sim_sed['sed_noneb'] = sim_sed['sed_noneb'][:,wlim].copy() 
    shared_sim_sed['sed_onlyneb'] = sim_sed['sed_onlyneb'][:,wlim].copy() 

    #--- inference with ABC-PMC below ---
    # prior 
    prior_min = prior_range[0] 
    prior_max = prior_range[1] 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 
    
    # sampler 
    abcpmc_sampler = abcpmc.Sampler(
            N=N_p,                  # N_particles
            Y=x_obs,                # data
            postfn=_sumstat_model_shared,   # simulator 
            dist=distance_metric,   # distance metric 
            threads=nthread,
            postfn_kwargs={'dem': dem},
            dist_kwargs={'method': 'L2'}
            )      

    # threshold 
    eps = abcpmc.ConstEps(T, eps0) 
    print('eps0', eps.eps)

    pools = []
    for pool in abcpmc_sampler.sample(prior, eps):
        eps_str = ", ".join(["{0:>.4f}".format(e) for e in pool.eps])
        print("T: {0}, eps: [{1}], ratio: {2:>.4f}".format(pool.t, eps_str, pool.ratio))

        for i, (mean, std) in enumerate(zip(*abcpmc.weighted_avg_and_std(pool.thetas, pool.ws, axis=0))):
            print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
        print('dist', pool.dists)
        
        # write out theta, weights, and distances to file 
        writeABC('eps', pool, abc_dir=abc_dir)
        writeABC('theta', pool, abc_dir=abc_dir) 
        writeABC('w', pool, abc_dir=abc_dir) 
        writeABC('rho', pool, abc_dir=abc_dir) 

        # update epsilon based on median thresholding 
        eps.eps = np.median(pool.dists, axis=0)
        pools.append(pool)
        print('eps%i' % pool.t, eps.eps)
        print('----------------------------------------')
        if pool.ratio <0.2:
            break
    abcpmc_sampler.close()
    return None 


def distance_metric(x_obs, x_model, method='L2'): 
    ''' distance metric between forward model m(theta) and observations

    notes
    -----
    * we have implemented the simplest L2 Norm 
    ''' 
    med_fnuv_obs, med_balmer_obs = x_obs
    med_fnuv_mod, med_balmer_mod = x_model
    
    if method == 'L2': 
        # L2 norm of the median balmer ratio measurement log( (Ha/Hb)/(Ha/Hb)I )
        _finite = np.isfinite(med_balmer_mod) & np.isfinite(med_balmer_obs)
        if np.sum(_finite) == 0: 
            rho_balmer = np.Inf
        else: 
            rho_balmer = np.sum((med_balmer_mod[_finite] - med_balmer_obs[_finite])**2)/float(np.sum(_finite))

        # L2 norm of median FUV-NUV color 
        _finite = np.isfinite(med_fnuv_mod) & np.isfinite(med_fnuv_obs)
        if np.sum(_finite) == 0: 
            rho_fnuv = np.Inf
        else: 
            rho_fnuv = np.sum((med_fnuv_mod[_finite] - med_fnuv_obs[_finite])**2)/float(np.sum(_finite))

        return [rho_balmer, rho_fnuv] 
    else: 
        raise NotImplemented 


def sumstat_obs(Fmag, Nmag, Rmag, Haflux, Hbflux, z): 
    ''' calculate summary statistics for SDSS observations  
    '''
    FUV_NUV =  Fmag - Nmag
    Ha_sdss = Haflux * (4.*np.pi * (z * 2.9979e10/2.2685e-18)**2) * 1e-17
    Hb_sdss = Hbflux * (4.*np.pi * (z * 2.9979e10/2.2685e-18)**2) * 1e-17
    balmer_ratio = Ha_sdss/Hb_sdss 
    
    HaHb_I = 2.86 # intrinsic balmer ratio 
    _, med_fnuv = median_alongr(Rmag, FUV_NUV, rmin=-16., rmax=-24., nbins=16)
    _, med_balmer = median_alongr(Rmag, np.log10(balmer_ratio/HaHb_I), rmin=-16., rmax=-24., nbins=16)

    return [med_fnuv, med_balmer]


def _sumstat_model_shared(theta, dem='slab_calzetti'): 
    ''' wrapper for sumstat_model that works with shared memory? 
    '''
    return sumstat_model(theta, sed=shared_sim_sed, dem=dem) 


def sumstat_model(theta, sed=None, dem='slab_calzetti'): 
    ''' calculate summary statistics for forward model m(theta) 

    notes
    -----
    * still need to implement noise model
    '''
    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            np.log10(sed['mstar']),
            dem='slab_calzetti') 
    
    # observational measurements 
    F_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_fuv') 
    N_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_nuv') 
    R_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 
    FUV_NUV = F_mag - N_mag 
    # balmer measurements 
    Ha_dust, Hb_dust = measureObs.L_em(['halpha', 'hbeta'], sed['wave'], sed_dusty) 
    balmer_ratio = Ha_dust/Hb_dust
    # noise model somewhere here
    # noise model somewhere here
    # noise model somewhere here

    # calculate the distance 
    HaHb_I = 2.86 # intrinsic balmer ratio 
    _, med_fnuv = median_alongr(R_mag, FUV_NUV, rmin=-16., rmax=-24., nbins=16)
    _, med_balmer = median_alongr(R_mag, np.log10(balmer_ratio/HaHb_I), rmin=-16., rmax=-24., nbins=16)
    
    return [med_fnuv, med_balmer]


def median_alongr(rmag, values, rmin=-16., rmax=-24., nbins=16): 
    ''' find the median of specified values as a function of rmag  
    '''
    dr = (rmin - rmax)/float(nbins) 

    medians = [] 
    for i in range(nbins-1): 
        rbin = (rmag < rmin-dr*i) & (rmag >= rmin-dr*(i+1)) & np.isfinite(values) 
        medians.append(np.median(values[rbin])) 
    rmid = rmin - dr*(np.arange(nbins-1).astype(int)+0.5)

    return rmid, np.array(medians) 


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


def writeABC(type, pool, prior=None, abc_dir=None): 
    ''' Given abcpmc pool object. Writeout specified ABC pool property
    '''
    if abc_dir is None: 
        abc_dir = os.path.join(dat_dir, 'abc') 

    if type == 'init': # initialize
        if not os.path.exists(abc_dir): 
            try: 
                os.makedirs(abc_dir)
            except OSError: 
                pass 
        # write specific info of the run  
        f = open(os.path.join(abc_dir, 'info.md'), 'w')
        f.write('# '+run+' run specs \n')
        f.write('N_particles = %i \n' % pool.N)
        f.write('Distance function = %s \n' % pool.dist.__name__)
        # prior 
        f.write('Top Hat Priors \n')
        f.write('Prior Min = [%s] \n' % ','.join([str(prior_obj.min[i]) for i in range(len(prior_obj.min))]))
        f.write('Prior Max = [%s] \n' % ','.join([str(prior_obj.max[i]) for i in range(len(prior_obj.max))]))
        f.close()

    elif type == 'eps': # threshold writeout 
        if pool is None: # write or overwrite threshold writeout
            f = open(os.path.join(abc_dir, 'epsilon.dat'), "w")
        else: 
            f = open(os.path.join(abc_dir, 'epsilon.dat'), "a") # append
            f.write(str(pool.eps)+'\t'+str(pool.ratio)+'\n')
        f.close()
    elif type == 'theta': # particle thetas
        np.savetxt(os.path.join(abc_dir, 'theta.t%i.dat' % (pool.t)), pool.thetas) 
    elif type == 'w': # particle weights
        np.savetxt(os.path.join(abc_dir, 'w.t%i.dat' % (pool.t)), pool.ws)
    elif type == 'rho': # distance
        np.savetxt(os.path.join(abc_dir, 'rho.t%i.dat' % (pool.t)), pool.dists)
    else: 
        raise ValueError
    return None 
