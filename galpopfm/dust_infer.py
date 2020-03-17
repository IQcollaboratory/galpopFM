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


def dust_abc(name, T, eps0=[0.1, 1.], N_p=100, prior_range=None, dem='slab_calzetti', abc_dir=None, mpi=False, nthread=1):
    ''' run ABC-PMC to infer posteriors for dust empirical model parameters 
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
    # only keep centrals
    cens = sim_sed['censat'].astype(bool) 
    
    # save as global variable that can be accessed by multiprocess 
    #global shared_sim_sed
    shared_sim_sed = {} 
    shared_sim_sed['logmstar']      = sim_sed['logmstar'][cens].copy()
    shared_sim_sed['wave']          = sim_sed['wave'][wlim].copy()
    shared_sim_sed['sed_noneb']     = sim_sed['sed_noneb'][cens,:][:,wlim].copy() 
    shared_sim_sed['sed_onlyneb']   = sim_sed['sed_onlyneb'][cens,:][:,wlim].copy() 

    #--- inference with ABC-PMC below ---
    # prior 
    prior_min = prior_range[0] 
    prior_max = prior_range[1] 
    prior = abcpmc.TophatPrior(prior_min, prior_max) 
    
    # sampler 
    if mpi: 
        mpi_pool = mpi_util.MpiPool()
        
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,                  # N_particles
                Y=x_obs,                # data
                postfn=_sumstat_model_shared,   # simulator 
                dist=distance_metric,   # distance metric 
                pool=mpi_pool, 
                postfn_kwargs={'dem': dem, 'shared_sim_sed': shared_sim_sed},
                dist_kwargs={'method': 'L2'}
                )      
    else: 
        abcpmc_sampler = abcpmc.Sampler(
                N=N_p,                  # N_particles
                Y=x_obs,                # data
                postfn=_sumstat_model_shared,   # simulator 
                dist=distance_metric,   # distance metric 
                threads=nthread,
                postfn_kwargs={'dem': dem, 'shared_sim_sed': shared_sim_sed},
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
        # plot ABC particles 
        plotABC(pool, prior=prior, dem=dem, abc_dir=abc_dir)

        # update epsilon based on median thresholding 
        eps.eps = np.median(pool.dists, axis=0)
        pools.append(pool)
        print('eps%i' % pool.t, eps.eps)
        print('----------------------------------------')
        if pool.ratio <0.2:
            break
    abcpmc_sampler.close()
    return None 


def distance_metric(x_obs, x_model, method='L2', x_err=None): 
    ''' distance metric between forward model m(theta) and observations

    notes
    -----
    * simple L2 norm between the 3D histogram of [Rmag, Balmer, FUV-NUV]
    ''' 
    if method == 'L2': # L2 norm  
        if x_err is None: 
            x_err = 1. 

        rho = np.sum((x_obs - x_model)**2/x_err**2)

        print('     (%.5f)' % rho)
        return rho 
    else: 
        raise NotImplemented 


def sumstat_obs(name='sdss'): 
    ''' summary statistics for SDSS observations is the 3D histgram of 
    [M_r, log10( (HA/HB)/(HA/HB)_I ), FUV - NUV]. 

    notes
    -----
    * see `nb/observables.ipynb` to see exactly how the summary statistic is
    calculated. 
    '''
    _, _, _, x_obs, _ = np.load(os.path.join(dat_dir, 'obs',
            'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_Balmer_FUVNUV.npy'))
    return x_obs 


def sumstat_model(theta, sed=None, dem='slab_calzetti', f_downsample=1.): 
    ''' calculate summary statistics for forward model m(theta) 

    :param sed: 
        dictionary with SEDs of **central** galaxies  

    notes
    -----
    * still need to implement noise model
    '''
    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            sed['logmstar'],
            sed['logsfr.100'],
            dem=dem) 
    
    # observational measurements 
    F_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_fuv') 
    N_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_nuv') 
    R_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 
    FUV_NUV = F_mag - N_mag 

    # balmer measurements 
    Ha_dust, Hb_dust = measureObs.L_em(['halpha', 'hbeta'], sed['wave'], sed_dusty) 
    balmer_ratio = Ha_dust/Hb_dust

    HaHb_I = 2.86 # intrinsic balmer ratio 
    
    # don't touch these values!
    nbins = [8, 10, 10]
    ranges = [(20, 24), (-1, 1.), (-1, 4.)]
    dRmag   = 0.5 
    dbalmer = 0.2
    dfuvnuv = 0.5

    data_vector = np.array([-1.*R_mag, np.log10(balmer_ratio/HaHb_I), FUV_NUV]).T
    Nbins, _ = np.histogramdd(data_vector, bins=nbins, range=ranges)
    
    # volume of simulation 
    vol = {'simba': 100.**3, 'tng': 75.**3}[sed['sim']]  

    rho = Nbins.astype(float) / vol / dRmag / dbalmer / dfuvnuv
    return rho


def median_alongr(rmag, values, rmin=-20., rmax=-24., nbins=16): 
    ''' find the median of specified values as a function of rmag  
    '''
    dr = (rmin - rmax)/float(nbins) 

    medians = [] 
    for i in range(nbins-1): 
        rbin = (rmag < rmin-dr*i) & (rmag >= rmin-dr*(i+1)) & np.isfinite(values) 
        medians.append(np.median(values[rbin])) 
    rmid = rmin - dr*(np.arange(nbins-1).astype(int)+0.5)

    return rmid, np.array(medians) 


def _read_sed(name, seed=0): 
    ''' read in sed files 
    '''
    if name not in ['simba', 'tng']: raise NotImplementedError
    fhdf5 = os.path.join(dat_dir, 'sed', '%s.hdf5' % name) 

    f = h5py.File(fhdf5, 'r') 
    sed = {} 
    sed['wave']         = f['wave'][...] 
    sed['sed_neb']      = f['sed_neb'][...]
    sed['sed_noneb']    = f['sed_noneb'][...]
    sed['sed_onlyneb']  = sed['sed_neb'] - sed['sed_noneb'] # only nebular emissoins 
    sed['logmstar']     = f['logmstar'][...] 
    sed['logsfr.100']   = f['logsfr.100'][...] 
    sed['censat']       = f['censat'][...] 
    f.close() 
    
    '''
    # deal with SFR resolution effect by unifromly sampling the SFR 
    # over 0 to resolution limit 
    if name == 'simba': 
        res_sfr = 0.182
    elif name == 'tng': 
        res_sfr = 0.005142070183729021 # THIS IS WRONG!!!
    
    np.random.seed(seed)
    isnan = (~np.isfinite(sed['logsfr.100']))
    sed['logsfr.100'][isnan] = np.log10(np.random.uniform(0., res_sfr, size=np.sum(isnan))) 
    '''
    isnan = (~np.isfinite(sed['logsfr.100']))
    sed['logsfr.100'][isnan] = -999.
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


def plotABC(pool, prior=None, dem='slab_calzetti', abc_dir=None): 
    ''' Given abcpmc pool object plot the particles 
    '''
    import corner as DFM 
    import matplotlib as mpl
    import matplotlib.pyplot as plt 
    try: 
        # sometimes this formatting fails 
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['axes.xmargin'] = 1
        mpl.rcParams['xtick.labelsize'] = 'x-large'
        mpl.rcParams['xtick.major.size'] = 5
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.labelsize'] = 'x-large'
        mpl.rcParams['ytick.major.size'] = 5
        mpl.rcParams['ytick.major.width'] = 1.5
        mpl.rcParams['legend.frameon'] = False
    except: 
        pass 

    # prior range
    prior_range = [(_min, _max) for _min, _max in zip(prior.min, prior.max)]

    # theta labels 
    if dem == 'slab_calzetti': 
        lbls = [r'$m_{\tau}$', r'$c_{\tau}$', r'$f_{\rm neb}$'] 
    elif dem == 'slab_noll_simple': 
        lbls = [r'$c_{\tau}$', r'$c_{\delta}$'] 
    elif dem == 'slab_noll_m': 
        lbls = [r'$m_{\tau}$', r'$c_{\tau}$', r'$m_\delta$', r'$c_\delta$',
                r'$m_E$', r'$c_E$', r'$f_{\rm neb}$'] 
    elif dem == 'slab_noll_msfr': 
        lbls = [r'$m_{\tau,1}$', r'$m_{\tau,2}$', r'$c_{\tau}$', 
                r'$m_{\delta,1}$', r'$m_{\delta,2}$', r'$c_\delta$',
                r'$m_E$', r'$c_E$', r'$f_{\rm neb}$'] 
    else: 
        raise NotImplementedError

    if abc_dir is None: 
        abc_dir = os.path.join(dat_dir, 'abc') 
        
    fig = DFM.corner(
            pool.thetas, 
            range=prior_range,
            weights=pool.ws,
            quantiles=[0.16, 0.5, 0.84], 
            levels=[0.68, 0.95],
            nbin=20, 
            smooth=True, 
            labels=lbls, 
            label_kwargs={'fontsize': 20}) 
    fig.savefig(os.path.join(abc_dir, 'abc.t%i.png' % pool.t) , bbox_inches='tight') 
    return None 
