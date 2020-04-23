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


def distance_metric(x_obs, x_model, method='chi2', x_err=None): 
    ''' distance metric between forward model m(theta) and observations

    notes
    -----
    * simple L2 norm between the 3D histogram of [Rmag, Balmer, FUV-NUV]
    ''' 
    if x_err is None: 
        x_err = [1. for _x in x_obs]

    if method == 'chi2': # chi-squared
        rho = [np.sum((_obs - _mod)**2/_err**2) 
                for _obs, _mod, _err in zip(x_obs, x_model, x_err)]
    elif method == 'L2': # chi-squared
        rho = [np.sum((_obs - _mod)**2) 
                for _obs, _mod, _err in zip(x_obs, x_model, x_err)]
    elif method == 'L1': # L1 morm 
        rho = [np.sum(np.abs(_obs - _mod))
                for _obs, _mod, _err in zip(x_obs, x_model, x_err)]
    else: 
        raise NotImplementedError
    print('     (%s)' % ', '.join([str(_rho) for _rho in rho]))
    return rho


def sumstat_obs(name='sdss', statistic='2d', return_bins=False): 
    ''' summary statistics for SDSS observations is the 3D histgram of 
    [M_r, G-R, FUV - NUV]. 

    notes
    -----
    * see `nb/observables.ipynb` to see exactly how the summary statistic is
    calculated. 
    '''
    if statistic == '1d': 
        r_edges, gr_edges, fn_edges, x_gr, x_fn, _, _ = np.load(os.path.join(dat_dir, 'obs',
            'tinker_SDSS_centrals_M9.7.Mr_complete.Mr.GR.FUVNUV.npy'), 
            allow_pickle=True)
        dgr = gr_edges[1] - gr_edges[0]
        nbar = dgr * np.sum(x_gr)
        x_obs = [nbar, x_gr, x_fn]

    elif statistic == '2d': 
        r_edges, gr_edges, fn_edges, x_gr, x_fn, _, _ = np.load(os.path.join(dat_dir, 'obs',
            'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_GR.Mr_FUVNUV.npy'), 
            allow_pickle=True) 
        dr = r_edges[1] - r_edges[0]
        dgr = gr_edges[1] - gr_edges[0]
        nbar = dr * dgr * np.sum(x_gr),
        x_obs = [nbar, x_gr, x_fn]

    elif statistic == '3d': 
        r_edges, gr_edges, fn_edges, _x_obs, _ = np.load(os.path.join(dat_dir, 'obs',
            'tinker_SDSS_centrals_M9.7.Mr_complete.Mr_GR_FUVNUV.npy'), 
            allow_pickle=True)
        dr = r_edges[1] - r_edges[0]
        dgr = gr_edges[1] - gr_edges[0]
        dfn = fn_edges[1] - fn_edges[0]
        nbar = dr * dgr * dfn * np.sum(_x_obs)
        x_obs = [nbar, _x_obs]
    
    if return_bins: 
        return r_edges, gr_edges, fn_edges, x_obs

    return x_obs 


def sumstat_model(theta, sed=None, dem='slab_calzetti', f_downsample=1.,
        statistic='2d', return_datavector=False, extra_data=None): 
    ''' calculate summary statistics for forward model m(theta) 
    
    :param theta: 
        array of input parameters
    :param sed: 
        dictionary with SEDs of **central** galaxies  
    :param dem: 
        string specifying the dust empirical model
    :param f_downsample: 
        if f_downsample > 1., then the SED dictionary is downsampled. 
    :param extra_data: 
        fixed extra data to be included in the data_vector. Has to be
        of the form [Rmag, G-R, FUV-NUV] (see note below) 

    notes
    -----
    * still need to implement noise model
    * 4/22/2020: extra_data kwarg added. This is to pass pre-sampled
    observables for SFR = 0 galaxies 
    '''
    # don't touch these values! they are set to agree with the binning of
    # obersvable
    nbins = [8, 400, 200]
    ranges = [(20, 24), (-5., 20.), (-5, 45.)]
    dRmag   = 0.5
    dGR     = 0.0625
    dfuvnuv = 0.25

    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            sed['logmstar'],
            sed['logsfr.inst'],
            dem=dem) 
    
    # observational measurements 
    F_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_fuv') 
    N_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_nuv') 
    G_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='g_sdss') 
    R_mag = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 

    FUV_NUV = F_mag - N_mag 
    G_R = G_mag - R_mag

    data_vector = np.array([-1.*R_mag, G_R, FUV_NUV]).T

    if extra_data is not None: #append extra data to data vector
        # assumes extra_data = [R_mag, G-R, FUV-NUV]
        extra_data[0] = -1. * extra_data[0]
        data_vector = np.concatenate([data_vector, np.array(extra_data).T], axis=0) 

    if return_datavector: 
        return data_vector.T

    Nbins, _ = np.histogramdd(data_vector, bins=nbins, range=ranges)
    
    # volume of simulation 
    vol = {'simba': 100.**3, 'tng': 75.**3}[sed['sim']]  

    x_model = Nbins.astype(float) / vol / dRmag / dGR / dfuvnuv / f_downsample
    nbar = dRmag * dGR * dfuvnuv * np.sum(x_model)
    
    if statistic == '3d': 
        return [nbar, x_model]
    elif statistic == '2d': 
        x_r_gr = dfuvnuv * np.sum(x_model, axis=2)
        x_r_fn = dGR * np.sum(x_model, axis=1)
        return [nbar, x_r_gr, x_r_fn]
    elif statistic == '1d': 
        x_gr = dRmag * np.sum(dfuvnuv * np.sum(x_model, axis=2), axis=0)
        x_fn = dRmag * np.sum(dGR * np.sum(x_model, axis=1), axis=0) 
        return [nbar, x_gr, x_fn]


def _observable_zeroSFR(wave, sed): 
    ''' for SFR = 0 galaxies, sample G-R and FUV-NUV color directly from G-R
    and FUV-NUV distributions of quiescent SDSS galaxies. This is to remove
    these galaxies from consideration in the inference. 

    notes
    -----
    * in principle, the G-R and FUV-NUV sampling can done for R bins, but at
    the moment it does not. 
    * this only runs once so its not optimized in any way 
    '''
    ngal = sed.shape[0]  
    # read in G-R and FUV-NUV distributions of SDSS quiescent galaxies 
    gr_edges, gr_nbins = np.load(os.path.join(dat_dir, 'obs',
        'tinker_SDSS_centrals_M9.7.Mr_complete.quiescent.G_R_dist.npy'), 
        allow_pickle=True)

    fn_edges, fn_nbins = np.load(os.path.join(dat_dir, 'obs',
        'tinker_SDSS_centrals_M9.7.Mr_complete.quiescent.FUV_NUV_dist.npy'), 
        allow_pickle=True)
    
    # calculate Mr from SEDs 
    R_mag = measureObs.AbsMag_sed(wave, sed, band='r_sdss') 
    
    # now sample from SDSS distribution using inverse transform sampling  
    gr_cdf = np.cumsum(gr_nbins)/np.sum(gr_nbins) # calculate CDFs for both distributions
    fn_cdf = np.cumsum(fn_nbins)/np.sum(fn_nbins) 

    us      = np.random.rand(ngal) 
    G_R     = np.empty(ngal) 
    FUV_NUV = np.empty(ngal)
    for i, u in enumerate(us): 
        G_R[i] = 0.5*(gr_edges[:-1] + gr_edges[1:])[np.abs(u - gr_cdf).argmin()]
        FUV_NUV[i] = 0.5*(fn_edges[:-1] + fn_edges[1:])[np.abs(u - fn_cdf).argmin()]
    
    return [R_mag, G_R, FUV_NUV]


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
    sed['logsfr.inst']  = f['logsfr.inst'][...]
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
    isnan = (~np.isfinite(sed['logsfr.inst']))
    sed['logsfr.inst'][isnan] = -999.
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
