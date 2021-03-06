'''

script to check galpopfm.measure_obs

'''
import os 
import h5py 
import numpy as np 
# -- galpopfm --
from galpopfm import dustfm as dustFM
from galpopfm import measure_obs as measureObs
# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt 
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


def fm_balmer(name): 
    ''' check balmer measurements from the forward modeled SEDs and compare
    with SDSS balmer measurements 

    '''
    dat_dir = os.environ['GALPOPFM_DIR']
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

    theta = np.array([0.1, 0.2, 1./0.44]) 
    sed_dusty = dustFM.Attenuate(
            theta, 
            sed['wave'], 
            sed['sed_noneb'], 
            sed['sed_onlyneb'], 
            np.log10(sed['mstar']),
            dem='slab_calzetti') 

    wlim = (sed['wave'] > 4e3) & (sed['wave'] < 7e3) 

    # balmer measurements 
    import time 
    t0 = time.time() 
    Ha, Hb = measureObs.L_em(['halpha', 'hbeta'], sed['wave'][wlim], sed['sed_neb'][:,wlim]) 
    print('Ha, Hb measurement takes %s sec' % (time.time() - t0)) 
    Ha_dust, Hb_dust = measureObs.L_em(['halpha', 'hbeta'], sed['wave'][wlim], sed_dusty[:,wlim]) 
    HaHb = Ha/Hb # used to measure balmer decrement 
    HaHb_dust = Ha_dust/Hb_dust

    # read in SDSS measurements 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]
    Ha_sdss = Haflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    Hb_sdss = Hbflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    HaHb_sdss = Ha_sdss/Hb_sdss 

    HaHb_I = 2.86 # intrinsic balmer ratio 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    sub.scatter(np.log10(HaHb_sdss/HaHb_I), np.log10(Ha_sdss), c='k', s=0.1, label='SDSS')
    sub.scatter(np.log10(HaHb/HaHb_I), np.log10(Ha), c='C0', s=0.1, label='SIMBA w/o dust')
    sub.scatter(np.log10(HaHb_dust/HaHb_I), np.log10(Ha_dust), c='C1', s=0.1, label='SIMBA w/ dust')
    sub.legend(loc='lower right', fontsize=15, markerscale=20) 
    sub.set_xlabel(r'$\log (H_\alpha/H_\beta)/(H_\alpha/H_\beta)_I$', fontsize=20) 
    sub.set_xlim(-0.1, 0.5) 
    sub.set_ylabel(r'$\log H_\alpha$ luminosity', fontsize=20) 
    #sub.set_ylim(0., Ha_sdss[np.isfinite(Ha_sdss)].max())
    ffig = fhdf5.replace('.hdf5', '.ha_hb.png')
    fig.savefig(ffig, bbox_inches='tight') 
    return None 


def medfilter_tets(): 
    ''' speed test for medfilt w/ for loop versu medfilt2d 

    medfilt2d is an order of magnitude faster, but still not very fast... 
    '''
    from scipy.signal import medfilt
    from scipy.signal import medfilt2d

    a = np.linspace(0., 1, 1000000).reshape((1000,1000)) 

    import time 
    t0 = time.time() 
    for i in range(a.shape[0]): 
        medfilt(a[i], 151) 
    
    print('for loop medfilt1d %.2f' % (time.time() - t0))
    t0 = time.time() 
    medfilt2d(a, [1,151]) 
    print('medfilt2d %.2f' % (time.time() - t0)) 
    
    from scipy.interpolate import interp1d 
    t0 = time.time() 
    cont_sparse = medfilt2d(a[:,::5], [1, 31]) 
    cont_interp = interp1d(np.arange(a.shape[1])[::5], cont_sparse, fill_value='extrapolate')
    cont_interp(np.arange(a.shape[1]))
    print('sparse medfilt2d %.2f' % (time.time() - t0)) 
    
    from scipy.ndimage.filters import median_filter
    t0 = time.time() 
    median_filter(a, (1, 151))
    print('median filter %.2f' % (time.time() - t0)) 
    return None 


def get_spec_em_test(name): 
    dat_dir = os.environ['GALPOPFM_DIR']
    if name == 'simba': 
        fhdf5 = os.path.join(dat_dir, 'sed', 'simba.hdf5') 
    else: 
        raise NotImplementedError

    f = h5py.File(fhdf5, 'r') 
    sed = {} 
    sed['wave']         = f['wave'][...]
    sed['sed_neb']      = f['sed_neb'][...][:1000,:]
    f.close() 
    print(sed['wave'].min(), sed['wave'].max())
    wlim = (sed['wave'] > 6554.6) & (sed['wave'] < 6574.6) # Yan et al.
    wlim0 = (sed['wave'] > 6000.) & (sed['wave'] < 7000.) # Yan et al.

    from scipy.signal import medfilt2d 
    from scipy.interpolate import interp1d 
    import time 
    t0 = time.time()
    spec_medfilt2d = medfilt2d(sed['sed_neb'], [1, 151])
    print('medfilt2d %.2f' % (time.time() - t0)) 
    
    t0 = time.time()
    cont_sparse = medfilt2d(sed['sed_neb'][:,::5], [1, 31]) 
    cont_interp = interp1d(np.arange(sed['sed_neb'].shape[1])[::5], cont_sparse, fill_value='extrapolate')
    spec_sparsefilt = cont_interp(np.arange(sed['sed_neb'].shape[1]))
    print('sparse medfilt2d %.2f' % (time.time() - t0)) 
    
    t0 = time.time()
    cont_sparse = medfilt2d(sed['sed_neb'][:,::10], [1, 15]) 
    cont_interp = interp1d(np.arange(sed['sed_neb'].shape[1])[::10], cont_sparse, fill_value='extrapolate')
    spec_sparsefilt1 = cont_interp(np.arange(sed['sed_neb'].shape[1]))
    print('sparse medfilt2d 15 %.2f' % (time.time() - t0)) 

    fig = plt.figure(figsize=(10,10))
    for i in range(2):
        sub = fig.add_subplot(2,1,i+1)  
        sub.plot(sed['wave'], sed['sed_neb'][i,:], c='C0')
        sub.plot(sed['wave'][wlim], spec_medfilt2d[i,wlim], c='k', ls='--') 
        sub.plot(sed['wave'], spec_sparsefilt[i,:], c='C3', ls=':') 
        sub.plot(sed['wave'], spec_sparsefilt1[i,:], c='C4', ls=':') 
    #sub.plot(sed['wave'], measureObs.get_spec_em(np.atleast_2d(sed['sed_neb'][0,:]))[0]) 
        sub.set_xlim(6500.6, 6600.6) 
    ffig = os.path.join(dat_dir, 'get_spec_em_test.png') 
    fig.savefig(ffig, bbox_inches='tight') 
    return None  
    

if __name__=="__main__": 
    #fm_balmer('simba')
    #medfilter_tets()
    get_spec_em_test('simba')
