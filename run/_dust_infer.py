'''
'''
import os 
import h5py 
import numpy as np 
import corner as DFM 
# -- galpopfm --
from galpopfm import dustfm as dustFM
from galpopfm import dust_infer as dustInfer
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


def explore_distances(name):
    '''
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
    
    # magnitudes
    F_mag_dust = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_fuv') 
    N_mag_dust = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='galex_nuv') 
    R_mag_dust = measureObs.AbsMag_sed(sed['wave'], sed_dusty, band='r_sdss') 
    FUV_NUV_dust = F_mag_dust - N_mag_dust 
    #a_fuv_dust = measureObs.A_FUV(F_mag, N_mag, R_mag) 

    wlim = (sed['wave'] > 4e3) & (sed['wave'] < 7e3) 

    # balmer measurements 
    Ha_dust, Hb_dust = measureObs.L_em(['halpha', 'hbeta'], sed['wave'][wlim], sed_dusty[:,wlim]) 
    HaHb_dust = Ha_dust/Hb_dust

    # read in SDSS measurements 
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    F_mag_sdss = sdss['ABSMAG'][...][:,0]
    N_mag_sdss = sdss['ABSMAG'][...][:,1]
    R_mag_sdss = sdss['ABSMAG'][...][:,4]
    FUV_NUV_sdss = F_mag_sdss - N_mag_sdss
    #a_fuv_sdss = measureObs.A_FUV(F_mag_sdss, N_mag_sdss, R_mag_sdss) 
    #print(a_fuv)
    #print(a_fuv_sdss)

    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]
    Ha_sdss = Haflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    Hb_sdss = Hbflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    HaHb_sdss = Ha_sdss/Hb_sdss 

    HaHb_I = 2.86 # intrinsic balmer ratio 

    fig = plt.figure(figsize=(11,5))

    # Mr - Balmer ratio
    sub = fig.add_subplot(121)

    DFM.hist2d(R_mag_sdss, np.log10(HaHb_sdss/HaHb_I), color='k', 
            levels=[0.68, 0.95], range=[[-15, -24], [-0.1, 0.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_dust, np.log10(HaHb_dust/HaHb_I), color='C1', 
            levels=[0.68, 0.95], range=[[-15, -24], [-0.1, 0.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 
    #subscatter(R_mag_sdss, np.log10(HaHb_sdss/HaHb_I), c='k', s=0.1, label='SDSS')
    #sub.scatter(R_mag_dust, np.log10(HaHb_dust/HaHb_I), c='C1', s=0.05, label='SIMBA dust')
    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss, np.log10(HaHb_sdss/HaHb_I), rmin=-16, rmax=-24, nbins=16)
    rmid_dust, med_dust = dustInfer.median_alongr(R_mag_dust, np.log10(HaHb_dust/HaHb_I), rmin=-16, rmax=-24, nbins=16)
    print(rmid_sdss, med_sdss) 
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x', label='SDSS')
    sub.scatter(rmid_dust, med_dust, c='C1', s=30, marker='x', label='SIMBA dust')
    sub.legend(loc='upper left', fontsize=15, handletextpad=0.2) 
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-15, -24) 
    sub.set_ylabel(r'$\log (H_\alpha/H_\beta)/(H_\alpha/H_\beta)_I$', fontsize=20) 
    sub.set_ylim(-0.1, 0.5) 
    
    # Mr - A_FUV
    sub = fig.add_subplot(122)
    DFM.hist2d(R_mag_sdss, FUV_NUV_sdss, color='k', 
            levels=[0.68, 0.95], range=[[-15, -24], [-0.5, 2.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 
    DFM.hist2d(R_mag_dust, FUV_NUV_dust, color='C1', 
            levels=[0.68, 0.95], range=[[-15, -24], [-0.5, 2.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            ax=sub) 
    rmid_sdss, med_sdss = dustInfer.median_alongr(R_mag_sdss, FUV_NUV_sdss, rmin=-16, rmax=-24, nbins=16)
    rmid_dust, med_dust = dustInfer.median_alongr(R_mag_dust, FUV_NUV_dust, rmin=-16, rmax=-24, nbins=16)
    sub.scatter(rmid_sdss, med_sdss, c='k', s=30, marker='x')
    sub.scatter(rmid_dust, med_dust, c='C1', s=30, marker='x')
    sub.set_xlabel(r'$M_r$', fontsize=20) 
    sub.set_xlim(-16, -24) 
    sub.set_ylabel(r'$FUV - NUV$', fontsize=20) 
    sub.set_ylim(-0.5, 2.5) 
    ffig = fhdf5.replace('.hdf5', '.explore_distance.png')
    fig.savefig(ffig, bbox_inches='tight') 

    # -- kNN divergence ---
    #from skl_groups.features import Features
    #from skl_groups.divergences import KNNDivergenceEstimator

    #X_sdss = np.vstack([R_mag_sdss, FUV_NUV_sdss, np.log10(HaHb_sdss/HaHb_I)]).T
    #X_dust = np.vstack([R_mag_dust, FUV_NUV_dust, np.log10(HaHb_dust/HaHb_I)]).T
    #        
    #kNN = KNNDivergenceEstimator(div_funcs=['kl'], Ks=5, version='slow', clamp=False, n_jobs=1)
    #feat = Features([X_sdss, X_dust])
    #div_knn = kNN.fit_transform(feat)
    #print(div_knn[0][0][0][1]) 
    return None 


def test_distance(name): 
    # read in observations 
    dat_dir = os.environ['GALPOPFM_DIR']
    fsdss = os.path.join(dat_dir, 'obs', 'tinker_SDSS_centrals_M9.7.valueadd.hdf5') 
    sdss = h5py.File(fsdss, 'r') 
    
    F_mag_sdss = sdss['ABSMAG'][...][:,0]
    N_mag_sdss = sdss['ABSMAG'][...][:,1]
    R_mag_sdss = sdss['ABSMAG'][...][:,4]

    Haflux_sdss = sdss['HAFLUX'][...]
    Hbflux_sdss = sdss['HBFLUX'][...]
    #Ha_sdss = Haflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17
    #Hb_sdss = Hbflux_sdss * (4.*np.pi * (sdss['Z'][...] * 2.9979e10/2.2685e-18)**2) * 1e-17

    x_obs = dustInfer.sumstat_obs(F_mag_sdss, N_mag_sdss, R_mag_sdss, Haflux_sdss, Hbflux_sdss, sdss['Z'][...])

    # read SED for sims 
    sim_sed = dustInfer._read_sed(name) 

    # pass through the minimal amount of memory 
    wlim = (sim_sed['wave'] > 1e3) & (sim_sed['wave'] < 1e4) 
    sim_sed_wlim = {}
    sim_sed_wlim['mstar']   = sim_sed['mstar'] 
    sim_sed_wlim['wave']    = sim_sed['wave'][wlim] 
    for k in ['sed_noneb', 'sed_onlyneb']: 
        sim_sed_wlim[k] = sim_sed[k][:,wlim]
    
    import time 
    for i in range(10): 
        t0 = time.time()
        theta = np.array([0.1, 0.2+0.05*float(i), 1./0.44]) 
        print(dustInfer.distance_metric(theta, sim_sed_wlim, x_obs, dem='slab_calzetti')) 
        print('distance takes %.f' % (time.time() - t0))
    return None 


if __name__=="__main__": 
    #explore_distances('simba')
    test_distance('simba')
