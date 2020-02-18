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


dat_dir = os.environ['GALPOPFM_DIR']
abc_dir = os.path.join(dat_dir, 'abc', 'test') 


prior_min = np.array([0.0, 0.0, 2.]) 
prior_max = np.array([1., 1., 4.]) 

dustInfer.dust_abc(
        'simba', 
        3, 
        eps0=[0.5, 1.], 
        N_p=3, 
        prior_range=[prior_min, prior_max], 
        dem='slab_calzetti', 
        abc_dir=abc_dir)
