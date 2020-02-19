'''
'''
import os 
import sys 
import numpy as np 
# -- galpopfm --
from galpopfm import dust_infer as dustInfer


dat_dir = os.environ['GALPOPFM_DIR']
abc_dir = os.path.join(dat_dir, 'abc', 'test') 


def run_test(T, Np): 
    ''' simplest ABC setup 
    '''
    prior_min = np.array([0.0, 0.0, 2.]) 
    prior_max = np.array([1., 1., 4.]) 

    dustInfer.dust_abc(
            'simba', 
            T, 
            eps0=[0.5, 1.], 
            N_p=Np, 
            prior_range=[prior_min, prior_max], 
            dem='slab_calzetti', 
            abc_dir=abc_dir, 
            mpi=True
            )
    return None 


if __name__=="__main__": 

    name = sys.argv[1] # name of ABC run

    niter = int(sys.argv[2]) # number of iterations
    npart = int(sys.argv[3]) # number of particles 

    if name == 'test': 
        print('Runnin test ABC with ...') 
        print('%i iterations' % niter)
        print('%i particles' % npart)

        run_test(niter, npar) 
    else: 
        raise NotImplementedError
