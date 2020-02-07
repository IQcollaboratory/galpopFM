'''

script for some preprocessing of SED files 


'''
import os 
import h5py 
import numpy as np 

dat_dir = os.environ['GALPOPFM_DIR']


def compile_seds(name): 
    ''' compile the seds and their corresponding meta-data and save to hdf5 files
    '''
    if name == 'simba': 
        nfile = 24 
        fprop = os.path.join(dat_dir, 'prop', 'SIMBA-mstar-posvel-cen.txt') # ID, Mstar[Msun], central(True)/satellite(False), pos x 3 (kpc), vel x 3 (km/s)
        fneb = lambda i: os.path.join(dat_dir, 'sed', 'simba', '%i_SIMBA_Fullspectra_Nebular_onlyAGBdust.txt' % i) 
        fnoneb = lambda i: os.path.join(dat_dir, 'sed', 'simba', '%i_SIMBA_Fullspectra_noNebular_onlyAGBdust.txt' % i) 
        fhdf5 = os.path.join(dat_dir, 'simba.hdf5') 

    else:
        raise NotImplementedError
    
    data = {} 
    # add meta data 
    _id, mstar, censat, pos0, pos1, pos2, vel0, vel1, vel2 = np.loadtxt(fprop, skiprows=1, unpack=True, usecols=range(9))
    ngal = len(_id) 
    data['id']      = _id.astype(int)
    data['mstar']   = mstar
    data['censat']  = censat    
    data['pos']     = np.vstack([pos0, pos1, pos2]).T
    assert data['pos'].shape[0] == ngal 
    data['vel']     = np.vstack([vel0, vel1, vel2]).T
    assert data['vel'].shape[0] == ngal 

    # read in FSPS wavelength file 
    data['wave'] = np.loadtxt(os.path.join(dat_dir, 'sed', 'FSPS_wave_Full.txt'), unpack=True, usecols=[0]) 
    nwave = len(data['wave'])

    sed_neb, sed_noneb = [], [] 
    for ifile in range(nfile): 
        sed_neb_i   = np.loadtxt(fneb(ifile)) 
        sed_noneb_i = np.loadtxt(fnoneb(ifile)) 

        assert sed_neb_i.shape[1] == nwave 
        assert sed_noneb_i.shape[1] == nwave 

        sed_neb.append(sed_neb_i) 
        sed_noneb.append(sed_noneb_i) 

    data['sed_neb'] = np.concatenate(sed_neb, axis=0) 
    data['sed_noneb'] = np.concatenate(sed_noneb, axis=0) 
    assert data['sed_noneb'].shape[0] == ngal 
    
    f = h5py.File(fhdf5, 'w') 
    for k in data.keys(): 
        f.create_dataset(k, data=data[k]) 
    f.close() 
    return None 


if __name__=="__main__": 
    compile_seds('simba')  
