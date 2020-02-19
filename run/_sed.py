'''

script for preprocessing Tjitske's SED files 


'''
import os 
import h5py 
import numpy as np 

dat_dir = os.environ['GALPOPFM_DIR']


def compile_seds(name): 
    ''' compile the seds and their corresponding meta-data and save to hdf5 files in a consistent way
    
    notes
    -----
    * central/satellite: 1 = central; 0 = satellite 
    '''
    nfile = 24 

    fneb = lambda i: os.path.join(dat_dir, 'sed', name, '%i_%s_Fullspectra_Nebular_onlyAGBdust.txt' % (i, name.upper())) 
    fnoneb = lambda i: os.path.join(dat_dir, 'sed', name, '%i_%s_Fullspectra_noNebular_onlyAGBdust.txt' % (i, name.upper())) 
    fhdf5 = os.path.join(dat_dir, 'sed', '%s.hdf5' % name) 
    
    # add meta data 
    data = {} 
    if name == 'simba': 
        fprop = os.path.join(dat_dir, 'prop', 'SIMBA-mstar-posvel-cen.txt') # ID, Mstar[Msun], central(True)/satellite(False), pos x 3 (kpc), vel x 3 (km/s)
        _id, mstar, censat, pos0, pos1, pos2, vel0, vel1, vel2 = np.loadtxt(fprop, skiprows=1, unpack=True, usecols=range(9))

        ngal = len(_id) 
        data['id']          = _id.astype(int)
        data['logmstar']    = np.log10(mstar) 
        data['censat']      = censat.astype(int)
        data['pos']         = np.vstack([pos0, pos1, pos2]).T
        data['vel']         = np.vstack([vel0, vel1, vel2]).T
        assert data['pos'].shape[0] == ngal 
        assert data['vel'].shape[0] == ngal 
    elif name == 'tng': 
        fprop0 = os.path.join(dat_dir, 'prop', 'IQSFSdata_TNG_99.txt') # SubhaloID, log10(total stellar mass)[Msun], log10(total gas mass)[Msun], log10(total SFR)[Msun/yr], log10(SFR over100Myr)[Msun/yr], satellite? (1:yes, 0: central)
        _id, logmstar, logmgas, logsfrtot, logsfr100, censat = np.loadtxt(fprop0, skiprows=1, unpack=True, usecols=range(6))
        fprop1 = os.path.join(dat_dir, 'prop', 'IQSFSdataTNG-MstarPosVelSat.txt') # SubhaloID, Mstar[Msun], Pos0[kpc], Pos1[kpc], Pos2[kpc], Vel0[km/s], Vel1[km/s], Vel2[km/s], icentral(0)/satellite(1)
        pos0, pos1, pos2, vel0, vel1, vel2 = np.loadtxt(fprop1, skiprows=1, unpack=True, usecols=[2,3,4,5,6,7])

        ngal = len(_id) 
        data['id']          = _id.astype(int)
        data['logmstar']    = logmstar
        data['logsfr.tot']  = logsfrtot
        data['logsfr.100']  = logsfr100
        data['censat']      = (np.abs(censat - 1)).astype(int)
        data['pos']         = np.vstack([pos0, pos1, pos2]).T
        data['vel']         = np.vstack([vel0, vel1, vel2]).T
        assert data['pos'].shape[0] == ngal 
        assert data['vel'].shape[0] == ngal 
    elif name == 'eagle':
        fprop0 = os.path.join(dat_dir, 'prop', 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt') # GroupNr, SubGroupNr, log10(StellarMass)[Msun], SFR10Myr[Msun/yr], SFR1Gyr[Msun/yr], Central_SUBFIND
        logmstar, logsfr10, logsfr1gyr, censat = np.loadtxt(fprop0, skiprows=1, unpack=True, usecols=[2,3,4,5])
        fprop1 = os.path.join(dat_dir, 'prop', 'EAGLE_RefL0100_PosVelMstar_allabove1.8e8Msun.txt') # pos[Mpc], vel[km/s], Mstar[Msun]
        pos0, pos1, pos2, vel0, vel1, vel2 = np.loadtxt(fprop1, skiprows=1, unpack=True, usecols=range(6))

        ngal = len(logmstar) 
        data['logmstar']    = logmstar
        data['logsfr.10']   = logsfr10
        data['logsfr.1g']   = logsfr1gyr
        data['censat']      = censat.astype(int)
        data['pos']         = np.vstack([pos0, pos1, pos2]).T
        data['vel']         = np.vstack([vel0, vel1, vel2]).T
        assert data['pos'].shape[0] == ngal 
        assert data['vel'].shape[0] == ngal 
    else: 
        raise NotImplementedError

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
    #compile_seds('simba')  
    #compile_seds('tng') 
    compile_seds('eagle') 
