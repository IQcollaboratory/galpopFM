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
        fprop = os.path.join(dat_dir, 'prop', 'SIMBAz0.txt') #  Mstar[Msun], SFR_inst[Msun/yr], SFR_100Myr[Msun/yr], cen(1)/sat(0)
        mstar, sfr_inst, sfr_100, censat = np.loadtxt(fprop, skiprows=1,
                unpack=True, usecols=range(4)) 

        ngal = len(mstar) 
        data['logmstar']    = np.log10(mstar) 
        data['logsfr.inst'] = np.log10(sfr_inst) 
        data['logsfr.100']  = np.log10(sfr_100)
        data['censat']      = censat.astype(int)

    elif name == 'tng': 
        # SubhaloID, log10(total inst. stellar mass)[Msun], log10(total gas
        # mass)[Msun], log10(total SFR)[Msun/yr], log10(SFR
        # over100Myr)[Msun/yr], satellite? (1:yes, 0: central)
        fprop0 = os.path.join(dat_dir, 'prop', 'IQSFSdata_TNG_99-corrected.txt') 
        _id, logmstar, logmgas, logsfrinst, logsfr100, censat = np.loadtxt(fprop0, skiprows=1, unpack=True, usecols=range(6))
        #fprop1 = os.path.join(dat_dir, 'prop', 'IQSFSdataTNG-MstarPosVelSat.txt') # SubhaloID, Mstar[Msun], Pos0[kpc], Pos1[kpc], Pos2[kpc], Vel0[km/s], Vel1[km/s], Vel2[km/s], icentral(0)/satellite(1)
        #pos0, pos1, pos2, vel0, vel1, vel2 = np.loadtxt(fprop1, skiprows=1, unpack=True, usecols=[2,3,4,5,6,7])

        ngal = len(_id) 
        data['id']          = _id.astype(int)
        data['logmstar']    = logmstar
        data['logsfr.inst'] = logsfrinst
        data['logsfr.100']  = logsfr100
        data['censat']      = (np.abs(censat - 1)).astype(int)
    elif name == 'eagle':
        fprop0 = os.path.join(dat_dir, 'prop', 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt') # GroupNr, SubGroupNr, log10(StellarMass)[Msun], SFR10Myr[Msun/yr], SFR1Gyr[Msun/yr], Central_SUBFIND
        logmstar, logsfr10, logsfr1gyr, censat = np.loadtxt(fprop0, skiprows=1, unpack=True, usecols=[2,3,4,5])

        ngal = len(logmstar) 
        data['logmstar']    = logmstar
        data['logsfr.10']   = logsfr10
        data['logsfr.1g']   = logsfr1gyr
        data['censat']      = censat.astype(int)
    else: 
        raise NotImplementedError

    # read in FSPS wavelength file 
    data['wave'] = np.loadtxt(os.path.join(dat_dir, 'sed', 'FSPS_wave_Full.txt'), unpack=True, usecols=[0]) 
    nwave = len(data['wave'])

    sed_neb, sed_noneb = [], [] 
    for ifile in range(nfile): 
        print('%i of %i' % (ifile+1, nfile))
        sed_neb_i   = np.loadtxt(fneb(ifile)) 
        sed_noneb_i = np.loadtxt(fnoneb(ifile)) 

        assert sed_neb_i.shape[1] == nwave 
        assert sed_noneb_i.shape[1] == nwave 

        sed_neb.append(sed_neb_i) 
        sed_noneb.append(sed_noneb_i) 

    data['sed_neb'] = np.concatenate(sed_neb, axis=0) 
    data['sed_noneb'] = np.concatenate(sed_noneb, axis=0) 
    assert data['sed_noneb'].shape[0] == ngal, print('%i SEDs, %i galaxies' % (data['sed_noneb'].shape[0],ngal))
    
    f = h5py.File(fhdf5, 'w') 
    for k in data.keys(): 
        f.create_dataset(k, data=data[k]) 
    f.close() 
    return None 


def _eagle_append_sfrinst(): 
    ''' postprocessing hack to add in instantaneous SFR to preprocessed
    eagle.hdf5 file.

    notes
    -----
    * There's a one element mismatch between EAGLE_mstar_sfr_RefL0100N1504.txt
      and EAGLE_RefL0100_PosVelMstar_allabove1.8e8Msun.txt where the former has
      an extra element (index 29807)  
    '''
    # read Ms and inst. SFR
    fprop1 = os.path.join(dat_dir, 'prop', 'EAGLE_mstar_sfr_RefL0100N1504.txt') # M* and SFR inst
    ms, sfr_inst = np.loadtxt(fprop1, skiprows=1, unpack=True, usecols=[0, 1]) 
    # read in file used for rest of the galaxy properties to ensure line match 
    fprop2 = os.path.join(dat_dir, 'prop', 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt')
    logm_ref = np.loadtxt(fprop2, skiprows=1, unpack=True, usecols=[2])
    
    # mass cut 
    masscut = (ms > 1.8e8) 
    logms = np.log10(ms[masscut]) 
    # skip extra row in full EAGELdata
    extrarow = (np.arange(np.sum(masscut)) == 29807)
    assert np.allclose(logms[~extrarow], logm_ref, rtol=1e-4), "M* don't match"

    logsfr_inst = np.log10(sfr_inst)[masscut][~extrarow]

    fhdf5 = os.path.join(dat_dir, 'sed', 'eagle.hdf5') 
    f = h5py.File(fhdf5, 'a') 
    f.create_dataset('logsfr.inst', data=logsfr_inst) 
    f.close()
    return None


if __name__=="__main__": 
    #compile_seds('simba')  
    #compile_seds('tng') 
    #compile_seds('eagle') 
    _eagle_append_sfrinst()
