#!bin/bash/

sfr0=sfrmin
#dem=simba.slab_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr_fixbump.L2.3d
dem=simba.slab_noll_mssfr_fixbump.L2_only.3d.sfrmin
#dem=simba.slab_noll_msfr_kink_fixbump.L2.3d
#python ~/projects/galpopFM/run/plot_abc.py True $dem 23 26 $sfr0 

#dem=tng.slab_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr_fixbump.L2.3d
dem=tng.slab_noll_mssfr_fixbump.L2_only.3d.sfrmin
#dem=tng.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 23 25 $sfr0

#dem=eagle.slab_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr_fixbump.L2.3d
dem=eagle.slab_noll_mssfr_fixbump.L2_only.3d.sfrmin
#dem=eagle.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 26 27 $sfr0 
