#!bin/bash/

#dem=simba.slab_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr_fixbump.L2.3d
dem=simba.slab_noll_msfr_fixbump.L2.3d
#dem=simba.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 14 15

#dem=tng.slab_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr_fixbump.L2.3d
dem=tng.slab_noll_msfr_fixbump.L2.3d
#dem=tng.slab_noll_msfr_kink_fixbump.L2.3d
#python ~/projects/galpopFM/run/plot_abc.py True $dem 7 7

#dem=eagle.slab_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr_fixbump.L2.3d
dem=eagle.slab_noll_msfr_fixbump.L2.3d 
#dem=eagle.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 19 19
