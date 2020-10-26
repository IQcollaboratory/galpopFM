#!bin/bash/

#dem=simba.slab_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr.L2.3d
#dem=simba.tnorm_noll_msfr_fixbump.L2.3d
dem=simba.slab_noll_mssfr_fixbump.L2_only.3d
#dem=simba.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 11 12

#dem=tng.slab_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr.L2.3d
#dem=tng.tnorm_noll_msfr_fixbump.L2.3d
dem=tng.slab_noll_mssfr_fixbump.L2_only.3d
#dem=tng.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 10 16

#dem=eagle.slab_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr.L2.3d
#dem=eagle.tnorm_noll_msfr_fixbump.L2.3d
dem=eagle.slab_noll_mssfr_fixbump.L2_only.3d 
#dem=eagle.slab_noll_msfr_kink_fixbump.L2.3d
python ~/projects/galpopFM/run/plot_abc.py True $dem 11 19
