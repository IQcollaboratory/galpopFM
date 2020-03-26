#!bin/bash/

#dem=simba.slab_noll_m.L2.3d
#dem=tng.slab_noll_m.L2.3d
#dem=simba.slab_noll_msfr.L2.3d
dem=tng.slab_noll_msfr.L2.3d

for i in {0..10}; do 
    python /Users/ChangHoon/projects/galpopFM/run/plot_abc.py $dem $i 
done
