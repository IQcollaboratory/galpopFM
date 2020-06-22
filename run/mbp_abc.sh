# !/bin/bash
source ~/.bash_profile
conda activate iq 

#sim='simba'
#sim='tng'
sim='eagle'
#dem='slab_noll_m'
#dem='slab_noll_msfr'
#dem='tnorm_noll_msfr'
#dem='slab_noll_msfr_fixbump'
#dem='tnorm_noll_msfr_fixbump'
dem='slab_noll_msfr_kink_fixbump'
dist='L2'
stat='3d'
################################################################################ 
python /Users/ChangHoon/projects/galpopFM/run/run_abc.py \
    mbp $sim $dem $dist $stat $sim"."$dem"."$dist"."$stat 3 False 11
################################################################################ 
