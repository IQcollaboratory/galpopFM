# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N eagle_tnormnollmsfr.L2.3d
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
conda activate iq 

#sim='simba'
#sim='tng'
sim='eagle'
#dem='slab_noll_msfr'
#dem='tnorm_noll_msfr'
#dem='slab_noll_msfr_fixbump'
#dem='tnorm_noll_msfr_fixbump'
dem='slab_noll_msfr_kink_fixbump'
dist='L2'
stat='3d'
################################################################################ 
# log
################################################################################ 
# 4/23/2020: running after implement instantaneous SFR=0 pre-sampling
################################################################################ 
ofile="/home/users/hahn/projects/galpopFM/run/_siro/_abc_"$sim"."$dem"."$dist"."$stat".o"
>$ofile 

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/run_abc.py \
    siro $sim $dem $dist $stat $sim"."$dem"."$dist"."$stat 20 False 1000 \
    &>> $ofile
################################################################################ 
