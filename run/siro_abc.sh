# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N simba_slabnollmssfrfixbump_sfrmin
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
conda activate iq 

sim='simba'
#sim='tng'
#sim='eagle'

#dem='slab_noll_msfr'
#dem='tnorm_noll_msfr'
#dem='slab_noll_msfr_fixbump'
#dem='tnorm_noll_msfr_fixbump'
#dem='slab_noll_msfr_kink_fixbump'

dem='slab_noll_mssfr_fixbump'
dist='L2_only'
stat='3d'
sfr0='sfrmin' # 'adhoc' 
################################################################################ 
# log
################################################################################ 
# 09/23/2020: implemented central+satellite; modified observational sumstat 
# 4/23/2020: running after implement instantaneous SFR=0 pre-sampling
# 10/20/2020: running after changing parameterizations from SFR to SSFR 
################################################################################ 
ofile="/home/users/hahn/projects/galpopFM/run/_siro/_abc_"$sim"."$dem"."$dist"."$stat"."$sfr0".o"

>$ofile 

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/run_abc.py \
    siro $sim $dem $dist $stat $sfr0 $sim"."$dem"."$dist"."$stat"."$sfr0 30 False 1000 \
    >> $ofile

# restart ABC 
#mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/run_abc.py \
#    siro $sim $dem $dist $stat $sfr0 $sim"."$dem"."$dist"."$stat"."$sfr0 30 True 19 \
#    &>> $ofile
################################################################################ 
