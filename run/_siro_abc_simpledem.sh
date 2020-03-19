# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N sim.slabnollsimple.L2.3d
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

sim='simba'
dem='slab_noll_simple'
dist='L2'
stat='3d'
################################################################################ 
ofile="/home/users/hahn/projects/galpopFM/run/_siro/abc_"$sim"."$dem"."$dist"."$stat".o"
>$ofile 

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/_abc_siro_simpledem.py \
    $sim $dem $dist $stat $sim"."$dem"."$dist"."$stat 20 False 1000 \
    &>> $ofile
################################################################################ 
