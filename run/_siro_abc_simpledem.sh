# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N sim_slabnollsimple
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

sim='simba' 
edm="slab_noll_simple"
################################################################################ 
ofile="/home/users/hahn/projects/galpopFM/run/_siro/abc_"$sim"_"$edm".o"
>$ofile 

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/_abc_siro_simpledem.py \
    simba slab_noll_simple simba_slab_noll_simple 20 False 1000 \
    &>> $ofile
################################################################################ 
