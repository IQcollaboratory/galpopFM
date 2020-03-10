# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N sim_slabnollm
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

sim='simba' 
edm="slab_noll_m"
#edm="slab_noll_msfr"
################################################################################ 
ofile="/home/users/hahn/projects/galpopFM/run/_siro/abc_"$sim"_"$edm".o"
>$ofile 

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/abc_siro.py \
    $sim $edm $sim"_"$edm 20 False 1000 \
    &>> $ofile
################################################################################ 
