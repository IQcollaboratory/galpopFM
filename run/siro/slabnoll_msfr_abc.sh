# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abctest
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

name="slabnoll_msfr"

mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/abc_slabnoll_msfr.py \
    $name 20 False 1000 &>> "/home/users/hahn/projects/galpopFM/run/siro/abc_"$name".o"
