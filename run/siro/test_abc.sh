# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abctest
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

name="test"

#mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/abc_schwimmbad.py $name 20 1000 
mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/abc_schwimmbad_restart.py $name 20 0 &>> "/home/users/hahn/projects/galpopFM/run/siro/abc_"$name".o"

#mpirun -n 1 -np 20  --bind-to none python /home/users/hahn/projects/galpopFM/run/abc_mpi.py "test1" 20 40 &> "/home/users/hahn/projects/galpopFM/run/siro/abc_mpi.o"
#python /home/users/hahn/projects/galpopFM/run/abc.py "test_mp" 2 50 20 >> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
