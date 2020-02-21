# !/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -N abctest
#PBS -m bea
#PBS -M changhoonhahn@lbl.gov
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source /home/users/hahn/.bashrc
source activate iq 

mpirun -np 1 --bind-to none python /home/users/hahn/projects/galpopFM/run/abc.py "test_mp" 20 1000 20 &> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o


#mpirun -np 20 python /home/users/hahn/projects/galpopFM/run/abc_mpi.py "test" 2 50 &> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
#python /home/users/hahn/projects/galpopFM/run/abc.py "test_mp" 2 50 20 >> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
