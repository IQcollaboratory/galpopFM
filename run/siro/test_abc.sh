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

name="test"

mpirun -n 1 --bind-to none python /home/users/hahn/projects/galpopFM/run/abc.py $name 20 1000 24 &> "/home/users/hahn/projects/galpopFM/run/siro/abc_"$name".o"
#python /home/users/hahn/projects/galpopFM/run/abc.py $name 5 40 24 &> "/home/users/hahn/projects/galpopFM/run/siro/abc_"$name".o"

#mpirun -n 10 --bind-to none python /home/users/hahn/projects/galpopFM/run/abc_mpi.py "test1" 20 500 &> "/home/users/hahn/projects/galpopFM/run/siro/abc_"$name".o"
#python /home/users/hahn/projects/galpopFM/run/abc.py "test_mp" 2 50 20 >> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
