# !/bin/bash
# PBS -l nodes=1:ppn=24
# PBS -N abctest
cd $PBS_O_WORKDIR
export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`
export PATH="/home/users/hahn/anaconda3/bin:$PATH"

source ~/.bashrc    
source activate iq 

#mpirun -np 24 python /home/users/hahn/projects/galpopFM/run/abc.py "test" 10 1000 >> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
python /home/users/hahn/projects/galpopFM/run/abc.py "test_mbp" 10 1000 24 &>> /home/users/hahn/projects/galpopFM/run/siro/test_abc.o
