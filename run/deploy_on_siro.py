'''

python script to deploy jobs on sirocco 

'''

'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def deploy_abc(sim, dem): 
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#PBS -l nodes=1:ppn=24", 
        "#PBS -N %s_%s" % (sim, dem.strip('_')), 
        "cd $PBS_O_WORKDIR", 
        "export NPROCS=`wc -l $PBS_NODEFILE |gawk '//{print $1}'`", 
        "export PATH='/home/users/hahn/anaconda3/bin:$PATH'", 
        "", 
        "source /home/users/hahn/.bashrc", 
        "conda activate iq ", 
        "", 
        "sim='%s'" % sim, 
        "dem='%s'" % dem, 
        "dist='L2'", 
        "stat='3d'", 
        "ofile='/home/users/hahn/projects/galpopFM/run/_siro/_abc_'$sim'.'$dem'.'$dist'.'$stat'.o'", 
        ">$ofile", 
        "", 
        'mpiexec -n 8 python /home/users/hahn/projects/galpopFM/run/run_abc.py siro $sim $dem $dist $stat $sim"."$dem"."$dist"."$stat 20 False 1000 &>> $ofile', 
        ""
        ]) 
    # create the slurm script execute it and remove it
    f = open('_abc_%s_%s.sh' % (sim, dem.strip('_')),'w')
    f.write(cntnt)
    f.close()
    os.system('qsub _abc_%s_%s.sh' % (sim, dem.strip('_')))
    return None 


if __name__=="__main__": 
    sim = sys.argv[1]
    dem = sys.argv[2]
    deploy_abc(sim, dem) 
