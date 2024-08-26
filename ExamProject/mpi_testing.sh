#!/bin/sh
#SBATCH -p normal
#SBATCH --nodes=4 
#SBATCH -o ./%j-mpitest.log
#SBATCH -e ./%j-mpitest.err
#SBATCH -t 00:50:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make mpi
echo "Executed of $1 with $3 process"
mpirun -n $3 ./$1.o $2 $4
echo "done"