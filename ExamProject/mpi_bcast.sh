#!/bin/sh
#SBATCH -p normal
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j-mpibcast.log
#SBATCH -e ./%j-mpibcast.err
#SBATCH -t 00:50:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make mpi
echo "Executed of $1 with $3 process"
mpirun -n $3 ./$1.o $2 $4
echo "done"