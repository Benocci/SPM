#!/bin/sh
#SBATCH -p normal
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j.log
#SBATCH -e ./%j.err
#SBATCH -t 00:50:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make mpi
mpirun -n 4 ./MPIUTW.o $1 $2 $3
echo "done"