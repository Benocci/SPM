#!/bin/sh
#SBATCH -p normal
#SBATCH --nodes=4 
#SBATCH -o ./%j-mpitest.log
#SBATCH -e ./%j-mpitest.err
#SBATCH -t 00:50:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make mpi
echo "Executed of $1 with $3 process"
make run_average_mpi FILE="$1.o" ARGS="$2 $3 $4"
echo "done"