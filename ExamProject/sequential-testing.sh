#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j-seq.log
#SBATCH -e ./%j-seq.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make $1
echo "Execution of $1 with N=$2"
make run_average FILE="$1.o" ARGS="$2 $3"
echo "done"