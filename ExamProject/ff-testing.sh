#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j-ff.log
#SBATCH -e ./%j-ff.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make FFUTW
echo "Execution with N=$1 and thread=$2"
make run_average FILE="FFUTW.o" ARGS="$1 $2 $3"
echo "done"