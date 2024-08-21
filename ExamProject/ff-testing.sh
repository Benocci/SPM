#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j-ff.log
#SBATCH -e ./%j-ff.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make $1
echo "Execution of $1 with N=$2 and thread=$3"
make run_average FILE="$1.o" ARGS="$2 $3 $4"
echo "done"