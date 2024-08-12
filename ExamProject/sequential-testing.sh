#!/bin/sh
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -o ./%j-seq.log
#SBATCH -e ./%j-seq.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make SequentialUTW
echo "Execution with N=$1"
make run_average FILE="SequentialUTW.o" ARGS="$1 $2"
echo "done"