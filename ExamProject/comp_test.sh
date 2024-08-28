#!/bin/sh
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH -o ./%j-cmp.log
#SBATCH -e ./%j-cmp.err
#SBATCH -t 02:00:00

echo "Test executed on: $SLURM_JOB_NODELIST"
make SequentialUTWv2
make FFUTWv2
make mpi
echo "Execution with N=$1"
echo "Sequential time: "
make run_single FILE="SequentialUTWv2.o" ARGS="$1 $3"
echo "FF time with $2 threads:"
make run_single FILE="FFUTWv2.o" ARGS="$1 $2 $3"
echo "MPI time with $2 processes:"
mpirun -n $2 ./MPIUTW_Allgather.o $1 $3
echo "done"