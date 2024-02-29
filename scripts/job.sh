#!/bin/bash
#SBATCH -A GEO111
#SBATCH -t 60:00
#SBATCH -N1
#SBATCH -J Event_fix
#SBATCH --output=Event_fix.%J.o.txt
#SBATCH --error=Event_fix.%J.e.txt
#SBATCH --mem=0


module purge
module load PrgEnv-cray amd-mixed cray-mpich craype-accel-amd-gfx90a
module load core-personal hdf5-personal
module unload darshan-runtime

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0

source ~/miniconda3/bin/activate gf

srun -n 50 --unbuffered python fix_events.py