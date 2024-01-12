#!/bin/bash
#SBATCH -A GEO111
#SBATCH -t 01:00
#SBATCH -N1
#SBATCH -J Event_fix
#SBATCH --output=Event_fix.%J.o.txt
#SBATCH --error=Event_fix.%J.e.txt
#SBATCH --mem=0

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
export OMP_NUM_THREADS=1
module load openmpi/4.0.4 hdf5

source /sw/andes/python/3.7/anaconda-base/bin/activate gf-andes

srun -n 32 --unbuffered python fix_events.py