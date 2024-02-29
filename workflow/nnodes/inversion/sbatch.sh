#!/bin/bash
# Begin SLURM Directives
#SBATCH -A GEO111
#SBATCH -t 02:00:00
#SBATCH -N 40
#SBATCH -J Inversion
#SBATCH --mem=0
#SBATCH --requeue
# SBATCH -q debug
#SBATCH --output=R-%x.%j.o.txt
#SBATCH --error=R-%x.%j.e.txt
#SBATCH --array=1-4%1
#SBATCH --mail-type=end
#SBATCH --mail-user=lsawade@princeton.com

module purge
module load PrgEnv-cray amd-mixed cray-mpich craype-accel-amd-gfx90a
module load core-personal hdf5-personal
module unload darshan-runtime

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0

source ~/miniconda3/bin/activate gf

# if [ -f root.pickle ]; then
#     python -c "from nnodes import root; root.init(); root.save(async_save=False)"
# fi

python -c "from nnodes import root; root.run()"

