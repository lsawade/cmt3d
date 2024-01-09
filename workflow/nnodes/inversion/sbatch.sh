#!/bin/bash
# Begin LSF Directives
#SBATCH -A GEO111
#SBATCH -t 04:00:00
#SBATCH -N 10
#SBATCH -J Inversion
#SBATCH --mem=0
#SBATCH --output=R-%x.%j.o.txt
#SBATCH --error=R-%x.%j.e.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=lsawade@princeton.com

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
export OMP_NUM_THREADS=1

source /sw/andes/python/3.7/anaconda-base/bin/activate gf-andes

module load openmpi/4.0.4 hdf5

python -c "from nnodes import root; root.run()"

