#!/bin/bash
#SBATCH -J Download
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -o slurm.%J.o
#SBATCH -e slurm.%J.e
#SBATCH -A GEO111


. /ccs/home/lsawade/dtn_miniconda3/bin/activate gf-dtn
conda activate gf-dtn

python -c "from nnodes import root; root.run()"
