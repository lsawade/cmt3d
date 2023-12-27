#!/bin/bash
# Begin LSF Directives
#BSUB -P GEO111
#BSUB -W 02:00
#BSUB -nnodes 50
#BSUB -J Inversion
#BSUB -o inversion.%J.o.txt
#BSUB -e inversion.%J.e.txt
#BSUB -B
#### XB BSUB -alloc_flags "gpumps smt1"

export MPLCONFIGDIR=${SCRATCH}/.matplotlib
export OMP_NUM_THREADS=1

source ~/anaconda3_summit/bin/activate gf

python -c "from nnodes import root; root.run()"
