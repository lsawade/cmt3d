#!/bin/bash
# Begin LSF Directives
#BSUB -P GEO111
#BSUB -W 01:00
#BSUB -nnodes 5
#BSUB -J Database
#BSUB -o inversion.%J.o.txt
#BSUB -e inversion.%J.e.txt

export MPLCONFIGDIR=${SCRATCH}/.matplotlib
export OMP_NUM_THREADS=1

source ~/anaconda3_summit/bin/activate gf

python -c "from nnodes import root; root.run()"
