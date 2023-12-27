#BSUB -P GEO111
#BSUB -W 01:00
#BSUB -nnodes 1
#BSUB -J Event_fix
#BSUB -o Event_fix.%J.o.txt
#BSUB -e Event_fix.%J.e.txt
#BSUB -B


export MPLCONFIGDIR=${SCRATCH}/.matplotlib
export OMP_NUM_THREADS=1

source ~/anaconda3_summit/bin/activate gf

jsrun -n 42 -a 1 -c 1 python fix_events.py