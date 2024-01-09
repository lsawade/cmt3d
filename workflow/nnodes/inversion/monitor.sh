#!/bin/bash

JOBID=$1

squeue -u lsawade
echo 
echo ----------------------------------
echo

./efficiency.sh $JOBID

echo
echo ----------------------------------
echo

./jobsteps.sh $JOBID
