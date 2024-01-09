#!/bin/bash

JOBID=$1
TIME_LIMIT_IN_SEC=$2

if [ -z "$JOBID" ]; then
    echo "Please provide a job id as first argument"
    exit 1
fi

if [ -z "$TIME_LIMIT_IN_SEC" ]; then
    echo "The second argument is the time limit in seconds. "
    echo "If a step takes longer than this time, it will be cancelled."
    echo "Default is 10 minutes (600 seconds)."
    echo "Time limit is set to $TIME_LIMIT_IN_SEC seconds."
    TIME_LIMIT_IN_SEC=600
else
    echo "Time limit is set to $TIME_LIMIT_IN_SEC seconds."
fi

for jobtime in $(sacct -j $JOBID -o "JobID%20,Elapsed,State" | grep RUNNING | tail -n +4 | awk  'BEGIN { OFS = ";" }{print $1, $2}');
do
    # Get step number
    step=$(echo $jobtime | cut -d ";" -f 1);

    # Get time stamp
    stamp=$(echo $jobtime | cut -d ";" -f 2);
    hh=$(echo $stamp | cut -d ":" -f 1 | sed 's/^0*//');
    mm=$(echo $stamp | cut -d ":" -f 2 | sed 's/^0*//');  nozero=$(echo $machinenumber )
    ss=$(echo $stamp | cut -d ":" -f 3 | sed 's/^0*//');

    # Check if hh, mm, ss are empty
    if [ -z "$hh" ]; then hh=0; fi;
    if [ -z "$mm" ]; then mm=0; fi;
    if [ -z "$ss" ]; then ss=0; fi;

    # Get time in seconds
    time_s=$(($hh*3600+$mm*60+$ss*1));

    if [ $time_s -gt $TIME_LIMIT_IN_SEC ];
    then
        echo "Step $step is running for more than $TIME_LIMIT_IN_SEC seconds. Cancelling it."
        scancel $step;
    fi;
done