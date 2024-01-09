#!/bin/bash

JOBID=$1

sacct -j $JOBID -o "User,JobID%20,ReqMem,ReqCPUS,TotalCPU,Elapsed,MaxRSS,NodeList%30,State" | grep RUNNING | head
echo "     :            :              :                 :             "
sacct -j $JOBID -o "User,JobID%20,ReqMem,ReqCPUS,TotalCPU,Elapsed,MaxRSS,NodeList%30,State" | grep RUNNING | tail
