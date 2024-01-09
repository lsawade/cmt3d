#!/bin/bash

ev=$1
DB=/lustre/orion/geo111/scratch/lsawade/gcmt/data;


for file in $(ls $DB/$ev/waveforms/);
do filesize=$(wc -c <$DB/$ev/waveforms/$file);
   if [[ $filesize -eq 0 ]];
   then
       rm -f $DB/$ev/waveforms/$file;
   fi;
done
