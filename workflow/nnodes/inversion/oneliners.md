# Reset step
```bash
for evid in $(nnlog | grep " - " | cut -d " " -f 4); do if [[ $(cat $SCRATCH/gcmt/nnodes/$evid/STEP.txt) -ne 0 ]]; then echo $(($(cat $SCRATCH/gcmt/nnodes/$evid/STEP.txt) - 1)) > $SCRATCH/gcmt/nnodes/$evid/STEP.txt; fi; done
```

# Reset subset init
```bash
for evid in $(ls $LUSTRE/gcmt/events); do rm -f $LUSTRE/gcmt/nnodes/$evid/INIT.txt; done
```
