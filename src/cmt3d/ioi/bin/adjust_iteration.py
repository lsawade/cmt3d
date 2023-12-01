import os
from sys import argv, exit
from nnodes import root
from ..functions.log import set_iter, write_log


def bin():
    """
    Checks whether the update iter has failed and if yes adjust the iteration number.

    Usage:

        cmt3d-ioi adjust-iteration

    """

    # Loading the workflow
    root.init()

    for cmtinversion in root[0]:

        if cmtinversion.elapsed:
            continue

        # Stage 2 is the inverst or not stage.
        stage = cmtinversion[1]

        # If the event is not being inverted due to e.g. not enough windows
        # this length is 0
        N = len(stage)

        # So skip if 0
        if not N > 0:
            continue

        # If not zero grab last entry check if it's an iteration and reset it
        if stage[N-1].name != 'iteration':
            print(f'  Skipping {cmtinversion.name}. Iteration not last step')
            continue

        # Get iteration in the workflow
        iteration = stage[N-1]

        # Skip if the iteration is done already
        if iteration.elapsed:
            print(f'  Skipping {cmtinversion.name}. Iteration elapsed')
            continue

        # Skip if iterations substeps are not created yet
        if not len(iteration) > 0:
            # Check whether iteration was already added because iteration check
            # was started.
            prev_iteration = stage[N-2]
            prev_iteration_check = prev_iteration[3]

            # If iteration check was done, skip
            if prev_iteration_check.elapsed:
                print(f'  Skipping {cmtinversion.name}. Iteration_check already done')
                continue
            else:
                N = N-1
                iteration = stage[N-1]

        # Check if step before iteration check is done
        linesearch = iteration[2]

        # Skip if linesearch is not done
        if not linesearch.elapsed:
            print(f'  Skipping {cmtinversion.name}. linesearch not done')
            continue

        # Get iteration check
        iteration_check = iteration[3]

        # If it's done already skip
        if iteration_check.elapsed:
            print(f'  Skipping {cmtinversion.name}. Iteration_check already done')
            continue

        # Get update_iter task
        update_iter = iteration_check[0]

        # If done skip
        if update_iter.elapsed:
            print(f'  Skipping {cmtinversion.name}. Update_iter already done.')
            continue

        # Reset the step in the directory.
        write_log(iteration.outdir, "=========================================")
        write_log(iteration.outdir, "Fixing iteration number ... ")
        write_log(iteration.outdir, "=========================================")
        set_iter(iteration.outdir, N-3)
        update_iter.reset()

        print(f'Resetting {cmtinversion.name} {iteration} '
                f'{N-1} in nnodes and {N-3} on disk.')

    root.save()