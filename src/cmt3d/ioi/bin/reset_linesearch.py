import os
from sys import argv, exit
from nnodes import root
from ..functions.log import reset_step, write_log


def bin():
    """
    Resets the linesearch for all inversions running with nnodes.

    Usage:

        cmt3d-ioi reset-linesearch

    """

    print("Current Dir:", os.getcwd())

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
        if stage[N-1].name == 'iteration':

            # Reset the stage in the workflow
            iteration = stage[N-1]
            iteration.reset()

            # Reset the step in the directory.
            write_log(iteration.outdir, "=========================================")
            write_log(iteration.outdir, "Resetting last iteration ... ")
            write_log(iteration.outdir, "=========================================")
            reset_step(iteration.outdir)

            print(f'Resetting {cmtinversion.name} {iteration} '
                  f'{N-1} in nnodes and {N-3} on disk.')

    root.save()