from sys import argv, exit
from nnodes import root
from ..functions.log import reset_step, write_log, set_step


def bin():
    """
    Resets the linesearch for all inversions running with nnodes.

    Usage:

        cmt3d-ioi reset-linesearch

    """

    # Initialize the workflow
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
            continue

        # Get iteration in the workflow
        iteration = stage[N-1]

        # If the iteration is not finished, reset the iteration and continue
        if not len(iteration) > 0:
            print(f'  Skipping {cmtinversion.name}. Iteration not yet started')
            iteration.reset()
            continue

        # If step prior to linesearch is not yet finished, reset iteration
        # and continue with next event
        if not iteration[1].elapsed:
            print(f'  Skipping {cmtinversion.name}. Iteration not yet started')
            iteration.reset()
            continue

        # So skip if 0
        linesearch = iteration[2]

        # Linesearch is split into search steps. To reset the last step simply,
        # reset the last step.
        M = len(linesearch)

        # So skip if 0
        if not M > 0:
            print(f'  Skipping {cmtinversion.name}. No steps to reset.')
            continue

        # Also skip if linesearch done
        if linesearch.elapsed:
            print(f'  Skipping {cmtinversion.name}. linesearch already done.')
            continue

        # Get the last step
        search_step = linesearch[M-1]

        # If the last step is not finished, reset the last step and continue
        # with next event
        if search_step.elapsed:
            print(f'  Skipping {cmtinversion.name}. last step done.')
            continue


        # Reset the step in the directory.
        write_log(search_step.outdir, "=========================================")
        write_log(search_step.outdir, "Resetting search step ... ")
        write_log(search_step.outdir, "=========================================")
        search_step.reset()
        set_step(search_step.outdir, M-1)

        print(f'Resetting {cmtinversion.name} {search_step} to {M-1} for '
              f'iteration {iteration} {N-1} in nnodes and {N-3} on disk.')


    root.save()



