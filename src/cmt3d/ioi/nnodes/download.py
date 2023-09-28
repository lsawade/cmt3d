# %%
import os
from nnodes import Node
import cmt3d
import cmt3d.ioi as ioi




# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):
    node.concurrent = True

    print('Hello')
    node.concurrent = False

    # Get input file
    inputparams = cmt3d.read_yaml(node.inputfile)

    # Get todo events
    event_dir = inputparams['events']

    # Get all events
    events = [os.path.join(event_dir, ev_file)
              for ev_file in os.listdir(event_dir)]

    # Sort the events
    events.sort()

    # The massdownloader suggest only 4 threads at a time. So here
    # we are doing 4 simultaneous events with each 1 thread
    event_chunks = cmt3d.chunkfunc(events, 4)


    for chunk in event_chunks:

        node.add(download_chunk, concurrent=True, chunk=chunk)


def download_chunk(node: Node):

    for eventfilename in node.chunk:

        eventname = os.path.basename(eventfilename)

        # Get the database directory
        out = ioi.optimdir(node.inputfile, eventfilename, get_dirs_only=True)

        node.add(download, name=eventname,
                 eventname=eventname,
                 outdir=out[0],
                 eventfile=eventfilename,
                 log=os.path.join(out[0], 'logs'))

# -----------------------------------------------------------------------------

# Performs iteration
def download(node: Node):

    try:
        # Will fail if ITER.txt does not exist
        firstiterflag = ioi.get_iter(node.outdir) >= 0

    except Exception:
        firstiterflag = True

    if firstiterflag:

        # Create the inversion directory/makesure all things are in place
        node.add(ioi.create_forward_dirs, args=(node.eventfile, node.inputfile),
                 name=f"create-dir", cwd=node.log)

    # Get data
    node.add_mpi(ioi.get_data, args=(node.outdir,), cwd=node.log)
