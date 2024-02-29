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

    # Filter by end index
    if node.end_index:
        print(f"Getting events until idx {node.end_index} ...")
        events = events[:node.end_index]

    # Filter by start index
    if node.start_index:
        print(f"Getting events from idx {node.start_index} ...")
        events = events[node.start_index:]

    # The massdownloader suggest only 4 threads at a time. So here
    # we are doing 4 simultaneous events with each 1 thread
    event_chunks = cmt3d.chunkfunc(events, 4)

    for chunk in event_chunks:

        node.add(download_chunk, chunk=chunk)


def download_chunk(node: Node):

    node.concurrent = True

    for eventfilename in node.chunk:

        eventname = os.path.basename(eventfilename)
        downdir, _, _ = ioi.downloaddir(node.inputfile, eventfilename,
                                        get_dirs_only=True)

        # Get data
        command = f"cmt3d-ioi download {node.inputfile} {eventfilename}"
        node.add_mpi(command, name=eventname, retry=3,
                     cwd=os.path.join(downdir, "nnodes"))

# ----------------------------------------------------------------------------
