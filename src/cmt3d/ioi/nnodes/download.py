# %%
import os
from nnodes import Node
import cmt3d
import cmt3d.ioi as ioi

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):
    node.concurrent = True

    # # Events to be inverted
    # print('Checking events TODO ...')
    # eventfiles = check_events_todo(node.inputfile)

    # # Specific event id(s)
    # eventflag = True if node.eventid is not None else False
    # print('Specfic event(s)?', eventflag)

    # # Maximum inversion flag
    # maxflag = True if node.max_events != 0 else False
    # print('Maximum # of events?', maxflag)

    # # If eventid in files only use the ids
    # if eventflag:
    #     print('Getting specific events...')
    #     nevents = []

    #     eventnames = [
    #         cmt3d.CMTSource.from_CMTSOLUTION_file(_file).eventname
    #         for _file in eventfiles]

    #     # Check whether multiple eventids are requested
    #     if isinstance(node.eventid, list):
    #         eventids = node.eventid
    #     else:
    #         eventids = [node.eventid]

    #     # If id in eventnames, add the eventfile
    #     for _id in eventids:
    #         idx = eventnames.index(_id)
    #         nevents.append(eventfiles[idx])

    #     eventfiles = nevents

    # # If max number of inversion select first X
    # if maxflag:
    #     print('Getting max # of events ...')
    #     eventfiles = eventfiles[:node.max_events]

    # # print list of events if not longer than 10
    # if len(eventfiles) < 11:
    #     for _ev in eventfiles:
    #         print(_ev)

    # Loop over inversions
    # for event in eventfiles:
    #     eventname = cmt3d.CMTSource.from_CMTSOLUTION_file(event).eventname
    #     out = optimdir(node.inputfile, event, get_dirs_only=True)
    #     outdir = out[0]

    scriptdir = "/ccs/home/lsawade/gcmt/cmt3d/scripts"
    datadir = os.path.join(scriptdir, 'data')
    subsetdir = os.path.join(datadir, 'subsets')
    eventdir = os.path.join(datadir, 'events')
    cmtfilename = os.path.join(eventdir, 'C201009071613A')

    eventname = os.path.basename(cmtfilename)
    out = ioi.optimdir(node.inputfile, cmtfilename, get_dirs_only=True)
    outdir = out[0]

    node.add(cmtinversion, concurrent=False, name=eventname,
             eventname=eventname,
             outdir=outdir,
             eventfile=cmtfilename,
             subsetdir=subsetdir,
             log=os.path.join(outdir, 'logs'))
# -----------------------------------------------------------------------------


# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event
def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')
    node.add(download)


# Performs iteration
def download(node: Node):

    node.concurrent = False

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
        node.add(ioi.get_data, args=(node.outdir,))
