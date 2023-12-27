# %%
import os
import cmt3d
import cmt3d.ioi as ioi


# %%
dbdir = "/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes"
outdirs = os.listdir(dbdir)

# %%

N = 0
events = []
for _i, _eventid in enumerate(outdirs):

    # Get absolute directory
    od = os.path.join(dbdir, _eventid)

    # Check if status exists. If not, skip.
    if not os.path.exists(os.path.join(od, "STATUS.txt")):
        continue

    # Only read event if you status is finished
    status = ioi.read_status(od)
    if "FINISHED" not in status:
        continue

    N += 1
    events.append(od)
    print(_eventid)

# %%

gcmt___eventdir = os.path.join('events', 'gcmt')
_cmt3d_eventdir = os.path.join('events', 'cmt3d')
gcmt3d_eventdir = os.path.join('events', 'gcmt3d')
gcmt3df_eventdir = os.path.join('events', 'gcmt3d_fix')

# Make event dirs
if not os.path.exists(gcmt___eventdir):
    os.makedirs(gcmt___eventdir)

if not os.path.exists(_cmt3d_eventdir):
    os.makedirs(_cmt3d_eventdir)

if not os.path.exists(gcmt3d_eventdir):
    os.makedirs(gcmt3d_eventdir)

if not os.path.exists(gcmt3df_eventdir):
    os.makedirs(gcmt3df_eventdir)



for od in events:

    # Get initial model
    icmt = ioi.get_cmt(od, it=0, ls=0)

    # Get final model
    iter = ioi.get_iter(od)

    # Get final model
    fcmt = ioi.get_cmt(od, it=iter, ls=0)

    # cmt3d events
    ccmtfile = os.path.join("..","cmt3d_events", f"{icmt.eventname}")
    ccmt = cmt3d.CMTSource.from_CMTSOLUTION_file(ccmtfile)

    # Get fixed event
    try:
        fixcmtfile = os.path.join(od, "meta", f"{fcmt.eventname}_GCMT3D_fix")
        fixcmt = cmt3d.CMTSource.from_CMTSOLUTION_file(fixcmtfile)
    except:
        continue

    # Write to file
    icmt.write_CMTSOLUTION_file(os.path.join(gcmt___eventdir, f"{icmt.eventname}"))
    ccmt.write_CMTSOLUTION_file(os.path.join(_cmt3d_eventdir, f"{ccmt.eventname}"))
    fcmt.write_CMTSOLUTION_file(os.path.join(gcmt3d_eventdir, f"{fcmt.eventname}"))
    fixcmt.write_CMTSOLUTION_file(os.path.join(gcmt3df_eventdir, f"{fixcmt.eventname}"))


