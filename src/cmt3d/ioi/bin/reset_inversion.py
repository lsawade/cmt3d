from nnodes import root
import cmt3d.ioi as ioi

def bin(events, level='inversion'):
    """levels: inversion, iteration, step"""

    # Initialize root
    root.init()


    for _wf in root:

        if _wf.name != 'Event-Loop':
            continue

        for _task in _wf:
            if _task.name == 'eventcheck' or _task.name not in events:
                continue

            if level=='inversion':
                print("Resetting:", _task.name)
                _task.reset()
                _task.it=0
                _task.step=0
                ioi.reset_step(_task.outdir)
                ioi.reset_iter(_task.outdir)
            elif level=='maybe-invert':
                if len(_task) == 2:
                    print("Resetting maybe-invert:", _task.name)
                    _task[1].reset()
                    _task.it=0
                    _task.step=0
                    ioi.reset_step(_task.outdir)
                    ioi.reset_iter(_task.outdir)
                else:
                    print("Can't reset maybe-invert:", _task.name)
            elif level=='window':
                if len(_task) == 2 and len(_task[0]) >= 3:
                    print("Resetting window:", _task.name)
                    _task[0][2].reset()
                else:
                    print("Can't reset window:", _task.name)
            elif level=='iteration':
                print("Resetting iteration not available yet")
            elif level=='step':
                print("Resetting step not available yet")
            elif level=='remove':
                print("Removing:", _task.name)
                _wf._children.pop(_wf._children.index(_task))
            else:
                raise ValueError('level not available can be any of: "inversion", "iteration", "step"')

    root.save()

