from nnodes import root
import cmt3d.ioi as ioi

def bin(events, level='inversion'):
    """levels: inversion, iteration, step"""

    # Initialize root
    root.init()


    for _wf in root:

        if _wf.name != 'Event-Loop' and _wf.name != 'Subset-Events':
            continue

        if _wf.name == 'Subset-Events' and (level=='remove' or level=='inversion'):
            print("entering subset loop")
            for _task in _wf:
                for _chunk in _task:
                    for _event in events:
                        # print("_chunk.name:", _chunk.name)
                        if _event in _chunk.name:
                            if level=='inversion':
                                print("Resetting:", _chunk.name)
                                _chunk.reset()
                            elif level=='remove':
                                print("Removing:", _event, "from subset loop")
                                _task._children.pop(_task._children.index(_chunk))
            continue

        Nwf = len(_wf)
        lasteventcheck = [_i for _i, _task in enumerate(_wf) if 'Event-Check' in _task.name][-1]
        taskoutdirdict = {}
        removeidx = []
        for _i, _task in enumerate(_wf):

            # print('Event-Check', _i, _task.name)

            if _task.name in events:
                pass
            elif 'Event-Check' in _task.name:

                if (_i == lasteventcheck):
                    for _event, _eventfile in taskoutdirdict.items():
                        print("Adding to queue:", _event, _eventfile)
                        _task.events.append(_eventfile)
                continue
            else:
                continue


            if level=='inversion':
                print("Removing to reset:", _task.name)
                removeidx.append(_wf._children.index(_task))
                taskoutdirdict[_task.name] = _task.eventfile

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
                removeidx.append(_wf._children.index(_task))
            else:
                raise ValueError('level not available can be any of: "inversion", "iteration", "step"')


        if level=='inversion' or level=='remove':

            for _i in removeidx[::-1]:
                print("Removing:", _wf._children[_i].name)
                _wf._children.pop(_i)

    root.save()

