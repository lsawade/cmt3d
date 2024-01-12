from nnodes import root
import cmt3d.ioi as ioi

def bin(events: list | None = None, print_mode='all'):
    """print_mode: all, done, not_done"""

    if events is not None:
        if len(events) == 0:
            events = None

    # Initialize root
    root.init()

    events_done = []
    events_not_done = []

    # Loop over workflow
    for _wf in root:

        if _wf.name != 'Event-Loop':
            continue


        for _task in _wf:
            # Skip eventchecks
            if 'Event-Check' in _task.name or 'eventcheck' in _task.name:
                continue

            # Skip if not in events if events are given
            if events is not None:
                if _task.name not in events:
                    continue

            if _task.done:
                events_done.append(_task)
            else:
                events_not_done.append(_task)

    # Sort events
    events_done.sort(key=lambda x: x.name)
    events_not_done.sort(key=lambda x: x.name)

    if print_mode == 'all':
        print(62*"=")
        print(25*"=", "   DONE   ", 25*"=")
        print(62*"=")

        # Print ID and elapsed time
        for _ev in events_done:
            print(_ev.name, _ev.elapsed)

        print(62*"=")
        print(25*"=", " NOT DONE ", 25*"=")
        print(62*"=")

        # Only print ID
        for _ev in events_not_done:
            print(_ev.name)

    elif print_mode == 'done':

        # Only print ID
        for _ev in events_done:
            print(_ev.name)


    elif print_mode == 'not_done':

        # Only print ID
        for _ev in events_not_done:
            print(_ev)
    else:
        raise ValueError(f"print_mode={print_mode} not supported")




