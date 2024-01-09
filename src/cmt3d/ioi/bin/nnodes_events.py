from nnodes import root

def bin():
    """levels: inversion, iteration, step"""

    # Initialize root
    root.init()

    eventsnames = []

    # Loop over workflow
    for _wf in root:

        if _wf.name != 'Event-Loop':
            continue


        for _task in _wf:
            # Skip eventchecks
            if _task.name == 'eventcheck':
                continue

            eventsnames.append(_task.name)

    # Sort events
    eventsnames.sort()


    for _ev in list(set(eventsnames)):
        print(_ev)


