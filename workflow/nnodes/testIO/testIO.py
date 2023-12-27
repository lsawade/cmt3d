import os
from nnodes import Node

def workflow(node: Node):

    node.concurrent = True
    print('workflow running')

    events = [f"event_{i:02d}" for i in range(10)]

    node.add(subsetwf, events=events)
    node.add(eventloop, events=events)


def subsetwf(node: Node):

    node.concurrent = False
    events = node.events
    print(events)
    chunks = chunkfunc(events, 2)

    node.add(subsetchunk, chunks=chunks)

def subsetchunk(node: Node):

    node.concurrent = True

    if len(node.chunks) >= 1:
        for event_id in node.chunks[0]:

            # Make event dir if it doesn't exist
            if not os.path.exists('events'):
                os.mkdir('events')

            # Make event file
            eventfile = os.path.join('events', event_id)
            node.add(f'touch {eventfile} && sleep 30', event_id=event_id,
                     name=f'create-{eventfile}')

    if len(node.chunks) > 1:
        node.parent.add(subsetchunk, chunks=node.chunks[1:])


def eventloop(node: Node):

    node.concurrent = True

    node.add(inversionloop)
    node.add(eventcheck, events=node.events)


def eventcheck(node: Node):

    node.concurrent = False

    # Get events
    events = node.events

    # Check if any files are there
    node.add('./checkfiles.py events', name='check-files')

    # Read file list
    if os.path.exists('tempfilelist.txt'):

        with open('tempfilelist.txt', 'r') as f:
            event_ids = f.readlines()

        # Add event to loop then remove. Pop events from the list
        idxs = []
        for event_id in event_ids:
            if event_id.strip() in events:
                node.parent[0].add(inversion, event=event_id.strip())
                idxs.append(events.index(event_id.strip()))

        # Pop backwards to not mess with the list
        idxs.sort(reverse=True)

        # Pop events
        for i in idxs:
            events.pop(i)

    if len(events) > 0:
        print(events)
        node.add('sleep 15', name='eventcheck-sleep')
        node.parent.add(eventcheck, events=events)


def inversionloop(node: Node):

    node.concurrent = True
    node.add('sleep 15', name='inversionloop-sleep')
    if len(node.parent[len(node.parent)-1].events) != 0:
        node.parent.add(inversionloop)

def inversion(node: Node):

    print('Inverting event: ', node.event)
    node.add('sleep 60', name='inversion-sleep')



def chunkfunc(sequence, n: int):
    """
    Converse to split, this function preserves the order of the events. But,
    the last chunk has potentially only a single element. For the case, of
    I/O this is not so bad, but if you are looking for performance, `split()`
    above is the better option.

    sequence = sequence to be split
    n = number of elements per chunk

    return list of n-element chunks where the number of chunks depends on the
    length of the sequence
    """
    n = max(1, n)
    return [sequence[i:i+n] for i in range(0, len(sequence), n)]
