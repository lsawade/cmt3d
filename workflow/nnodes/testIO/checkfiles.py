#!/bin/env python


def check_files():
    import os
    import sys
    from glob import glob

    # Check files
    event_ids = glob(os.path.join(sys.argv[1],'*'))

    with open('tempfilelist.txt', 'w') as f:
        for event_id in event_ids:
            print(os.path.basename(event_id), file=f)


if __name__ == '__main__':
    check_files()