import os


def write_status(outdir, message):

    statdir = outdir
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "w") as f:
        f.truncate(0)
        f.write(message)


def read_status(outdir):

    statdir = outdir
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "r") as f:
        message = f.read()
    
    return message


def write_log(outdir, message):

    logdir = outdir
    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "a") as f:
        f.write(message + "\n")


def clear_log(outdir):

    logdir = outdir

    fname = "LOG.txt"
    file = os.path.join(logdir, fname)

    with open(file, "w") as f:
        f.close()


def update_iter(outdir):

    # Iter file path
    fname = 'ITER.txt'
    path = os.path.join(outdir, fname)

    with open(path, "r") as f:
        iter = int(f.read())

    with open(path, "w") as f:
        f.write(f"{iter + 1:d}")

def reset_iter(outdir):

    # Iter file path
    fname = 'ITER.txt'
    path = os.path.join(outdir, fname)

    with open(path, "w") as f:
        f.write(f"{0:d}")


def get_iter(outdir):

    # Iter file path
    fname = 'ITER.txt'
    path = os.path.join(outdir, fname)

    with open(path, "r") as f:
        it = int(f.read())

    return it


def update_step(outdir):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    with open(path, "r") as f:
        step = int(f.read())

    with open(path, "w") as f:
        f.write(f"{step + 1:d}")

def reset_step(outdir):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    with open(path, "w") as f:
        f.write(f"{0:d}")

def get_step(outdir):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    with open(path, "r") as f:
        step = int(f.read())

    return step