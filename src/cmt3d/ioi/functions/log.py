import os
import sys
import threading
try:
    import thread
except ImportError:
    import _thread as thread

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt


def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


@exit_after(10)
def write_to_file_with_timeout(file, message, mode='a'):
    with open(file, mode) as f:
        f.write(message)


def write_to_file(file, message, mode="a"):
    """Wraps the file with timeout, since we want to catch
    KeyboardInterrupt, remove the file and write to the file before exiting."""

    try:
        write_to_file_with_timeout(file, message, mode=mode)

    except (KeyboardInterrupt, OSError) as e:

        print("KeyboardInterrupt")
        print("Could not write to file " + file + ". Removing file, trying to write again.")
        write_to_file_with_timeout(
            file,
            'Input/Output error occurred. Removed file: ' + file + '\n', mode="a")
        os.remove(file)
        write_to_file_with_timeout(file, 'Restarted: ' + file + '\n', mode="a")
        write_to_file_with_timeout(file, message, mode="a")


def write_status(outdir, message):

    statdir = outdir
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    write_to_file(file, message, mode="w")


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

    write_to_file(file, message + "\n", mode="a")


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

    # Writing to file with timeout
    write_to_file(path, f"{iter + 1:d}", mode="w")

def set_iter(outdir, it):

    # Iter file path
    fname = 'ITER.txt'
    path = os.path.join(outdir, fname)

    # Writing to file with timeout
    write_to_file(path, f"{it:d}", mode="w")

def reset_iter(outdir):

    # Iter file path
    fname = 'ITER.txt'
    path = os.path.join(outdir, fname)

    write_to_file(path, f"{0:d}", mode="w")

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

    write_to_file(path, f"{step + 1:d}", mode="w")


def set_step(outdir, ls):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    with open(path, "w") as f:
        f.write(f"{ls:d}")


def reset_step(outdir):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    write_to_file(path, f"{0:d}", mode="w")


def get_step(outdir):

    # Iter file path
    fname = 'STEP.txt'
    path = os.path.join(outdir, fname)

    with open(path, "r") as f:
        step = int(f.read())

    return step
