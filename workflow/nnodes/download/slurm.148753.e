
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.



CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.



CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.



CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.



CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.


Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 320, in _exec_task
    await result
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 242, in mpiexec
    raise err
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 214, in mpiexec
    raise RuntimeError(d.read(f'{fname}.error'))
RuntimeError: Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 94, in <module>
    _call(0, 0)
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 69, in _call
    if asyncio.iscoroutine(result := func(*a)):
                                     ^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 29, in get_data
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/source.py", line 90, in from_CMTSOLUTION_file
    with open(filename, "rt") as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/C091703C/meta/init_model.cmt'


Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 320, in _exec_task
    await result
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 242, in mpiexec
    raise err
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 214, in mpiexec
    raise RuntimeError(d.read(f'{fname}.error'))
RuntimeError: Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 94, in <module>
    _call(0, 0)
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 69, in _call
    if asyncio.iscoroutine(result := func(*a)):
                                     ^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 29, in get_data
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/source.py", line 90, in from_CMTSOLUTION_file
    with open(filename, "rt") as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/C091704C/meta/init_model.cmt'


Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 320, in _exec_task
    await result
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 242, in mpiexec
    raise err
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 214, in mpiexec
    raise RuntimeError(d.read(f'{fname}.error'))
RuntimeError: Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 94, in <module>
    _call(0, 0)
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 69, in _call
    if asyncio.iscoroutine(result := func(*a)):
                                     ^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 29, in get_data
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/source.py", line 90, in from_CMTSOLUTION_file
    with open(filename, "rt") as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/C091795C/meta/init_model.cmt'


Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 320, in _exec_task
    await result
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 242, in mpiexec
    raise err
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpiexec.py", line 214, in mpiexec
    raise RuntimeError(d.read(f'{fname}.error'))
RuntimeError: Traceback (most recent call last):
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 94, in <module>
    _call(0, 0)
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 69, in _call
    if asyncio.iscoroutine(result := func(*a)):
                                     ^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 29, in get_data
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/source.py", line 90, in from_CMTSOLUTION_file
    with open(filename, "rt") as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/C091797B/meta/init_model.cmt'


shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
mkdir: cannot create directory ‘/gpfs/alpine’: Stale file handle
shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
mkdir: cannot create directory ‘/gpfs/alpine’: Stale file handle
shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
mkdir: cannot create directory ‘/gpfs/alpine’: Stale file handle
shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 264, in run
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/asyncio/runners.py", line 190, in run
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/asyncio/runners.py", line 118, in run
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/root.py", line 86, in execute
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 269, in execute
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 376, in _exec_children
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 269, in execute
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 371, in _exec_children
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 269, in execute
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 376, in _exec_children
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 268, in execute
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 289, in _exec_task
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/root.py", line 109, in save
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/directory.py", line 329, in dump
OSError: [Errno 116] Stale file handle
slurmstepd: error: If munged is up, restart with --num-threads=10
slurmstepd: error: Munge encode failed: Failed to access "/var/run/munge/munge.socket.2": No such file or directory
slurmstepd: error: slurm_buffers_pack_msg: auth_g_create: REQUEST_COMPLETE_BATCH_SCRIPT has authentication error
