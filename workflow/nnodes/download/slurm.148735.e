
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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 72, in get_data
    if (len(os.listdir(waveformdir)) <= 30) \
            ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/data/B011500A/waveforms'


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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 72, in get_data
    if (len(os.listdir(waveformdir)) <= 30) \
            ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/data/B011501F/waveforms'


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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 72, in get_data
    if (len(os.listdir(waveformdir)) <= 30) \
            ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/gpfs/alpine/geo111/scratch/lsawade/gcmt/data/B011595B/waveforms'


