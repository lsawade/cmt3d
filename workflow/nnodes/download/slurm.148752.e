
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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 45, in _call
    (func, args, mpiarg, group_mpiarg) = root.load(f'{argv[1]}.pickle')
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/directory.py", line 293, in load
    return pickle.load(fb)
           ^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/__init__.py", line 16, in <module>
    from .functions.forward_kernel import forward_kernel
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/forward_kernel.py", line 60
    elif _mname in Constants.mt_params:
    ^^^^
IndentationError: expected an indented block after 'if' statement on line 55


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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/mpi.py", line 45, in _call
    (func, args, mpiarg, group_mpiarg) = root.load(f'{argv[1]}.pickle')
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/directory.py", line 293, in load
    return pickle.load(fb)
           ^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/__init__.py", line 16, in <module>
    from .functions.forward_kernel import forward_kernel
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/forward_kernel.py", line 60
    elif _mname in Constants.mt_params:
    ^^^^
IndentationError: expected an indented block after 'if' statement on line 55


