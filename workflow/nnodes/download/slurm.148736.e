
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
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 298, in _exec_task
    task = parse_import(task)
           ^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/nnodes/node.py", line 25, in parse_import
    target = import_module(path[0])
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1126, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1126, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/__init__.py", line 8, in <module>
    from .functions.get_data import get_data
  File "/ccs/home/lsawade/dtn_miniconda3/envs/gf-dtn/lib/python3.11/site-packages/cmt3d/ioi/functions/get_data.py", line 22, in <module>
    @retry(5, 10.0)
     ^^^^^
NameError: name 'retry' is not defined

