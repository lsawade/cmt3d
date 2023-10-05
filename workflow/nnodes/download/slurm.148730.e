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
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/ioi/functions/get_data.py", line 64, in get_data
    cmt3d.download_waveforms_to_storage(
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/download_waveforms_to_storage.py", line 93, in download_waveforms_to_storage
    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
TypeError: obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download() got multiple values for keyword argument 'threads_per_client'


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
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/ioi/functions/get_data.py", line 64, in get_data
    cmt3d.download_waveforms_to_storage(
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/download_waveforms_to_storage.py", line 93, in download_waveforms_to_storage
    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
TypeError: obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download() got multiple values for keyword argument 'threads_per_client'


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
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/ioi/functions/get_data.py", line 64, in get_data
    cmt3d.download_waveforms_to_storage(
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/download_waveforms_to_storage.py", line 93, in download_waveforms_to_storage
    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
TypeError: obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download() got multiple values for keyword argument 'threads_per_client'


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
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/ioi/functions/get_data.py", line 64, in get_data
    cmt3d.download_waveforms_to_storage(
  File "/autofs/nccs-svm1_home1/lsawade/gcmt/cmt3d/src/cmt3d/download_waveforms_to_storage.py", line 93, in download_waveforms_to_storage
    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
TypeError: obspy.clients.fdsn.mass_downloader.mass_downloader.MassDownloader.download() got multiple values for keyword argument 'threads_per_client'


