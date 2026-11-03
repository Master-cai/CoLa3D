from pathlib import Path
import os
import shutil
import copy
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
import tempfile
import errno
import time
BLENDER_PATH = os.environ.get('BLENDER_PATH') or shutil.which('blender')

def _resolve_tmp_root(root: str) -> str:
    path = Path(root) / '.processing_tmp'
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def _cleanup_tmp_dir(tmp_dir: str, retries: int=6, delay_sec: float=0.5):
    for attempt in range(retries):
        try:
            shutil.rmtree(tmp_dir)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            filename = os.path.basename(e.filename) if e.filename else ''
            is_nfs_busy = e.errno == errno.EBUSY or filename.startswith('.nfs')
            if is_nfs_busy and attempt < retries - 1:
                time.sleep(delay_sec)
                continue
            if is_nfs_busy:
                print(f'[WARN] Skip tmp cleanup for {tmp_dir}: {e}')
                return
            raise

def _check_blender() -> None:
    if not BLENDER_PATH or not Path(BLENDER_PATH).is_file():
        raise FileNotFoundError('Install Blender and set BLENDER_PATH to its executable.')

def _dump_mesh(file_path, metadatum, root, data_root=None):
    sha256 = metadatum['sha256']
    tmp_root = _resolve_tmp_root(root)
    tmp_dir = tempfile.mkdtemp(prefix='trellis_mesh_', dir=tmp_root)
    try:
        temp_path = os.path.join(tmp_dir, f'{sha256}.pickle')
        output_path = os.path.join(root, 'mesh_dumps', f'{sha256}.pickle')
        args = [BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'dump_mesh.py'), '--', '--object', os.path.expanduser(file_path), '--output_path', os.path.expanduser(temp_path)]
        file_identifier = metadatum.get('file_identifier')
        if isinstance(file_identifier, str):
            mapping_path = Path(data_root or root) / 'raw' / 'object_mapping' / f'{file_identifier}.json'
            if mapping_path.is_file():
                args.extend(['--object_mapping', str(mapping_path)])
        if file_path.endswith('.blend'):
            args.insert(1, file_path)
        call(args, stdout=DEVNULL, stderr=DEVNULL)
        if os.path.exists(temp_path):
            shutil.move(temp_path, output_path)
            return {'sha256': sha256, 'mesh_dumped': True}
        elif os.path.exists(temp_path + '_error.txt'):
            with open(temp_path + '_error.txt', 'r') as f:
                error_msg = f.read()
            raise ValueError(f'Failed to dump mesh. File {file_path}. Error message: {error_msg}')
        else:
            raise ValueError(f'Failed to dump mesh. File {file_path}.')
    finally:
        _cleanup_tmp_dir(tmp_dir)
if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'data_toolkit.datasets.{sys.argv[1]}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None, help='Directory to save the downloaded files')
    parser.add_argument('--mesh_dump_root', type=str, default=None, help='Directory to save the mesh dumps')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None, help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None, help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.download_root = opt.download_root or opt.root
    opt.mesh_dump_root = opt.mesh_dump_root or opt.root
    os.makedirs(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'new_records'), exist_ok=True)
    print('Checking blender...', flush=True)
    _check_blender()
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'mesh_dumped' in metadata.columns:
            metadata = metadata[metadata['mesh_dumped'] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []
    sha256_list = os.listdir(os.path.join(opt.mesh_dump_root, 'mesh_dumps'))
    sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith('.pickle')]
    for sha256 in sha256_list:
        records.append({'sha256': sha256, 'mesh_dumped': True})
    print(f'Found {len(sha256_list)} dumped mesh')
    metadata = metadata[~metadata['sha256'].isin(sha256_list)]
    print(f'Processing {len(metadata)} objects...')
    func = partial(_dump_mesh, root=opt.mesh_dump_root, data_root=opt.download_root)
    mesh_dumped = dataset_utils.foreach_instance(metadata, opt.download_root, func, max_workers=opt.max_workers, desc='Dumping mesh')
    mesh_dumped = pd.concat([mesh_dumped, pd.DataFrame.from_records(records)])
    mesh_dumped.to_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'new_records', f'part_{opt.rank}.csv'), index=False)
