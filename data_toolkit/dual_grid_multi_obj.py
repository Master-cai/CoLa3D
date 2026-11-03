import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import importlib
import argparse
import json
import pandas as pd
import numpy as np
import torch
import pickle
import o_voxel
from easydict import EasyDict as edict
from functools import partial

def process_single_pickle(pickle_path, output_dir, resolution, force=False):
    meta_path = os.path.join(output_dir, 'meta.json')
    if not force and os.path.exists(meta_path):
        print(f'Already processed: {output_dir}')
        return
    print(f'Loading {pickle_path}...')
    with open(pickle_path, 'rb') as f:
        dump = pickle.load(f)
    valid_objects = []
    all_vertices = []
    for i, obj in enumerate(dump['objects']):
        if obj['vertices'].size == 0 or obj['faces'].size == 0:
            continue
        valid_objects.append(obj)
        all_vertices.append(obj['vertices'])
    if len(valid_objects) == 0:
        print(f'No valid objects in {pickle_path}')
        return
    all_vertices_concat = np.concatenate(all_vertices, axis=0)
    all_vertices_tensor = torch.from_numpy(all_vertices_concat).float()
    vertices_min = all_vertices_tensor.min(dim=0)[0]
    vertices_max = all_vertices_tensor.max(dim=0)[0]
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    print(f'Global normalization: center={center.tolist()}, scale={scale.item():.6f}')
    print(f'Processing {len(valid_objects)} objects at resolution {resolution}...')
    os.makedirs(output_dir, exist_ok=True)
    object_infos = []
    for i, obj in enumerate(valid_objects):
        vertices = torch.from_numpy(obj['vertices']).float()
        vertices = (vertices - center) * scale
        faces = torch.from_numpy(obj['faces']).long()
        try:
            voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(vertices, faces, grid_size=resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], face_weight=1.0, boundary_weight=0.2, regularization_weight=0.01, timing=False)
        except Exception as e:
            print(f'  Object {i}: Failed to convert - {e}')
            continue
        if len(voxel_indices) == 0:
            print(f'  Object {i}: No voxels generated (object too small?)')
            continue
        dual_vertices = dual_vertices * resolution - voxel_indices
        dual_vertices = torch.clamp(dual_vertices, 0, 1)
        dual_vertices = (dual_vertices * 255).type(torch.uint8)
        intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
        slot = len(object_infos)
        instance_id = torch.full((len(voxel_indices), 1), slot, dtype=torch.uint8)
        obj_path = os.path.join(output_dir, f'obj_{slot}.vxz')
        o_voxel.io.write_vxz(obj_path, voxel_indices, {'vertices': dual_vertices, 'intersected': intersected, 'instance_id': instance_id})
        object_infos.append({'index': slot, **({'original_object_id': int(obj['original_object_id'])} if 'original_object_id' in obj else {}), 'num_voxels': len(voxel_indices), 'file': f'obj_{slot}.vxz'})
        print(f'  Object {i}: {len(voxel_indices)} voxels -> {obj_path}')
    meta = {'num_objects': len(object_infos), 'objects': object_infos, 'resolution': resolution, 'center': center.tolist(), 'scale': float(scale), 'source_pickle': os.path.basename(pickle_path)}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved metadata to {meta_path}')
    print(f'Done: {len(object_infos)} objects converted')

def _dual_grid_multi_obj(file, metadatum, mesh_dump_root, root, resolution, force=False):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        for res in resolution:
            output_dir = os.path.join(root, f'dual_grid_{res}', sha256)
            meta_path = os.path.join(output_dir, 'meta.json')
            if not force and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    pack[f'dual_grid_converted_{res}'] = True
                    pack[f'dual_grid_num_objects_{res}'] = meta['num_objects']
                    pack[f'dual_grid_size_{res}'] = sum((obj['num_voxels'] for obj in meta['objects']))
                    continue
                except Exception as e:
                    print(f'Error reading {meta_path}: {e}')
            pickle_path = os.path.join(mesh_dump_root, 'mesh_dumps', f'{sha256}.pickle')
            with open(pickle_path, 'rb') as f:
                dump = pickle.load(f)
            valid_objects = []
            all_vertices = []
            for obj in dump['objects']:
                if obj['vertices'].size == 0 or obj['faces'].size == 0:
                    continue
                valid_objects.append(obj)
                all_vertices.append(obj['vertices'])
            if len(valid_objects) == 0:
                pack['error'] = 'no valid objects'
                return pack
            all_vertices_concat = np.concatenate(all_vertices, axis=0)
            all_vertices_tensor = torch.from_numpy(all_vertices_concat).float()
            vertices_min = all_vertices_tensor.min(dim=0)[0]
            vertices_max = all_vertices_tensor.max(dim=0)[0]
            center = (vertices_min + vertices_max) / 2
            scale = 0.99999 / (vertices_max - vertices_min).max()
            os.makedirs(output_dir, exist_ok=True)
            object_infos = []
            total_voxels = 0
            for i, obj in enumerate(valid_objects):
                vertices = torch.from_numpy(obj['vertices']).float()
                vertices = (vertices - center) * scale
                faces = torch.from_numpy(obj['faces']).long()
                try:
                    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(vertices, faces, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], face_weight=1.0, boundary_weight=0.2, regularization_weight=0.01, timing=False)
                except Exception as e:
                    print(f'  Object {i} of {sha256}: Failed to convert - {e}')
                    continue
                if len(voxel_indices) == 0:
                    continue
                dual_vertices = dual_vertices * res - voxel_indices
                dual_vertices = torch.clamp(dual_vertices, 0, 1)
                dual_vertices = (dual_vertices * 255).type(torch.uint8)
                intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
                slot = len(object_infos)
                instance_id = torch.full((len(voxel_indices), 1), slot, dtype=torch.uint8)
                obj_path = os.path.join(output_dir, f'obj_{slot}.vxz')
                o_voxel.io.write_vxz(obj_path, voxel_indices, {'vertices': dual_vertices, 'intersected': intersected, 'instance_id': instance_id})
                object_infos.append({'index': slot, **({'original_object_id': int(obj['original_object_id'])} if 'original_object_id' in obj else {}), 'num_voxels': len(voxel_indices), 'file': f'obj_{slot}.vxz'})
                total_voxels += len(voxel_indices)
            if len(object_infos) == 0:
                pack['error'] = 'no voxels generated'
                return pack
            meta = {'num_objects': len(object_infos), 'objects': object_infos, 'resolution': res, 'center': center.tolist(), 'scale': float(scale)}
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            pack[f'dual_grid_converted_{res}'] = True
            pack[f'dual_grid_num_objects_{res}'] = len(object_infos)
            pack[f'dual_grid_size_{res}'] = total_voxels
        return pack
    except Exception as e:
        print(f'Error processing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}

def main():
    if len(sys.argv) > 1 and (not sys.argv[1].startswith('-')) and (sys.argv[1] != 'single'):
        dataset_name = sys.argv[1]
        dataset_utils = importlib.import_module(f'data_toolkit.datasets.{dataset_name}')
        parser = argparse.ArgumentParser(description='Convert mesh dumps to per-object O-Voxels (batch mode)')
        parser.add_argument('--root', type=str, required=True, help='Directory with metadata.csv')
        parser.add_argument('--mesh_dump_root', type=str, default=None, help='Directory to load mesh dumps')
        parser.add_argument('--output_root', type=str, default=None, help='Directory to save per-object O-Voxels')
        parser.add_argument('--filter_low_aesthetic_score', type=float, default=None, help='Filter objects with aesthetic score lower than this value')
        parser.add_argument('--instances', type=str, default=None, help='Instances to process (comma-separated or file path)')
        dataset_utils.add_args(parser)
        parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--resolution', type=str, default='512')
        parser.add_argument('--world_size', type=int, default=1)
        parser.add_argument('--max_workers', type=int, default=0)
        parser.add_argument('--force', action='store_true', help='Force reprocessing')
        opt = parser.parse_args(sys.argv[2:])
        opt = edict(vars(opt))
        opt.resolution = [int(x) for x in opt.resolution.split(',')]
        opt.mesh_dump_root = opt.mesh_dump_root or opt.root
        opt.output_root = opt.output_root or opt.root
        for res in opt.resolution:
            os.makedirs(os.path.join(opt.output_root, f'dual_grid_{res}', 'new_records'), exist_ok=True)
        if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
            raise ValueError('metadata.csv not found')
        metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
        if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
            metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')).set_index('sha256'))
        if os.path.exists(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')):
            metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.mesh_dump_root, 'mesh_dumps', 'metadata.csv')).set_index('sha256'))
        for res in opt.resolution:
            dual_grid_meta_path = os.path.join(opt.output_root, f'dual_grid_{res}', 'metadata.csv')
            if os.path.exists(dual_grid_meta_path):
                dual_grid_metadata = pd.read_csv(dual_grid_meta_path).set_index('sha256')
                rename_map = {'dual_grid_converted': f'dual_grid_converted_{res}', 'dual_grid_num_objects': f'dual_grid_num_objects_{res}', 'dual_grid_size': f'dual_grid_size_{res}'}
                dual_grid_metadata = dual_grid_metadata.rename(columns=rename_map)
                metadata = metadata.combine_first(dual_grid_metadata)
        metadata = metadata.reset_index()
        if opt.instances is None:
            if opt.filter_low_aesthetic_score is not None:
                metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
            metadata = metadata[metadata['mesh_dumped'] == True]
            if not opt.force:
                mask = np.zeros(len(metadata), dtype=bool)
                for res in opt.resolution:
                    col = f'dual_grid_converted_{res}'
                    if col in metadata.columns:
                        mask |= metadata[col] != True
                    else:
                        output_dirs = [os.path.join(opt.output_root, f'dual_grid_{res}', sha256, 'meta.json') for sha256 in metadata['sha256']]
                        mask |= ~np.array([os.path.exists(p) for p in output_dirs])
                metadata = metadata[mask]
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
        print(f'Processing {len(metadata)} objects...')
        func = partial(_dual_grid_multi_obj, root=opt.output_root, mesh_dump_root=opt.mesh_dump_root, resolution=opt.resolution, force=opt.force)
        results = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Dual griding (multi-obj)')
        if 'error' in results.columns:
            errors = results[results['error'].notna()]
            if len(errors) > 0:
                with open('errors_multi_obj.txt', 'w') as f:
                    f.write('\n'.join(errors['sha256'].tolist()))
        for res in opt.resolution:
            col = f'dual_grid_converted_{res}'
            if col in results.columns:
                converted = results[results[col] == True]
                if len(converted) > 0:
                    save_df = converted[['sha256']].copy()
                    save_df['dual_grid_converted'] = True
                    if f'dual_grid_num_objects_{res}' in converted.columns:
                        save_df['dual_grid_num_objects'] = converted[f'dual_grid_num_objects_{res}']
                    if f'dual_grid_size_{res}' in converted.columns:
                        save_df['dual_grid_size'] = converted[f'dual_grid_size_{res}']
                    save_df.to_csv(os.path.join(opt.output_root, f'dual_grid_{res}', 'new_records', f'part_{opt.rank}.csv'), index=False)
        print('Done!')
    else:
        parser = argparse.ArgumentParser(description='Convert a single mesh pickle to per-object O-Voxels')
        parser.add_argument('mode', nargs='?', default='single', help='Mode (single)')
        parser.add_argument('input', type=str, help='Input pickle file from dump_mesh.py')
        parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: ovoxel-test/<basename>)')
        parser.add_argument('--resolution', type=int, default=512, help='Grid resolution')
        parser.add_argument('--force', action='store_true', help='Force reprocessing')
        args = parser.parse_args()
        if args.output_dir is None:
            basename = os.path.basename(args.input).replace('.pickle', '')
            args.output_dir = os.path.join('ovoxel-test/multi', basename)
        process_single_pickle(args.input, args.output_dir, args.resolution, args.force)
if __name__ == '__main__':
    main()
