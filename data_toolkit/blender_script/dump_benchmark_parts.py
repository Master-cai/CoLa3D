"""Bake benchmark GT parts with the original Blender import and normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_mesh import init_scene, normalize_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if bpy.app.version[:2] != (4, 5):
        raise RuntimeError('InstPIFu Point cache preparation requires Blender 4.5.')
    objects = json.loads(args.objects.read_text())
    init_scene()
    imported = []
    for item in objects:
        previous = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=item['mesh'], merge_vertices=True,
                                  import_shading='NORMALS', bone_heuristic='TEMPERANCE')
        meshes = [obj for obj in bpy.context.scene.objects if obj not in previous and obj.type == 'MESH']
        if not meshes:
            raise ValueError(f"GT object {item['id']} has no Blender mesh")
        imported.append((int(item['id']), meshes))
    normalize_scene()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arrays = {'original_inst_ids': np.array([item[0] for item in imported], dtype=np.int32)}
    for index, (_, meshes) in enumerate(imported):
        all_vertices, all_faces = [], []
        vertex_offset = 0
        for obj in meshes:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            bm = bmesh.new()
            bm.from_mesh(mesh)
            bm.transform(obj.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.free()
            vertices = np.array([vertex.co[:] for vertex in mesh.vertices], dtype=np.float32)
            faces = np.array([[mesh.loops[loop].vertex_index for loop in poly.loop_indices]
                              for poly in mesh.polygons], dtype=np.int64)
            if not vertices.size or not faces.size:
                raise ValueError(f'Empty GT mesh: {obj.name}')
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(vertices)
            evaluated.to_mesh_clear()
        arrays[f'vertices_{index}'] = np.concatenate(all_vertices)
        arrays[f'faces_{index}'] = np.concatenate(all_faces)
    np.savez_compressed(args.output, **arrays)


if __name__ == '__main__':
    main()
