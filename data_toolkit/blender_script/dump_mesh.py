import json
from pathlib import Path
import argparse, sys, os, math, io
from typing import *
import bpy
import bmesh
from mathutils import Vector, Matrix
import numpy as np
import pickle

def init_scene() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    if Path(object_path).suffix.lower() not in {'.glb', '.gltf'}:
        raise ValueError('Scene preparation expects a GLB or glTF input.')
    bpy.ops.import_scene.gltf(filepath=object_path)

def delete_invisible_objects() -> None:
    to_remove = []
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)

def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple((min(x, y) for x, y in zip(bbox_min, coord)))
            bbox_max = tuple((max(x, y) for x, y in zip(bbox_max, coord)))
    if not found:
        raise RuntimeError('no objects in scene to compute bounding box for')
    return (Vector(bbox_min), Vector(bbox_max))

def normalize_scene() -> Tuple[float, Vector]:
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new('ParentEmpty', None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    return (scale, offset)

def main(arg):
    if arg.object.endswith('.blend'):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
    print('[INFO] Scene initialized.')
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    output = {'objects': []}
    mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']
    node_ids = None
    if arg.object_mapping:
        mapping = json.loads(Path(arg.object_mapping).read_text())
        node_ids = {}
        object_ids = set()
        for item in mapping['objects']:
            object_id = int(item['object_id'])
            if object_id < 0 or object_id in object_ids or (not item['nodes']):
                raise ValueError('Invalid object mapping IDs or empty node list')
            object_ids.add(object_id)
            for name in item['nodes']:
                if name in node_ids:
                    raise ValueError(f'Duplicate mapped node: {name}')
                node_ids[name] = object_id
        imported_names = [obj.name for obj in mesh_objects]
        if len(set(imported_names)) != len(imported_names) or set(imported_names) != set(node_ids):
            raise ValueError(f'Object mapping coverage mismatch: imported={imported_names}, mapped={list(node_ids)}')
    for original_index, obj in enumerate(mesh_objects):
        if obj.type != 'MESH':
            continue
        pack = {'vertices': None, 'faces': None}
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(eval_mesh)
        bm.free()
        pack['vertices'] = np.array([v.co[:] for v in eval_mesh.vertices], dtype=np.float32)
        pack['faces'] = np.array([[eval_mesh.loops[i].vertex_index for i in poly.loop_indices] for poly in eval_mesh.polygons], dtype=np.int32)
        pack['original_object_id'] = node_ids[obj.name] if node_ids is not None else original_index
        output['objects'].append(pack)
    if node_ids is not None:
        grouped = {}
        for pack in output['objects']:
            grouped.setdefault(pack['original_object_id'], []).append(pack)
        output['objects'] = []
        for object_id in sorted(grouped):
            parts = grouped[object_id]
            vertices = []
            faces = []
            offset = 0
            for part in parts:
                vertices.append(part['vertices'].reshape(-1, 3))
                faces.append(part['faces'].reshape(-1, 3) + offset)
                offset += len(part['vertices'])
            output['objects'].append({'original_object_id': object_id, 'vertices': np.concatenate(vertices, axis=0), 'faces': np.concatenate(faces, axis=0)})
    os.makedirs(os.path.dirname(arg.output_path), exist_ok=True)
    with open(arg.output_path, 'wb') as f:
        pickle.dump(output, f)
    print('[INFO] Output saved to {}.'.format(arg.output_path))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--object_mapping', type=str, default=None)
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_path', type=str, default='/tmp', help='The path the output will be dumped to.')
    argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_args(argv)
    main(args)
