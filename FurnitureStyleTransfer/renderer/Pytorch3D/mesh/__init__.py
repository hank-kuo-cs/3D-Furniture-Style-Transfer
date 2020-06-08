import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from ....config import config


def load_mesh(obj_path):
    device = torch.device(config.cuda.device)

    if not os.path.exists(obj_path):
        raise FileNotFoundError('Cannot find moon object from \'%s\'' % obj_path)

    vertices, faces, aux = load_obj(obj_path)

    textures = load_texture(aux, faces)

    vertices = vertices.to(device)
    faces = faces.verts_idx.to(device)

    mesh = Meshes(verts=[vertices],
                  faces=[faces],
                  textures=textures)

    return mesh


def load_texture(aux, faces):
    if aux.verts_uvs is None:
        return None

    device = torch.device(config.cuda.device)

    vertices_uvs = aux.verts_uvs[None, ...].to(device)
    faces_uvs = faces.textures_idx[None, ...].to(device)

    texture_maps = aux.texture_images
    texture_maps = list(texture_maps.values())[0]
    texture_maps = texture_maps[None, ...].to(device)

    return Textures(verts_uvs=vertices_uvs,
                    faces_uvs=faces_uvs,
                    maps=texture_maps)
