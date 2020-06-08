import torch
import torch.nn as nn
from ..latent_code import LatentCode
from ...config import config


class MeshDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    # ToDo: Decoder offset and color of vertices to reconstruct the mesh
    def forward(self, latent_code: LatentCode) -> (torch.Tensor, torch.Tensor):
        vertices_offset, vertices_color = None, None

        return vertices_offset, vertices_color
