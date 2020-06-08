import torch
import torch.nn as nn
from .models import VGG19
from ..latent_code import LatentCode
from ...config import config


class MultiViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = self._set_model()

    def forward(self, images: list) -> LatentCode:
        batch_size = images[0].size(0)
        latent_dim = config.multiview_encoder.latent_dim

        latent_code = torch.zeros((batch_size, latent_dim), dtype=torch.float).to(config.cuda.device)

        for img in images:
            latent_one_image = self._model(img.to(config.cuda.device))
            latent_code += latent_one_image

        latent_code = LatentCode(latent_code)

        return latent_code

    @staticmethod
    def _set_model():
        return {
            'VGG19': VGG19
        }[config.multiview_encoder.network_model]()

