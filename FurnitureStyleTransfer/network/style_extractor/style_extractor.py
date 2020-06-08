import torch
import torch.nn as nn
from .models import VGG19, ResNet18
from ...config import config


class StyleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = self._set_model()

    def forward(self, images: list) -> torch.Tensor:
        batch_size = images[0].size()[0]
        feature_dim = config.style_extractor.feature_dim

        style_features = torch.zeros((batch_size, feature_dim), dtype=torch.float).to(config.cuda.device)

        for img in images:
            features = self._model(img.to(config.cuda.device))
            style_features += features

        return style_features

    @staticmethod
    def _set_model():
        return {
            'VGG19': VGG19,
            'ResNet18': ResNet18
        }[config.style_extractor.network_model]()
