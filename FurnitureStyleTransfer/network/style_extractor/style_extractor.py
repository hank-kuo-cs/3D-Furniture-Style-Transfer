import torch
import torch.nn as nn
from .models import VGG19
from ...config import config


class StyleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_epoch = 1
        self._model = self._set_model()

    def forward(self, images):
        batch_size = config.style_extractor.batch_size
        feature_dim = config.style_extractor.feature_dim

        style_features = torch.zeros((batch_size, feature_dim), dtype=torch.float)

        for img in images:
            features = self._model(img)
            style_features += features

        print('\nstyle features')
        print(style_features.requires_grad)

        return style_features

    @staticmethod
    def _set_model():
        return {
            'VGG19': VGG19
        }[config.style_extractor.network_model]()


