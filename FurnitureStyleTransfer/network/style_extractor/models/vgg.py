import torch.nn as nn
from torchvision.models import vgg19_bn
from ....config import config


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self._model = vgg19_bn(pretrained=True)
        self._model.avgpool = self._make_avg_pool()
        self._model.style_features = self._make_linear()

    def forward(self, x):
        out = self._model.features(x)
        out = self._model.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self._model.style_features(out)

        return out

    @staticmethod
    def _make_linear():
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, config.style_extractor.feature_dim),
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
