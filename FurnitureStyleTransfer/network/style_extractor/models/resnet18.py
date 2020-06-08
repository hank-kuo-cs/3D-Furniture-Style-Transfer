import torch.nn as nn
from torchvision.models import resnet18
from ....config import config


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self._model = resnet18(pretrained=True)
        self._model.avgpool = self._make_avg_pool()
        self._model.style_features = self._make_linear()

    def forward(self, x):
        out = self._model.conv1(x)
        out = self._model.bn1(out)
        out = self._model.relu(out)
        out = self._model.maxpool(out)

        out = self._model.layer1(out)
        out = self._model.layer2(out)
        out = self._model.layer3(out)
        out = self._model.layer4(out)

        out = self._model.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self._model.style_features(out)

        return out

    @staticmethod
    def _make_linear():
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, config.style_extractor.feature_dim),
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
