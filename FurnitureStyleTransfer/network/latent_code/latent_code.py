import torch
from ...config import config


class LatentCode:
    def __init__(self, latent_code: torch.Tensor):
        self._style_dim = config.style_extractor.feature_dim
        self._functionality_dim = config.multiview_encoder.latent_dim - self._style_dim
        self._batch_size = latent_code.size(0) if latent_code.ndimension() > 1 else 0
        self.is_batch = True if self._batch_size > 0 else False
        self.latent_code = latent_code

        self._check_dimensions()

    def _check_dimensions(self):
        latent_size = self.latent_code.size(1) if self.is_batch else self.latent_code.size(0)
        assert latent_size == self._style_dim + self._functionality_dim

    # ToDo: Change style features between two latent codes.
    def change_style_features(self):
        pass

    @property
    def functionality_features(self):
        if self.is_batch:
            return self.latent_code[:, :self._functionality_dim]
        else:
            return self.latent_code[:self._functionality_dim]

    @property
    def style_features(self):
        if self.is_batch:
            return self.latent_code[:, self._functionality_dim: self._functionality_dim + self._style_dim]
        else:
            return self.latent_code[self._functionality_dim: self._functionality_dim + self._style_dim]
