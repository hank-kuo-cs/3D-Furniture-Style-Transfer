import os
import re
import torch
import logging
from glob import glob
from ...network.style_extractor import StyleExtractor
from ...config import config


class StyleExtractorSetting:
    def __init__(self, arguments):
        self._is_scratch = arguments.scratch
        self._epoch_of_pretrain = arguments.epoch_of_pretrain
        self.style_extractor = StyleExtractor()

    def set_up(self) -> StyleExtractor:
        self._set_style_extractor()
        return self.style_extractor

    def _set_style_extractor(self):
        self._set_style_extractor_parallel()
        self._set_style_extractor_device()
        self._set_style_extractor_pretrain()

    def _set_style_extractor_parallel(self):
        if config.cuda.is_parallel:
            gpu_ids = config.cuda.parallel_gpus
            self.style_extractor = torch.nn.DataParallel(self.style_extractor, device_ids=gpu_ids)

    def _set_style_extractor_device(self):
        self.style_extractor = self.style_extractor.to(config.cuda.device)

    def _set_style_extractor_pretrain(self):
        if self._epoch_of_pretrain and self._is_scratch:
            raise ValueError('Cannot use both argument \'pretrain_model\' and \'scratch\'!')
        model_path = self.get_pretrain_model_path()
        if model_path and not self._is_scratch:
            self.style_extractor.init_epoch = self.get_epoch_num(model_path) + 1
            self.style_extractor.load_state_dict(torch.load(model_path))
            logging.info('Use pretrained model %s to continue training' % model_path)
        else:
            logging.info('Train from scratch')

    def get_pretrain_model_path(self):
        if not self._epoch_of_pretrain:
            pretrain_model_paths = glob('%s/model*' % self.checkpoint_path)
            model_path = sorted(pretrain_model_paths)[-1] if pretrain_model_paths else None
        else:
            model_path = '%s/model_epoch%.3d.pth' % (self.checkpoint_path, int(self._epoch_of_pretrain))

        return model_path

    @staticmethod
    def get_epoch_num(model_path: str):
        assert isinstance(model_path, str)

        epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
        if epoch_num_str:
            return int(epoch_num_str[0])
        raise ValueError('Cannot find epoch number in the model path: %s' % model_path)

    @property
    def checkpoint_path(self):
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        return os.path.join(dir_path, '../../checkpoint/style_extractor/')
