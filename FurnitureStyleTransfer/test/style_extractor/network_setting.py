import os
import re
import torch
import logging
from glob import glob
from ...network.style_extractor import StyleExtractor
from ...config import config


class StyleExtractorTestSetting:
    def __init__(self, arguments):
        self._is_all_models = arguments.all
        self._epoch_of_pretrain = arguments.epoch_of_pretrain

        self.checkpoint_path = self.set_checkpoint_path()
        self.test_model_paths = self.set_test_model_paths()

        self.style_extractor = StyleExtractor()
        self.epoch_now = 0

    def set_up(self) -> (StyleExtractor, int):
        self._set_style_extractor_cuda()

        for model_path in self.test_model_paths:
            self._set_style_extractor_pretrain(model_path)
            yield self.style_extractor, self.epoch_now

    def _set_style_extractor_cuda(self):
        self._set_style_extractor_parallel()
        self._set_style_extractor_device()

    def _set_style_extractor_parallel(self):
        if config.cuda.is_parallel:
            gpu_ids = config.cuda.parallel_gpus
            self.style_extractor = torch.nn.DataParallel(self.style_extractor, device_ids=gpu_ids)

    def _set_style_extractor_device(self):
        self.style_extractor = self.style_extractor.to(config.cuda.device)

    def _set_style_extractor_pretrain(self, model_path):
        self.epoch_now = self.get_epoch_num(model_path)
        self.style_extractor.load_state_dict(torch.load(model_path))

    def set_test_model_paths(self):
        pretrain_model_paths = sorted(glob('%s/model*' % self.checkpoint_path))

        if not self._epoch_of_pretrain:
            init_model_path = pretrain_model_paths[0] if self._is_all_models else pretrain_model_paths[-1]
        else:
            init_model_path = '%s/model_epoch%.3d.pth' % (self.checkpoint_path, int(self._epoch_of_pretrain))

        if self._is_all_models:
            idx = pretrain_model_paths.index(init_model_path)
            test_model_paths = pretrain_model_paths[idx:]
        else:
            test_model_paths = [init_model_path]

        if not test_model_paths:
            raise FileNotFoundError('Cannot find pretrained weight of model.')

        return test_model_paths

    @staticmethod
    def get_epoch_num(model_path: str):
        assert isinstance(model_path, str)

        epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
        if epoch_num_str:
            return int(epoch_num_str[0])
        raise ValueError('Cannot find epoch number in the model path: %s' % model_path)

    @staticmethod
    def set_checkpoint_path():
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        return os.path.join(dir_path, '../../checkpoint/style_extractor')
