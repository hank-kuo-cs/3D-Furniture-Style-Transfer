import os
import torch
import logging
from torch.optim import SGD, Adam

from .network_setting import StyleExtractorTrainSetting
from ...visualize import StyleExtractorLogger
from ...network.style_extractor import TripletLoss
from ...config import config


class Training:
    def __init__(self, args, train_dataloader):
        self.train_dataloader = train_dataloader
        self.style_extractor, self.epoch_now = StyleExtractorTrainSetting(args).set_up()

        self.loss_func = TripletLoss()
        self.optimizer = self.set_optimizer()

        self.logger = StyleExtractorLogger(epoch_now=self.epoch_now)

        self.make_checkpoint_path()

    def run(self):
        init_epoch = self.epoch_now - 1

        for epoch_now in range(init_epoch, config.style_extractor.epoch_num):
            logging.info('Start training epoch %d' % (epoch_now + 1))
            self.train_one_epoch()

            self.save_model()
            self.logger.show_epoch_loss()
            self.add_epoch_now()

    def train_one_epoch(self):
        self.style_extractor.train()

        for idx, (sample_images, positive_images, negative_images) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            sample_features = self.style_extractor(sample_images)
            positive_features = self.style_extractor(positive_images)
            negative_features = self.style_extractor(negative_images)

            loss = self.loss_func(sample_features, positive_features, negative_features)

            loss.backward()
            self.optimizer.step()

            self.logger.add_step_and_loss(loss=loss.item(), step=idx + 1)

    def save_model(self):
        model_path = '%s/model_epoch%.3d.pth' % (self.checkpoint_path, self.epoch_now)
        torch.save(self.style_extractor.state_dict(), model_path)

    def add_epoch_now(self):
        self.epoch_now += 1
        self.logger.epoch_now += 1

    def make_checkpoint_path(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def set_optimizer(self):
        optimizer_type = config.style_extractor.optimizer
        optimizer_setting = {'params': self.style_extractor.parameters(),
                             'lr': config.style_extractor.lr,
                             'momentum': config.style_extractor.momentum,
                             'weight_decay': config.style_extractor.weight_decay}

        if optimizer_type != 'SGD':
            optimizer_setting.pop('momentum')

        return {
            'SGD': SGD,
            'Adam': Adam
        }[optimizer_type](**optimizer_setting)

    @property
    def checkpoint_path(self):
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        return os.path.join(dir_path, '../../checkpoint/style_extractor')
