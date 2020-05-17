import os
import torch
import argparse
import logging
from torch.optim import SGD
from torch.utils.data import DataLoader

from .network_setting import StyleExtractorSetting
from .logger import StyleExtractorLogger
from ...data import TripletFurnitureDataset
from ...network.style_extractor import TripletLoss
from ...config import config


def train_style_extractor():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_of_pretrain', type=str,
                        help='Use a pretrain model in checkpoint directory to continue training')
    parser.add_argument('-s', '--scratch', action='store_true',
                        help='Train model from scratch, do not use pretrain model')
    arguments = parser.parse_args()

    logging.info('Loading dataset...')

    train_triplet_dataset = TripletFurnitureDataset('train')

    assert len(train_triplet_dataset) == 13788

    train_triplet_dataloader = DataLoader(dataset=train_triplet_dataset,
                                          batch_size=config.style_extractor.batch_size,
                                          shuffle=True,
                                          num_workers=2)

    Training(arguments, train_triplet_dataloader).run()


class Training:
    def __init__(self, args, data_loader):
        self.data_loader = data_loader
        self.style_extractor, self.epoch_now = StyleExtractorSetting(args).set_up()

        self.loss_func = TripletLoss()
        self.optimizer = SGD(params=self.style_extractor.parameters(),
                             lr=config.style_extractor.lr,
                             momentum=config.style_extractor.momentum,
                             weight_decay=config.style_extractor.weight_decay)

        self.logger = StyleExtractorLogger(init_epoch=self.epoch_now)

        self.make_checkpoint_path()

    def run(self):
        self.style_extractor.train()

        for epoch_now in range(self.style_extractor.init_epoch - 1, config.style_extractor.epoch_num):
            logging.info('Start training epoch %d' % (epoch_now + 1))

            for idx, (sample_images, positive_images, negative_images) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                sample_features = self.style_extractor(sample_images)
                positive_features = self.style_extractor(positive_images)
                negative_features = self.style_extractor(negative_images)

                loss = self.loss_func(sample_features, positive_features, negative_features)

                loss.backward()
                self.optimizer.step()

                self.logger.add_step_and_loss(loss=loss.item(), step=idx+1)

            self.save_model()
            self.logger.show_epoch_loss()
            self.add_epoch_now()

    def save_model(self):
        model_path = '%s/model_epoch%.3d.pth' % (self.checkpoint_path, self.epoch_now)
        torch.save(self.style_extractor.state_dict(), model_path)

    def add_epoch_now(self):
        self.epoch_now += 1
        self.logger.epoch_now += 1

    def make_checkpoint_path(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @property
    def checkpoint_path(self):
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        return os.path.join(dir_path, '../../checkpoint/style_extractor/')
