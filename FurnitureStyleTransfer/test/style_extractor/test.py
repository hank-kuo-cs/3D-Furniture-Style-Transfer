import torch
import logging
from tqdm import tqdm
from .evaluate import get_correct_num
from .network_setting import StyleExtractorTestSetting
from ...visualize import StyleExtractorLogger
from ...config import config


class Testing:
    def __init__(self, args, test_dataloader):
        self.is_all_model = args.all
        self.test_dataloader = test_dataloader
        self.setting = StyleExtractorTestSetting(arguments=args)
        self.logger = StyleExtractorLogger()

        self.style_extractor = None
        self.epoch_now = 0

    def run(self):
        for style_extractor, epoch_now in self.setting.set_up():
            self.style_extractor, self.epoch_now = style_extractor, epoch_now
            logging.info('Start training epoch %d' % epoch_now)
            self.test_one_epoch()

    def test_one_epoch(self):
        self.style_extractor.eval()

        correct_num = 0
        with torch.no_grad():
            for idx, (sample_images, positive_images, negative_images) in tqdm(enumerate(self.test_dataloader)):
                sample_features = self.style_extractor(sample_images)
                positive_features = self.style_extractor(positive_images)
                negative_features = self.style_extractor(negative_images)

                correct_num += get_correct_num(sample_features, positive_features, negative_features)

        dataset_num = config.dataset.triplet_test_dataset_num
        accuracy = correct_num / dataset_num * 100

        logging.info('epoch {}, accuracy = {:.3f}%\n'.format(self.epoch_now, accuracy))

        if self.is_all_model:
            self.record_logger(accuracy)

    def record_logger(self, accuracy):
        self.logger.epoch_now = self.epoch_now
        self.logger.record_test_error(accuracy)
