import argparse
import logging
from torch.utils.data import DataLoader
from .test import Testing
from .network_setting import StyleExtractorTestSetting
from ...data import TripletFurnitureDataset
from ...config import config


def test_style_extractor():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_of_pretrain', type=str,
                        help='Use a model with assigned epoch in checkpoint directory to test')
    parser.add_argument('-a', '--all', action='store_true',
                        help='test model with all epochs and record the result on tensorboard')
    arguments = parser.parse_args()

    logging.info('Loading dataset...')

    test_triplet_dataset = TripletFurnitureDataset('test')

    test_triplet_dataloader = DataLoader(dataset=test_triplet_dataset,
                                         batch_size=config.style_extractor.batch_size,
                                         shuffle=True,
                                         num_workers=2)

    Testing(arguments, test_triplet_dataloader).run()
