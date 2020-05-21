import argparse
import logging
from torch.utils.data import DataLoader
from .train import Training
from .network_setting import StyleExtractorTrainSetting
from ...data import TripletFurnitureDataset
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
    test_triplet_dataset = TripletFurnitureDataset('test')

    train_triplet_dataloader = DataLoader(dataset=train_triplet_dataset,
                                          batch_size=config.style_extractor.batch_size,
                                          shuffle=True,
                                          num_workers=2)

    test_triplet_dataloader = DataLoader(dataset=test_triplet_dataset,
                                         batch_size=config.style_extractor.batch_size,
                                         shuffle=True,
                                         num_workers=2)

    Training(arguments, train_triplet_dataloader, test_triplet_dataloader).run()
