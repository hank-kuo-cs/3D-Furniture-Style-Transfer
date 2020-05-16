import logging
from FurnitureStyleTransfer.train import train_style_extractor
from FurnitureStyleTransfer.config import config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()
    train_style_extractor()
