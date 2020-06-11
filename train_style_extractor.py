import os
import logging
from FurnitureStyleTransfer.train import train_style_extractor
from FurnitureStyleTransfer.config import config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()

    if config.cuda.device == 'cuda' and not config.cuda.is_parallel:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda.cuda_num)

    train_style_extractor()
