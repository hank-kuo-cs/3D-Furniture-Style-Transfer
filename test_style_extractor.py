import os
import logging
from FurnitureStyleTransfer.test import test_style_extractor
from FurnitureStyleTransfer.config import config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    if config.cuda.device == 'cuda' and not config.cuda.is_parallel:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda.cuda_num)

    config.print_config()
    test_style_extractor()
