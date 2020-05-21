import logging
from FurnitureStyleTransfer.test import test_style_extractor
from FurnitureStyleTransfer.config import config

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()
    test_style_extractor()
