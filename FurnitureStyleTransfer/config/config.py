import os
from .cuda import CudaConfig
from .dataset import DatasetConfig
from .style_extractor import StyleExtractorConfig
from .tensorboard import TensorboardConfig


# If you want to use cpu or parallel gpus, please comment below code.
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Config:
    def __init__(self):
        self.cuda = CudaConfig(device='cpu',
                               is_parallel=False,
                               parallel_gpus=[0, 2, 3])

        self.dataset = DatasetConfig(shape_net_dataset_path='/Users/hank/Desktop/Shape-Style-Transfer/ShapeNet',
                                     furniture_images_dataset_path='/Users/hank/Desktop/Shape-Style-Transfer/FurnitureImageDataset',
                                     crowdsource_dataset_path='/Users/hank/Desktop/Shape-Style-Transfer/Crowdsource',
                                     triplet_train_dataset_num=13788,
                                     triplet_test_dataset_num=3685,
                                     style_transfer_train_dataset_num=0,
                                     style_transfer_test_dataset_num=0)

        self.style_extractor = StyleExtractorConfig(network_model='VGG19',
                                                    feature_margin=0.2,
                                                    feature_dim=6,
                                                    batch_size=2,
                                                    epoch_num=100,
                                                    lr=0.01,
                                                    momentum=0.9,
                                                    weight_decay=0.001)

        self.tensorboard = TensorboardConfig(tensorboard_path='/Users/hank/Desktop/Tensorboard',
                                             experiment_name='StyleExtractor_vgg19pretrain_lr1e-2_sgd_batch2_m0.2',
                                             loss_step=1,
                                             tsne_epoch_step=20,
                                             is_write_loss=True,
                                             is_write_tsne=False)

    def print_config(self):
        print('Config Setting:')

        data = {'cuda': self.cuda.__dict__,
                'dataset': self.dataset.__dict__,
                'style_extractor': self.style_extractor.__dict__,
                'tensorboard': self.tensorboard.__dict__}

        for k1, v1 in data.items():
            print(k1)
            for k2, v2 in v1.items():
                print('\t{0}: {1}'.format(k2, v2))
            print()
