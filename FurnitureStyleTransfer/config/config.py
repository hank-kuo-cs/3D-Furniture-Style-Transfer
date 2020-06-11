import os
from .cuda import CudaConfig
from .dataset import DatasetConfig
from .style_extractor import StyleExtractorConfig
from .multiview_encoder import MultiViewEncoderConfig
from .mesh_decoder import MeshDecoderConfig
from .loss import LossConfig
from .tensorboard import TensorboardConfig


# If you want to use cpu or parallel gpus, please comment below code.
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Config:
    def __init__(self):
        self.cuda = CudaConfig(device='cuda',
                               cuda_num=2,
                               is_parallel=False,
                               parallel_gpus=[0, 2, 3])

        self.dataset = DatasetConfig(shape_net_dataset_path='/data/hank/Shape-Style-Transfer/ShapeNet',
                                     furniture_images_dataset_path='/data/hank/Shape-Style-Transfer/FurnitureImageDataset',
                                     crowdsource_dataset_path='/data/hank/Shape-Style-Transfer/Crowdsource',
                                     triplet_train_dataset_num=13788,
                                     triplet_test_dataset_num=3685,
                                     style_transfer_train_dataset_num=0,
                                     style_transfer_test_dataset_num=0)

        self.style_extractor = StyleExtractorConfig(network_model='ResNet18',
                                                    feature_margin=0.2,
                                                    feature_dim=2,
                                                    batch_size=8,
                                                    epoch_num=100,
                                                    optimizer='Adam',
                                                    lr=0.01,
                                                    momentum=0.9,
                                                    weight_decay=0.001)

        self.multiview_encoder = MultiViewEncoderConfig(network_model='ResNet18',
                                                        latent_dim=8,
                                                        batch_size=8,
                                                        epoch_num=100,
                                                        optimizer='Adam',
                                                        lr=0.01,
                                                        momentum=0.9,
                                                        weight_decay=0.001)

        self.mesh_decoder = MeshDecoderConfig(network_model='FC',
                                              vertex_num=386,
                                              batch_size=8,
                                              epoch_num=100,
                                              optimizer='Adam',
                                              lr=0.01,
                                              momentum=0.9,
                                              weight_decay=0.001)

        self.loss = LossConfig(l_style=1.0,
                               l_img_compare=2.0,
                               multiview_loss_func='L1')

        self.tensorboard = TensorboardConfig(tensorboard_path='/home/hank/Shape-Style-Transfer/Tensorboard',
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
                'multiview_encoder': self.multiview_encoder.__dict__,
                'loss': self.loss.__dict__,
                'tensorboard': self.tensorboard.__dict__}

        for k1, v1 in data.items():
            print(k1)
            for k2, v2 in v1.items():
                print('\t{0}: {1}'.format(k2, v2))
            print()
