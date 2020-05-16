import os
from glob import glob
from ..config import config
from .furniture_class import FurnitureClass


class Furniture:
    def __init__(self,
                 furniture_id: str = '',
                 furniture_class: int = 0,
                 functionality_most_like_id: str = '',
                 dataset_type: str = ''):

        self.furniture_id = furniture_id
        self.furniture_class = furniture_class
        self.functionality_most_like_id = functionality_most_like_id
        self.dataset_type = dataset_type

        self.obj_path = ''
        self.dataset_type = ''
        self.images_path = []

        self._set_data_path()

    def _set_data_path(self):
        if not self.furniture_id:
            return

        if not self.dataset_type:
            self.dataset_type = config.dataset.get_dataset_type_of_object_id(self.furniture_id)

        shapenet_dataset_path = config.dataset.shape_net_dataset_path
        class_str = FurnitureClass.class_names[self.furniture_class]
        self.obj_path = os.path.join(shapenet_dataset_path, class_str, self.furniture_id, 'model.obj')

        furniture_images_dataset_path = config.dataset.furniture_images_dataset_path
        render_images_path = os.path.join(furniture_images_dataset_path, self.dataset_type, class_str, '3dw', self.furniture_id)
        self.images_path = sorted(glob(render_images_path + '/*.png'))[:15]
