import os
import re


class DatasetConfig:
    def __init__(self,
                 shape_net_dataset_path: str,
                 furniture_images_dataset_path: str,
                 crowdsource_dataset_path: str,
                 triplet_train_dataset_num: int,
                 triplet_test_dataset_num: int,
                 style_transfer_train_dataset_num: int,
                 style_transfer_test_dataset_num: int):

        self.shape_net_dataset_path = shape_net_dataset_path
        self.furniture_images_dataset_path = furniture_images_dataset_path
        self.crowdsource_dataset_path = crowdsource_dataset_path
        self.triplet_train_dataset_num = triplet_train_dataset_num
        self.triplet_test_dataset_num = triplet_test_dataset_num
        self.style_transfer_train_dataset_num = style_transfer_train_dataset_num
        self.style_transfer_test_dataset_num = style_transfer_test_dataset_num

        self.train_crowdsource_data = None
        self.test_crowdsource_data = None

        self._set_crowdsource_data()

    def _set_crowdsource_data(self):
        train_crowdsource_path = os.path.join(self.crowdsource_dataset_path, 'train.txt')
        test_crowdsource_path = os.path.join(self.crowdsource_dataset_path, 'test.txt')

        self.train_crowdsource_data = self.get_file_data(train_crowdsource_path)
        self.test_crowdsource_data = self.get_file_data(test_crowdsource_path)

    def get_dataset_type_of_object_id(self, obj_id):
        if re.findall(obj_id, self.train_crowdsource_data):
            return 'train'
        elif re.findall(obj_id, self.test_crowdsource_data):
            return 'test'
        else:
            raise ValueError('Cannot find this object id "%s" in crowdsource data.' % obj_id)

    @staticmethod
    def get_file_data(file_path):
        with open(file_path, 'r') as f:
            data = f.read()
        f.close()
        return data
