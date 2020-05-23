
import json
import os
import cv2
import re
import numpy as np
from glob import glob
from tqdm import tqdm
from FurnitureStyleTransfer.object import FurnitureClass

shapenet_dataset_path = '/data/hank/Shape-Style-Transfer/ShapeNet'
similar_json = {}
os.makedirs('ModelSimilarity', exist_ok=True)


def add_data_to_json(object_id: str, similar_object_id):
    """
    format: {'object_id': 'object_id', 'object_id': 'object_id'},
    :return:
    """
    assert isinstance(object_id, str)
    assert isinstance(similar_object_id, str)

    similar_json[object_id] = similar_object_id


def save_json(class_name: str):
    global similar_json

    data = json.dumps(similar_json)
    with open('ModelSimilarity/%s.json' % class_name, 'w') as f:
        f.write(data)
    f.close()
    similar_json.clear()


def get_mask_l1_loss(mask1, mask2):
    return np.linalg.norm(mask1 - mask2)


def get_similarity_loss_two_objects(class_type: str, obj1_id: str, obj2_id: str):
    assert isinstance(class_type, str)
    assert isinstance(obj1_id, str)
    assert isinstance(obj2_id, str)

    obj1_masks_path = sorted(glob(os.path.join(shapenet_dataset_path, class_type, obj1_id, 'mask_images/*')))
    obj2_masks_path = sorted(glob(os.path.join(shapenet_dataset_path, class_type, obj2_id, 'mask_images/*')))

    loss = 0.0

    for mask1, mask2 in zip(obj1_masks_path, obj2_masks_path):

        mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)

        loss += get_mask_l1_loss(mask1, mask2)

    return loss


def get_object_id_from_path(object_dir_path):
    return re.findall(r'ShapeNet/.+/([a-z0-9]+)', object_dir_path)[0]


if __name__ == '__main__':
    class_names = FurnitureClass.class_names

    for class_name in class_names:
        print('\nComparing similarity of "%s" dataset' % class_name)
        objects_of_one_class_dataset_path = glob(os.path.join(shapenet_dataset_path, class_name, '*'))

        for object1_dir_path in tqdm(objects_of_one_class_dataset_path):
            best_loss = -1

            for object2_dir_path in objects_of_one_class_dataset_path:
                if object1_dir_path == object2_dir_path:
                    continue

                obj1_id = get_object_id_from_path(object1_dir_path)
                obj2_id = get_object_id_from_path(object2_dir_path)

                similarity_loss = get_similarity_loss_two_objects(class_name, obj1_id, obj2_id)

                if similarity_loss < best_loss or best_loss < 0:
                    best_loss = similarity_loss
                    add_data_to_json(class_name, obj1_id, obj2_id)

        save_json(class_name)
