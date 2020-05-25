
import json
import os
import cv2
import re
import numpy as np
from glob import glob
from tqdm import tqdm
from FurnitureStyleTransfer.object import FurnitureClass

image_dataset_path = '/data/hank/Shape-Style-Transfer/FurnitureImageDataset'
shapenet_dataset_path = '/data/hank/Shape-Style-Transfer/ShapeNet'

# image_dataset_path = '/Users/hank/Desktop/Shape-Style-Transfer/FurnitureImageDataset'
# shapenet_dataset_path = '/Users/hank/Desktop/Shape-Style-Transfer/ShapeNet'

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


def get_all_object_ids_of_one_class(class_name):
    train_id_paths = glob(os.path.join(image_dataset_path, 'train', class_name, '3dw', '*'))
    train_ids = [get_object_id_from_path(class_name, train_id_path) for train_id_path in train_id_paths]

    test_id_paths = glob(os.path.join(image_dataset_path, 'test', class_name, '3dw', '*'))
    test_ids = [get_object_id_from_path(class_name, test_id_path) for test_id_path in test_id_paths]

    return train_ids + test_ids


def get_object_id_from_path(class_name: str, id_path: str):
    return re.findall(r't.+/%s/3dw/([a-z0-9]+)' % class_name, id_path)[0]


if __name__ == '__main__':
    class_names = FurnitureClass.class_names

    for class_name in class_names:
        print('\nComparing similarity of "%s" dataset' % class_name)
        object_ids = get_all_object_ids_of_one_class(class_name)

        for object1_id in tqdm(object_ids):
            best_loss = -1

            for object2_id in object_ids:
                if object1_id == object2_id:
                    continue

                similarity_loss = get_similarity_loss_two_objects(class_name, object1_id, object2_id)

                if similarity_loss < best_loss or best_loss < 0:
                    best_loss = similarity_loss
                    add_data_to_json(object1_id, object2_id)

        save_json(class_name)
