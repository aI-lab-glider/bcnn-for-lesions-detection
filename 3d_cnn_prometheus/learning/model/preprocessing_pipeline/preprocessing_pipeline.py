import os
import random

from ..constants import DatasetType
from .preprocessing_links import TransformNiftiToNpy, NormalizeImages, CreateChunks

from ..constants import IMAGES_DIR


class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.dataset_structure = self._split_indices()

    def run(self, dataset_type: DatasetType, batch_size: int = 1):
        def generator():
            steps = [TransformNiftiToNpy(self.config), NormalizeImages(self.config), CreateChunks(self.config)]
            for image_index in self.dataset_structure[dataset_type]:
                x_npy, y_npy = steps[0].run(self.config, image_index)
                x_npy_norm = steps[1].run(self.config, x_npy)
                for x_chunk, y_chunk in zip(steps[2].run(self.config, x_npy_norm), steps[2].run(self.config, y_npy)):
                    yield x_chunk, y_chunk
        return generator

    def _split_indices(self) -> dict:
        imgs_indices = [self._get_img_index(file_name) for file_name in
                        os.listdir(os.path.join(self.config['transform_nifti_to_npy']['path_from'], IMAGES_DIR))]
        random.shuffle(imgs_indices)

        parts_sum = self.config['create_folder_structure']['train_part'] + self.config['create_folder_structure']['valid_part'] + self.config['create_folder_structure']['test_part']

        valid_part = int(self.config['create_folder_structure']['valid_part'] * len(imgs_indices) / parts_sum)
        valid_indices_len = valid_part if valid_part > 0 else 1

        test_part = int(self.config['create_folder_structure']['test_part'] * len(imgs_indices) / parts_sum)
        test_indices_len = test_part if test_part > 0 else 1

        train_indices_len = len(imgs_indices) - valid_indices_len - test_indices_len

        return {
            DatasetType.TRAIN_DIR: imgs_indices[:train_indices_len],
            DatasetType.VALID_DIR: imgs_indices[train_indices_len: train_indices_len + valid_indices_len],
            DatasetType.TEST_DIR: imgs_indices[train_indices_len + valid_indices_len:]
        }

    @staticmethod
    def _get_img_index(img_file_name):
        return img_file_name.split('.')[0].split('_')[1]
