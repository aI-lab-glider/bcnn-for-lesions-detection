from typing import Dict
from preprocessing_pipeline.preprocessing_links.chain_link import ChainLink
import os
import numpy as np
import nibabel as nib


class TransformNiftiToNpy(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = getattr(global_config, 'transform_nifti_to_npy', None)
        if self.is_activated(link_config):
            for operation in link_config["operations"]:
                self._transform_nifti_to_npy(
                    operation['path_from'], operation['path_to'])

    def _transform_nifti_to_npy(self, path_from: str, path_to: str):
        for nifti_file_name in os.listdir(path_from):
            nifti_path = os.path.join(path_from, nifti_file_name)
            if 'nii' in nifti_path:
                nifti = nib.load(nifti_path)
                npy = nifti.get_fdata()
                npy_path = os.path.join(
                    path_to, nifti_file_name.split('.')[0] + '.npy')
                with open(npy_path, 'w') as f:
                    np.save(f, npy)
