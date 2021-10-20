from typing import Dict
from .chain_link import ChainLink
import os
import numpy as np
import nibabel as nib


class TransformNiftiToNpy(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('transform_nifti_to_npy', None)
        if self.is_activated(link_config):
            for operation in link_config["operations"]:
                self._transform_nifti_to_npy(
                    operation["amount"], operation['path_from'], operation['path_to'])

    def _transform_nifti_to_npy(self, amount: int, path_from: str, path_to: str):
        transformed = 0 
        for nifti_file_name in sorted(os.listdir(path_from)):
            if transformed == amount:
                break
            nifti_path = os.path.join(path_from, nifti_file_name)
            if not os.path.isdir(path_to):
                os.makedirs(path_to)

            if 'nii' in nifti_path:
                nifti = nib.load(nifti_path)
                npy = nifti.get_fdata()
                npy_path = os.path.join(
                    path_to, nifti_file_name.split('.')[0] + '.npy')
                with open(npy_path, 'wb+') as f:
                    np.save(f, npy)
                    transformed += 1
