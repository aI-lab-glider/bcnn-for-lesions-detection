from typing import Callable, List
from batchgenerators.augmentations.color_augmentations import (
    augment_contrast, augment_brightness_multiplicative, augment_gamma)

from batchgenerators.augmentations.noise_augmentations import (augment_rician_noise,
                                                               augment_gaussian_noise)
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file, save_as_nifti
import numpy as np
import nibabel as nib


def create_augmentations() -> List[Callable[[np.ndarray], np.ndarray]]:
    def wrap_augmentation(augmentation_function):
        def l(data): return augmentation_function(data.astype('float64')).astype('int16')
        l.__name__ = augmentation_function.__name__
        return l
    return list(map(wrap_augmentation, [
        augment_contrast,
        augment_brightness_multiplicative,
        augment_gamma,
        augment_rician_noise,
        augment_gaussian_noise,
        augment_linear_downsampling_scipy,
    ]))


if __name__ == '__main__':
    save_to = Paths.PROJECT_DIR.parent / 'experiments' / 'augmentation_tests'
    image_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format('0003', 'nii.gz')
    image = nib.load(image_path)

    for augmentation in create_augmentations():
        print('Runnig for ', augmentation.__name__)
        augmented_image = augmentation(image.get_fdata()[None, ...])
        save_as_nifti(np.squeeze(augmented_image, axis=0), save_to/augmentation.__name__, image.affine, image.header)
