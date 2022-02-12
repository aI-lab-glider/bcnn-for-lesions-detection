import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import random
from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.evaluation.utils import get_lungs_bounding_box_coords, load_config, load_lungs_mask, load_nifti_file, save_as_nifti


@dataclass
class EvaluationConfig:
    model_name: str
    patient_id: int


def get_patients_to_predict():
    idxs_with_lesions = [3, 4, 5, 6, 8, 10, 11, 14, 17, 21, 26, 27, 29, 30, 32, 34, 35, 36, 37, 39, 41, 42, 43, 44, 45,
                         46, 48, 49, 50, 51, 53, 54,
                         58, 59, 63, 65, 66, 67, 68, 69, 71, 73, 75, 78, 80, 81, 83, 85, 89, 93, 94, 96, 97, 98, 99,
                         101, 102, 104, 105, 106, 107, 108, 109, 111, 112, 114, 116, 118,
                         119, 120, 122, 123, 127, 128, 134, 138, 139, 141, 148, 150, 152, 155, 157, 158, 160, 164, 166,
                         167, 168,
                         171, 173, 175, 177, 178, 179, 180, 181, 182, 183, 184, 186, 188, 189, 190, 191, 192, 195, 199,
                         202, 204, 205, 206, 207, 209, 211, 212, 213, 215, 216, 218,
                         219, 223, 224, 225, 230, 231, 232, 233, 235, 236, 237, 239]
    return [3]  # , 4, 191]  # random.sample(idxs_with_lesions, k=3)


def main():
    folder = Path(sys.argv[1])
    model_config = load_config(folder / 'config.json')
    weights_path = get_latest_weights_from_folder(folder)

    results_dir = create_predictions_dir(weights_path)
    pacient_idxs = get_patients_to_predict()
    prediction_options = PredictionOptions(
        chunk_size=model_config['evaluation']['chunk_size'],
        stride=model_config['evaluation']['stride'],
        mc_sample=model_config['mc_samples']
    )
    for idx in pacient_idxs:
        make_prediction(weights_path, idx, prediction_options, results_dir)


def get_latest_weights_from_folder(folder: Path):
    return Path(max(glob.glob(f'{folder}/*.h5'), key=os.path.getctime))


def create_predictions_dir(weights_path: Path):
    weights_folder_name = weights_path.stem
    predictions_folder_name = f"{''.join(weights_folder_name)}_predictions"
    prediction_path = Path(weights_path.parent) / predictions_folder_name
    Path(prediction_path).mkdir(parents=True, exist_ok=True)
    return prediction_path


def crop_image_to_bounding_box_with_lungs(image, lungs_segmentation):
    lungs_bounding_box_coords = get_lungs_bounding_box_coords(lungs_segmentation)
    return image[lungs_bounding_box_coords]


@dataclass
class PredictionOptions:
    chunk_size: Tuple[int, int, int]
    mc_sample: int
    stride: Tuple[int, int, int]


def make_prediction(weights_path: Path, patient_idx, prediction_options: PredictionOptions, results_dir: Path):
    image_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format(
        f'{patient_idx:0>4}', 'nii.gz')
    segmentation_path = str(Paths.REFERENCE_SEGMENTATION_FILE_PATTERN_PATH).format(
        f'{patient_idx:0>4}', 'nii.gz')

    mask_path = str(Paths.MASK_FILE_PATTERN_PATH).format(f'{patient_idx:0>4}', 'nii.gz')

    nifti = nib.load(image_path)
    image = load_nifti_file(image_path)
    segmentation = load_lungs_mask(segmentation_path)
    image = crop_image_to_bounding_box_with_lungs(image, segmentation)
    cropped_segmentation = crop_image_to_bounding_box_with_lungs(segmentation, segmentation)

    model_evaluator = BayesianModelEvaluator(
        str(weights_path),
        prediction_options.chunk_size
    )

    predictions = model_evaluator.evaluate(
        image,
        cropped_segmentation,
        prediction_options.mc_sample,
        prediction_options.stride)

    mask = load_lungs_mask(mask_path)
    mask = crop_image_to_bounding_box_with_lungs(mask, segmentation)
    save_as_nifti(image, results_dir / f'LUNGS_BOUNDING_BOX_{patient_idx}', nifti.affine, nifti.header)
    save_as_nifti(mask, results_dir / f'LESION_{patient_idx}', nifti.affine, nifti.header)
    model_evaluator.save_predictions(
        results_dir,
        patient_idx,
        predictions,
        cropped_segmentation,
        nifti.affine,
        nifti.header)

if __name__ == '__main__':
    main()
