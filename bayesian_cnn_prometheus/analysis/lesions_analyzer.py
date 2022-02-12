import glob
import json
import os
from pathlib import Path
from typing import Tuple, List

import nibabel as nib
import numpy as np
from tqdm import tqdm

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.evaluation.evaluate_model import PredictionOptions, crop_image_to_bounding_box_with_lungs
from bayesian_cnn_prometheus.evaluation.utils import get_patient_index, load_nifti_file, load_lungs_mask, save_as_nifti


class LesionsAnalyzer:
    def __init__(self, model_path: str, input_path: str, prediction_options: PredictionOptions,
                 patients_to_analysis: List[int] = None):
        """
        Creates LesionsAnalyzer instance.
        :param model_path: path to the model in h5 form
        :param input_path: path to the directory with images, lesions and segmentations labels
        :param prediction_options: parameters for prediction
        :param patients_to_analysis: indices of patients to perform analysis on them, if None - every patient
        in the input dir will be analyzed
        """
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = os.path.join(input_path, Paths.RESULTS_DIR)
        self.prediction_options = prediction_options
        self.patients_to_analysis = patients_to_analysis
        self.model_evaluator = BayesianModelEvaluator(self.model_path, prediction_options.chunk_size)
        self.results = {'chunks_analyse': {'overall': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}},
                        'voxels_analyse': {'overall': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}}

    def run_analysis(self):
        for image_path in tqdm(glob.glob(os.path.join(self.input_path, Paths.IMAGES_DIR, '*.nii.gz'))):
            patient_idx = get_patient_index(image_path)

            if self.patients_to_analysis:
                if patient_idx not in self.patients_to_analysis:
                    continue

            segmentation_path = os.path.join(self.input_path, Paths.REFERENCE_SEGMENTATIONS_DIR,
                                             str(Paths.REFERENCE_SEGMENTATION_FILE_PATTERN)
                                             .format(f'{patient_idx:0>4}', 'nii.gz'))
            mask_path = os.path.join(self.input_path, Paths.MASKS_DIR,
                                     str(Paths.MASK_FILE_PATTERN).format(f'{patient_idx:0>4}', 'nii.gz'))
            variance_path = os.path.join(self.input_path, Paths.RESULTS_DIR,
                                         str(Paths.VARIANCE_FILE_PATTERN).format(f'{patient_idx:0>4}', 'nii.gz'))

            nifti = nib.load(image_path)
            image = nifti.get_fdata()
            segmentation = load_lungs_mask(segmentation_path)
            mask = load_lungs_mask(mask_path)

            cropped_image = crop_image_to_bounding_box_with_lungs(image, segmentation)
            cropped_segmentation = crop_image_to_bounding_box_with_lungs(segmentation, segmentation)
            cropped_mask = crop_image_to_bounding_box_with_lungs(mask, segmentation)

            if os.path.exists(variance_path):
                variance = load_nifti_file(variance_path)
            else:
                variance = self.get_variance(cropped_image, cropped_segmentation, variance_path, nifti.affine,
                                             nifti.header)

            self._analyze_chunks(patient_idx, cropped_mask, variance)
            self._analyze_voxels(patient_idx, cropped_mask, variance)

        self._summarize_results()
        self._save_results()

    def _analyze_chunks(self, patient_idx, mask: np.ndarray, variance: np.ndarray):
        self.results['chunks_analyse'][patient_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        origin_x, origin_y, origin_z = mask.shape
        stride_x, stride_y, stride_z = self.prediction_options.stride

        for x in range(0, origin_x, stride_x)[:-1]:
            for y in range(0, origin_y, stride_y)[:-1]:
                for z in range(0, origin_z, stride_z)[:-1]:
                    mask_chunk = mask[self._get_window((x, y, z))]
                    mask_mean = np.mean(mask_chunk)

                    variance_chunk = variance[self._get_window((x, y, z))]
                    variance_mean = np.mean(variance_chunk)

                    self._update_results(patient_idx, mask_mean, variance_mean, 'chunks_analyse')

    def _get_window(self, coord: Tuple[int, int, int]) -> Tuple[slice, ...]:
        return tuple([slice(dim_start, dim_start + chunk_dim) for (dim_start, chunk_dim) in
                      zip(coord, self.prediction_options.chunk_size[:3])])

    def _analyze_voxels(self, patient_idx, mask: np.ndarray, variance: np.ndarray):
        self.results['voxels_analyse'][patient_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        origin_x, origin_y, origin_z = mask.shape

        for x in range(0, origin_x)[:-1]:
            for y in range(0, origin_y)[:-1]:
                for z in range(0, origin_z)[:-1]:
                    mask_value = mask[x, y, z]
                    variance_value = variance[x, y, z]

                    self._update_results(patient_idx, mask_value, variance_value, 'voxels_analyse')

    def get_variance(self, image: np.ndarray, segmentation: np.ndarray, variance_path: str, nifti_affine, nifti_header,
                     should_save: bool = True):
        predictions = self.model_evaluator.evaluate(
            image,
            segmentation,
            self.prediction_options.mc_sample,
            self.prediction_options.stride)

        variance = self.model_evaluator.get_segmentation_variance(predictions, segmentation)

        if should_save:
            save_as_nifti(variance, Path(variance_path), nifti_affine, nifti_header)

        return variance

    def _update_results(self, patient_idx, ground_truth, prediction, analyse_type: str):
        if ground_truth and prediction:
            self.results[analyse_type][patient_idx]['tp'] += 1
            self.results[analyse_type]['overall']['tp'] += 1

        if not ground_truth and not prediction:
            self.results[analyse_type][patient_idx]['tn'] += 1
            self.results[analyse_type]['overall']['tn'] += 1

        if not ground_truth and prediction:
            self.results[analyse_type][patient_idx]['fp'] += 1
            self.results[analyse_type]['overall']['fp'] += 1

        if ground_truth and not prediction:
            self.results[analyse_type][patient_idx]['fn'] += 1
            self.results[analyse_type]['overall']['fn'] += 1

    def _summarize_results(self):
        for analyse_type in self.results.keys():
            overall = self.results[analyse_type]['overall']

            self.results[analyse_type]['overall']['tpr'] = overall['tp'] / max(1, (overall['tp'] + overall['fn']))
            self.results[analyse_type]['overall']['tnr'] = overall['tn'] / max(1, (overall['tn'] + overall['fp']))

    def _save_results(self):
        model_name = Path(self.model_path).stem
        results_path = os.path.join(self.output_path, f'results_{model_name}.json')
        with open(results_path, 'w') as r:
            json.dump(self.results, r)
