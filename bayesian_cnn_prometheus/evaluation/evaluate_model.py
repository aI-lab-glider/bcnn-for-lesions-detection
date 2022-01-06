from pathlib import Path
from typing import Tuple

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
import nibabel as nib
from dataclasses import dataclass

from bayesian_cnn_prometheus.evaluation.utils import load_config
import random
import os
import glob
import sys

@dataclass
class EvaluationConfig:
    model_name: str
    patient_id: int



def get_patients_to_predict():
    idxs_with_lesions = [3,4,5,6,8,10,11,14,17,21,26,27,29,30,32,34,35,36,37,39,41,42,43,44,45,46,48,49,50,51,53,54,
    58,59,63,65,66,67,68,69,71,73,75,78,80,81,83,85,89,93,94,96,97,98,99,101,102,104,105,106,107,108,109,111,112,114,116,118,
    119,120,122,123,127,128,134,138,139,141,148,150,152,155,157,158,160,164,166,167,168,
    171,173,175,177,178,179,180,181,182,183,184,186,188,189,190,191,192,195,199,202,204,205,206,207,209,211,212,213,215,216,218,
    219,223,224,225,230,231,232,233,235,236,237,239]
    return random.sample(idxs_with_lesions, k=3)


def main():
    folder = Path(sys.argv[1])
    model_config = load_config(folder/'config.json')
    weights_path = get_latest_weights_from_folder(folder)
    results_dir = create_predictions_dir(weights_path)
    pacient_idxs = get_patients_to_predict()
    for idx in pacient_idxs:
        make_prediction(weights_path, idx, PredictionOptions(
            chunk_size=model_config['preprocessing']['create_chunks']['chunk_size'],
            stride=model_config['preprocessing']['create_chunks']['stride'],
            mc_sample=model_config['mc_samples']
        ), results_dir)
    
def get_latest_weights_from_folder(folder: Path):
    return max(folder.iterdir(), key=os.path.getctime)

    
def create_predictions_dir(weights_path: Path):
    weights_folder_name = weights_path.stem.split('.')[:-1]
    weights_folder_name = f"{''.join(weights_folder_name)}_predictions"
    Path(weights_folder_name).mkdir(parents=True, exist_ok=True)
    return Path(weights_folder_name)

@dataclass
class PredictionOptions:
    chunk_size: Tuple[int,int,int]
    mc_sample: int
    stride: Tuple[int,int,int]
    

def make_prediction(weights_path: Path, patient_idx, prediction_options: PredictionOptions, results_dir: Path):
    image_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format(f'{patient_idx:0>4}', 'nii.gz')

    image = nib.load(image_path)
    model_evaluator = BayesianModelEvaluator(
        str(weights_path),
        prediction_options.chunk_size
    )

    predictions = model_evaluator.evaluate(
        image_path,
        prediction_options.mc_sample,
        prediction_options.stride)
    
    model_evaluator.save_predictions(
        results_dir,
        patient_idx,
        predictions,
        image.affine,
        image.header,
        False)

   

if __name__ == '__main__':
    main()
