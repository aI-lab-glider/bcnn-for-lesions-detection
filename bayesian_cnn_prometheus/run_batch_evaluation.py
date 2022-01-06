import glob
import os

from bayesian_cnn_prometheus.constants import Paths
if __name__ == '__main__':    
    model_folders = glob.glob('preprocessing*')
    for folder in model_folders:
        os.system(f'sbatch run_python_script.sh {Paths.PROJECT_DIR/"evaluation"/"evaluate_model.py"} {folder}')