from preprocessing_pipeline.main import main as run_pipeline
import os

run_pipeline()
os.system("cd network && sbatch ../run_python_script.sh train.py")

