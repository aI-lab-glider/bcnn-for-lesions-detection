#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J bayesian_prediction

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=15GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=70:00:00

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgonwelo

## Specyfikacja partycji
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu

## Plik ze standardowym wyjściem
#SBATCH --output="output_0.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error_0.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module load plgrid/tools/python/3.8
module load plgrid/apps/cuda/11.2

python3.6 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip3 install -e .

python3.6 $@

