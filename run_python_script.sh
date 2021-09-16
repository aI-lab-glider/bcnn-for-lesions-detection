#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J Some_name

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=15GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgpreludium

## Specyfikacja partycji
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu

## Plik ze standardowym wyjściem
#SBATCH --output="output_0.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error_0.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

export LD_LIBRARY_PATH=/net/people/plgztabor/cuda/lib64:$LD_LIBRARY_PATH
module load plgrid/libs/tensorflow-gpu/2.2.0-python-3.7
pip3 install -r bcnn/requirements.txt
python3 $1

