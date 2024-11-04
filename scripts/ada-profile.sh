#! /bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=profile-transformer
#SBATCH -o .sbatch-profile-logs.out.%j
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --mem=16GB

# Starts a profiling run for the training run. This is 

# This assumes the usage of uv, a Python tool for managing dependencies.
# It also assumes that the virtual environment is setup in .venv. Feel
# free to change this based on your preferences.

# Usage: sbatch ./scripts/ada-profile.sh

module add u18/python/3.11.2

# Ensures that virtual environment is running
source .venv/bin/activate

# Installs any new dependencies
cat requirements.txt | xargs uv add

# Runs the actual Python file
echo "starting profiling..."
python3 -m kernprof -r -o ${SLURM_JOBID}.lprof -l train.py -v -dp
