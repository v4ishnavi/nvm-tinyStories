#! /bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=train-transformer
#SBATCH -o .sbatch-training-logs.out.%j
#SBATCH --gpus=1
#SBATCH --mem=16GB

# Starts a generation run using Ada + SLURM. This allows you to run
# without having to create interactive sessions (that have several
# restrctions: most important of them being the interruptions caused).

# This assumes the usage of uv, a Python tool for managing dependencies.
# It also assumes that the virtual environment is setup in .venv. Feel
# free to change this based on your preferences.

# Usage: sbatch ./scripts/ada-generation-run.sh

module add u18/python/3.11.2

# Ensures that virtual environment is running
source .venv/bin/activate

# Installs any new dependencies
cat requirements.txt | xargs uv add

# Runs the actual Python file
echo "starting training..."
python3 test.py -v --model-path artifacts/model.pt --vocab-path artifacts/vocab.pkl --output-file artifacts/generated_sentences.json
