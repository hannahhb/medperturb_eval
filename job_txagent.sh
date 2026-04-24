#!/bin/bash
#
#SBATCH -A punim0478
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=medperturb_txagent

module load CUDA
source /home/${USER}/.bashrc
conda activate curebench

export HF_HOME="/data/projects/punim0478/bansaab"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH=/data/gpfs/projects/punim0478/bansaab/TxAgent_tool_tracking:$PYTHONPATH

echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start   : $(date)"

time python run.py --mode TXAGENT --start 0 --end 1

echo "End     : $(date)"
