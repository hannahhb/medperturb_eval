#!/bin/bash
#
#SBATCH -A punim0478
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cloud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=medperturb_aws

source /home/${USER}/.bashrc
conda activate curebench

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export AWS_REGION="${AWS_REGION:-us-east-1}"

# Model options: llama70b3_3  gpt_oss_120b  deepseekr1  qwen235b
MODEL="${MODEL:-llama70b3_3}"
START="${START:-0}"
END="${END:-1}"

echo "Node    : $(hostname)"
echo "Model   : $MODEL"
echo "Slice   : $START – $END"
echo "Start   : $(date)"

time python run.py --mode AWS --name "$MODEL" --start "$START" --end "$END"

echo "End     : $(date)"
