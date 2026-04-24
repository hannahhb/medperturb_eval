#!/bin/bash
#
# TxAgent with AWS Bedrock backend — no GPU required.
# TxAgent handles the multi-step tool-calling loop; each LLM call goes to Bedrock.
#
#SBATCH -A punim0478
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cloud
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=medperturb_txagent_bedrock

source /home/${USER}/.bashrc
conda activate curebench

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH=/data/gpfs/projects/punim0478/bansaab/TxAgent_tool_tracking:$PYTHONPATH
export AWS_REGION="${AWS_REGION:-us-east-1}"

# Model for the TxAgent LLM calls via Bedrock
MODEL="${MODEL:-llama70b3_3}"
START="${START:-0}"
END="${END:-1}"

echo "Node    : $(hostname)"
echo "Model   : $MODEL (Bedrock)"
echo "Slice   : $START – $END"
echo "Start   : $(date)"

time python run.py --mode TXAGENT_BEDROCK --name "$MODEL" --start "$START" --end "$END"

echo "End     : $(date)"
