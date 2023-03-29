#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW7 CS 601.471/671 homework"
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-0

module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies
pip install typing-extensions --upgrade

# define hyperparameter sweep
learning_rates=(1e-4 5e-4 1e-3)
training_epochs=(5 7 9)

# divide slurm array task id by 3 to get the index of the learning rate
j=$((SLURM_ARRAY_TASK_ID / 3))

# mod slurm array task id by 3 to get the index of the training epoch
k=$((SLURM_ARRAY_TASK_ID % 3))

# runs your code
srun python train.py \
    --experiment ${learning_rates[$j]}_${training_epochs[$k]} \
    --lr ${learning_rates[$j]} \
    --num_epochs ${training_epochs[$k]} \
    --device cuda \
    --model "facebook/rag-sequence-nq" \
    --batch_size "8" \
    --include_gold_passage True \
    --small_subset True 
