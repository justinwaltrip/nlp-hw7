#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-8

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
j = $((SLURM_ARRAY_TASK_ID / 3))

# mod slurm array task id by 3 to get the index of the training epoch
k = $((SLURM_ARRAY_TASK_ID % 3))

# runs your code
srun python train.py \
    --lr ${learning_rates[$j]} \
    --num_epochs ${training_epochs[$k]} \
    --device cpu \
    --model "facebook/rag-sequence-nq" \
    --batch_size "32" \
    --experiment ${SLURM_ARRAY_TASK_ID} \
    --small_subset True 
