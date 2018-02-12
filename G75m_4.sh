#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o outfile4  # send stdout to sample_experiment_outfile
#SBATCH -e errfile4  # send stderr to sample_experiment_errfile
#SBATCH -t 8:00:00  # time requested in hour:minute:secon
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=s1453274

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
export MLP_DATA_DIR="/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/"

python sgd_max.py --batch_size 128 --epochs 100 --experiment_prefix base1MaxoutSGD --dropout_rate 0.4 --batch_norm_use True --strided_dim_reduction False --seed 25012018 --classifier_type Baseline_classifier
