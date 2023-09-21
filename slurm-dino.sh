#!/bin/bash
#SBATCH --job-name=distance_estimation_dino
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --account=tra23_ELLIS
#SBATCH --reservation s_tra_Ellis
#SBATCH --cpus-per-task=8 ### Number of threads per task (OMP threads)
#SBATCH -o ./output/logs/dino-value.out
#SBATCH -e ./output/logs/dino-value.err
#SBATCH --time=24:00:00

module purge
module load profile/deeplrn python

nvidia-smi

source ./venv/bin/activate
python main.py --model zhu \
 --backbone dino \
 --regressor simple_roi \
 --batch_size 4 --input_h_w 720 1280 \
 --accumulation_steps 1\
 --lr 5e-05 \
 --loss l1 \
 --test_sampling_stride 1\
 --train_sampling_stride 1\
 --ds_path /leonardo/home/usertrain/a08tra51/distance_estimation_project/data/MOTSynth \
 --annotations_path /leonardo/home/usertrain/a08tra51/distance_estimation_project/annotations_clean\
 --num_gpus 2 \
 --epochs 20 \



