#!/bin/bash
#SBATCH --job-name=distance_estimation
#SBATCH --ntasks-per-node=1   
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --account=tra23_ELLIS
#SBATCH --reservation s_tra_Ellis1809
#SBATCH --cpus-per-task=10 ### Number of threads per task (OMP threads)
#SBATCH -o ./output/megatron/logs/%j.out
#SBATCH -e ./output/megatron/logs/%j.err

module purge
module load profile/deeplrn python

source ./venv/bin/activate
python main.py --model zhu --backbone resnet34 --regressor simple_roi --batch_size 4 --input_h_w 720 1280 --train_sampling_stride 50 --lr 5e-05 --loss l1



