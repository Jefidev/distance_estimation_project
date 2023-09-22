SLURM_NTASKS=9
#parallel -P $SLURM_NTASKS srun -t 0-3:00 --ntasks-per-node=1 --exclusive python prepare_joints_from_mp.py ::: {0..10}
parallel -P $SLURM_NTASKS srun -t 0-3:00 --ntasks-per-node=1 --exclusive python prepare_distances_from_mp.py ::: {0..9}