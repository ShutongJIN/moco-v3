#!/bin/bash
#SBATCH --gpus 8
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user shutong@kth.se
#SBATCH -o /home/x_shuji/moco-v3/logs/slurm-%A.out
#SBATCH -e /home/x_shuji/moco-v3/logs/slurm-%A.err

module load Anaconda/2021.05-nsc1
conda activate data4robotics

python main_moco.py \
  -a vit_small -b 1024 \
  --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
  --epochs=300 --warmup-epochs=40 \
  --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --checkpoint_dir /proj/cloudrobotics-nest/users/Stacking/dataset/CloudGripper_push_1k/moco/vit_small \
  --max_images 1000000 \
  /proj/cloudrobotics-nest/users/Stacking/dataset/CloudGripper_push_1k/hdf5
