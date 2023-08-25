#!/bin/bash
#SBATCH --job-name="Cha tiny_imagenet"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:40G
#SBATCH --mem=20G
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python3 tiny.py -a VisionTransformer  --strength 0.1 --logboard runs/tiny200-augment/vit/base_augment --ABA_variant no --epochs 50 --scheduler-interval 25 --lr 0.1 --batch-size 512 --wd 0.001 --augment 1 --k 5 --tinytransformer 1 /storage/group/dataset_mirrors/old_common_datasets/tiny-imagenet-200

#/storage/group/dataset_mirrors/imagenet2012/imagenet2012_download
#/storage/group/dataset_mirrors/old_common_datasets/tiny-imagenet-200
