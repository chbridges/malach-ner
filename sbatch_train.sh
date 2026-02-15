#!/bin/bash
#SBATCH -J NER                                  # name of this job
#SBATCH -o ./logs/%x.%j.out                     # name of output file (MyJob.AllocationNumber.out)
#SBATCH -p gpu-ms,gpu-troja                     # name of GPU partition (can also be gpu-troja)
#SBATCH -N 1                                    # number of computation nodes to request
#SBATCH -c 1                                    # number of CPUs to request (default 1)
#SBATCH -G 1                                    # number of GPUs to request (default 0)
#SBATCH -C "gpuram24G|gpuram48G"                # request GPU with 16G OR 24G
#SBATCH --mem=16G                               # request 16G memory per node (max 32G per GPU)
#SBATCH --mail-type=END,FAIL                    # send a mail when the job completes or crashes
#SBATCH --mail-user=bruckner@ufal.mff.cuni.cz

set -e                                          

[ "$#" -ge 1 ] || { echo Usage: "$0 [large|ehri|malach1_best|malach1_last|malach2_best|malach2_last]" >&2; exit 1; }  

python train_ner.py --model_name "$1"
