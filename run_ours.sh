
#! /bin/bash

# trainer=csghmc_cold_restarts
trainer=csghmc_cold_restarts_maple
# trainer=maple
dataset=$1
seed=$2
CFG=${3:-vit_b16}
DEVICE=${4:-0}
overwrite=${5:-false}

# Delete checkpoints if overwrite is true
# if [ "$overwrite" = "true" ]; then
#   echo "Overwrite flag set to true. Removing existing checkpoints..."
#   [ -d "output/base2new/train_base/$dataset/shots_16/CSGHMC_CR/$CFG/seed_$seed" ] && rm -rf "output/base2new/train_base/$dataset/shots_16/CSGHMC_CR/$CFG/seed_$seed" && echo "Removed train checkpoint."
#   [ -d "output/base2new/test_new/eurosat/shots_16/CSGHMC_CR/vit_b16/seed_$seed" ] && rm -rf "output/base2new/test_new/eurosat/shots_16/CSGHMC_CR/vit_b16/seed_$seed" && echo "Removed test checkpoint."
# fi
if [ "$overwrite" = "true" ]; then
  echo "deleting path 'output/base2new/train_base/$dataset/shots_16/CSGHMC_CR_MAPLE/$CFG/seed$seed'"

  echo "Overwrite flag set to true. Removing existing checkpoints..."
  [ -d "output/base2new/train_base/$dataset/shots_16/CSGHMC_CR_MAPLE/$CFG/seed$seed" ] && rm -rf "output/base2new/train_base/$dataset/shots_16/CSGHMC_CR_MAPLE/$CFG/seed$seed" && echo "Removed train checkpoint."
  [ -d "output/base2new/test_new/$dataset/shots_16/CSGHMC_CR_MAPLE/$CFG/seed$seed" ] && rm -rf "output/base2new/test_new/$dataset/shots_16/CSGHMC_CR_MAPLE/$CFG/seed$seed" && echo "Removed test checkpoint."
fi

CUDA_VISIBLE_DEVICES=$DEVICE bash ./scripts/${trainer}/base2new_train.sh $dataset $seed $CFG
CUDA_VISIBLE_DEVICES=$DEVICE bash ./scripts/${trainer}/base2new_test.sh $dataset $seed $CFG
# CUDA_VISIBLE_DEVICES=$DEVICE ./scripts/${trainer}/base2new_train_maple.sh $dataset $seed
# CUDA_VISIBLE_DEVICES=$DEVICE ./scripts/${trainer}/base2new_test_maple.sh $dataset $seed