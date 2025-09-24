trainer=csghmc_cold_restarts
dataset=$1
seed=$2
CFG=${3:-vit_b16}
DEVICE=${4:-0}
baseline=${5-}
CUDA_VISIBLE_DEVICES=$DEVICE ./scripts/${trainer}/base2new_train.sh $dataset $seed $CFG;
CUDA_VISIBLE_DEVICES=$DEVICE ./scripts/${trainer}/base2new_test.sh $dataset $seed $CFG;