#!/bin/bash

#cd ../..

# custom config
DATA=/home/ubuntu/omar/promptsrc/datasets
TRAINER=CSGHMC_CR

DATASET=$1
SEED=$2

CFG=vit_b16
SHOTS=16
SUB=new

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base
