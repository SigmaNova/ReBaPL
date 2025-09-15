#!/bin/bash

# Grid Search Batch Script for IVLP_cSGHMC
# Run all parameter combinations

DATA=/home/ubuntu/omar/promptsrc/datasets
TRAINER=IVLP_cSGHMC
DATASET=eurosat
SEED=1
SHOTS=16

echo "Starting grid search with $16 configurations..."


# Configuration: BETA=0.7, ALPHA=0.001
echo "Running configuration: vit_b16_beta0.7_alpha0.001"
CFG=vit_b16_beta0.7_alpha0.001
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.7_alpha0.001"
echo "------------------------"


# Configuration: BETA=0.7, ALPHA=0.01
echo "Running configuration: vit_b16_beta0.7_alpha0.01"
CFG=vit_b16_beta0.7_alpha0.01
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.7_alpha0.01"
echo "------------------------"


# Configuration: BETA=0.7, ALPHA=0.05
echo "Running configuration: vit_b16_beta0.7_alpha0.05"
CFG=vit_b16_beta0.7_alpha0.05
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.7_alpha0.05"
echo "------------------------"


# Configuration: BETA=0.7, ALPHA=0.1
echo "Running configuration: vit_b16_beta0.7_alpha0.1"
CFG=vit_b16_beta0.7_alpha0.1
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.7_alpha0.1"
echo "------------------------"


# Configuration: BETA=0.8, ALPHA=0.001
echo "Running configuration: vit_b16_beta0.8_alpha0.001"
CFG=vit_b16_beta0.8_alpha0.001
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.8_alpha0.001"
echo "------------------------"


# Configuration: BETA=0.8, ALPHA=0.01
echo "Running configuration: vit_b16_beta0.8_alpha0.01"
CFG=vit_b16_beta0.8_alpha0.01
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.8_alpha0.01"
echo "------------------------"


# Configuration: BETA=0.8, ALPHA=0.05
echo "Running configuration: vit_b16_beta0.8_alpha0.05"
CFG=vit_b16_beta0.8_alpha0.05
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.8_alpha0.05"
echo "------------------------"


# Configuration: BETA=0.8, ALPHA=0.1
echo "Running configuration: vit_b16_beta0.8_alpha0.1"
CFG=vit_b16_beta0.8_alpha0.1
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.8_alpha0.1"
echo "------------------------"


# Configuration: BETA=0.9, ALPHA=0.001
echo "Running configuration: vit_b16_beta0.9_alpha0.001"
CFG=vit_b16_beta0.9_alpha0.001
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.9_alpha0.001"
echo "------------------------"


# Configuration: BETA=0.9, ALPHA=0.01
echo "Running configuration: vit_b16_beta0.9_alpha0.01"
CFG=vit_b16_beta0.9_alpha0.01
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.9_alpha0.01"
echo "------------------------"


# Configuration: BETA=0.9, ALPHA=0.05
echo "Running configuration: vit_b16_beta0.9_alpha0.05"
CFG=vit_b16_beta0.9_alpha0.05
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.9_alpha0.05"
echo "------------------------"


# Configuration: BETA=0.9, ALPHA=0.1
echo "Running configuration: vit_b16_beta0.9_alpha0.1"
CFG=vit_b16_beta0.9_alpha0.1
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.9_alpha0.1"
echo "------------------------"


# Configuration: BETA=0.95, ALPHA=0.001
echo "Running configuration: vit_b16_beta0.95_alpha0.001"
CFG=vit_b16_beta0.95_alpha0.001
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.95_alpha0.001"
echo "------------------------"


# Configuration: BETA=0.95, ALPHA=0.01
echo "Running configuration: vit_b16_beta0.95_alpha0.01"
CFG=vit_b16_beta0.95_alpha0.01
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.95_alpha0.01"
echo "------------------------"


# Configuration: BETA=0.95, ALPHA=0.05
echo "Running configuration: vit_b16_beta0.95_alpha0.05"
CFG=vit_b16_beta0.95_alpha0.05
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.95_alpha0.05"
echo "------------------------"


# Configuration: BETA=0.95, ALPHA=0.1
echo "Running configuration: vit_b16_beta0.95_alpha0.1"
CFG=vit_b16_beta0.95_alpha0.1
DIR=output/grid_search/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/grid_search/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: vit_b16_beta0.95_alpha0.1"
echo "------------------------"


echo "Grid search completed!"
echo "Results saved in output/grid_search/"
