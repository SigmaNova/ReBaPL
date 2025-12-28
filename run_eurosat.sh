#!/bin/bash

# Experimental parameters
SEEDS=(2 3)

BATCH_SIZES=(1 16 32 64)  # 1 for single sample, -1 for all samples
# BATCH_SIZES=(1)  # 1 for single sample, -1 for all samples
REPULSION_STRENGTHS=(0.001 0.01 0.1)
DISTANCES=("wasserstein")
MAX_EPOCHS=(15)

# Fixed parameters
DATASET="eurosat"
TRAINER="CSGHMC_CR_MAPLE"
CONFIG="vit_b16"
SHOTS=16

echo "Starting CSGHMC_CR_MAPLE experiments on ${DATASET}"
echo "Testing ${#SEEDS[@]} seeds, ${#BATCH_SIZES[@]} batch sizes, ${#REPULSION_STRENGTHS[@]} repulsion strengths, ${#DISTANCES[@]} distances"
echo "Total experiments: $((${#SEEDS[@]} * ${#BATCH_SIZES[@]} * ${#REPULSION_STRENGTHS[@]} * ${#DISTANCES[@]} * 2))"
echo ""

experiment_count=0

total_experiments=$((${#SEEDS[@]} * ${#BATCH_SIZES[@]} * ${#REPULSION_STRENGTHS[@]} * ${#DISTANCES[@]} * ${#MAX_EPOCHS[@]}))

for seed in "${SEEDS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for repulsion_strength in "${REPULSION_STRENGTHS[@]}"; do
            for distance in "${DISTANCES[@]}"; do
                for max_epoch in "${MAX_EPOCHS[@]}"; do
                    experiment_count=$((experiment_count + 1))
                    
                    # Create parameter suffix for directory naming
                    PARAM_SUFFIX="_rs${repulsion_strength}_bs${batch_size}_${distance}_ep${max_epoch}"
                    
                    # Define directories
                    TRAIN_DIR="output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CONFIG}/seed${seed}${PARAM_SUFFIX}"
                    TEST_DIR="output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CONFIG}/seed${seed}${PARAM_SUFFIX}"
                    
                    echo "=========================================="
                    echo "Experiment ${experiment_count}/${total_experiments}"
                    echo "Seed: ${seed}, Batch Size: ${batch_size}, Repulsion: ${repulsion_strength}, Distance: ${distance}, Max Epoch: ${max_epoch}"
                    echo "=========================================="
                    
                    # Training phase
                    echo "Training on base classes..."
                    python train.py \
                    --root ~/datasets \
                    --seed ${seed} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                    --config-file configs/trainers/${TRAINER}/${CONFIG}.yaml \
                    --output-dir ${TRAIN_DIR} \
                    DATASET.NUM_SHOTS ${SHOTS} \
                    DATASET.SUBSAMPLE_CLASSES base \
                    CSGHMC.REPULSION.REPULSION_STRENGTH ${repulsion_strength} \
                    CSGHMC.REPULSION.BATCH_SIZE ${batch_size} \
                    CSGHMC.REPULSION.DISTANCE_TYPE ${distance} \
                    CSGHMC.CHAINS parallel \
                    OPTIM.MAX_EPOCH ${max_epoch}
                    
                    # Check if training was successful
                    if [ $? -eq 0 ]; then
                        echo "Training completed successfully."
                        
                        # Testing phase
                        echo "Testing on new classes..."
                        python train.py \
                        --root ~/datasets \
                        --seed ${seed} \
                        --trainer ${TRAINER} \
                        --dataset-config-file configs/datasets/${DATASET}.yaml \
                        --config-file configs/trainers/${TRAINER}/${CONFIG}.yaml \
                        --output-dir ${TEST_DIR} \
                        --model-dir ${TRAIN_DIR} \
                        --eval-only \
                        DATASET.NUM_SHOTS ${SHOTS} \
                        DATASET.SUBSAMPLE_CLASSES new \
                        CSGHMC.REPULSION.REPULSION_STRENGTH ${repulsion_strength} \
                        CSGHMC.REPULSION.BATCH_SIZE ${batch_size} \
                        CSGHMC.REPULSION.DISTANCE_TYPE ${distance} \
                        CSGHMC.CHAINS parallel \
                        OPTIM.MAX_EPOCH ${max_epoch}
                        
                        if [ $? -eq 0 ]; then
                            echo "Testing completed successfully."
                        else
                            echo "ERROR: Testing failed!"
                        fi
                    else
                        echo "ERROR: Training failed! Skipping testing."
                    fi
                    
                    echo ""
                done
            done
        done
    done
done

echo "All experiments completed!"
echo "Results saved in output/base2new/ directories with parameter suffixes."