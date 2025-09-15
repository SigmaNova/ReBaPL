import os
import yaml
from itertools import product

def generate_grid_search_configs():
    """Generate YAML configuration files for grid search optimization of cSGHMC parameters"""
    
    # Base configuration
    base_config = {
        'DATALOADER': {
            'TRAIN_X': {'BATCH_SIZE': 4},
            'TEST': {'BATCH_SIZE': 100},
            'NUM_WORKERS': 8
        },
        'INPUT': {
            'SIZE': [224, 224],
            'INTERPOLATION': 'bicubic',
            'PIXEL_MEAN': [0.48145466, 0.4578275, 0.40821073],
            'PIXEL_STD': [0.26862954, 0.26130258, 0.27577711],
            'TRANSFORMS': ['random_resized_crop', 'random_flip', 'normalize']
        },
        'OPTIM': {
            'NAME': 'sgd',
            'LR': 0.0025,
            'MAX_EPOCH': 20,
            'LR_SCHEDULER': 'cosine',
            'WARMUP_EPOCH': 1,
            'WARMUP_TYPE': 'constant',
            'WARMUP_CONS_LR': 1e-5
        },
        'TRAIN': {
            'PRINT_FREQ': 20
        },
        'MODEL': {
            'BACKBONE': {
                'NAME': 'ViT-B/16'
            }
        },
        'TRAINER': {
            'NAME': 'IVLP_cSGHMC',
            'IVLP': {
                'N_CTX_VISION': 4,
                'N_CTX_TEXT': 4,
                'CTX_INIT': 'a photo of a',
                'PREC': 'fp16',
                'PROMPT_DEPTH_VISION': 9,
                'PROMPT_DEPTH_TEXT': 9
            },
            'CSGHMC': {
                'CYCLE_LENGTH': 5,
                'M': 4
            }
        },
        'DATASET': {
            'NAME': 'EuroSAT',
            'NUM_SHOTS': 16
        }
    }
    
    # Grid search parameters
    beta_values = [0.7, 0.8, 0.9, 0.95]
    alpha_values = [0.001, 0.01, 0.05, 0.1]
    
    # Create output directory
    output_dir = "configs/trainers/IVLP_cSGHMC/grid_search"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    combinations = list(product(beta_values, alpha_values))
    
    print(f"Generating {len(combinations)} configuration files...")
    
    for i, (beta, alpha) in enumerate(combinations):

        # Create config for this combination
        config = base_config.copy()
        config['TRAINER'] = base_config['TRAINER'].copy()
        config['TRAINER']['CSGHMC'] = base_config['TRAINER']['CSGHMC'].copy()
        
        # Set the parameters
        config['TRAINER']['CSGHMC']['BETA'] = beta
        config['TRAINER']['CSGHMC']['ALPHA'] = alpha
        
        # Generate filename
        filename = f"vit_b16_beta{beta}_alpha{alpha}.yaml"
        filepath = os.path.join(output_dir, filename)
        
        # Save configuration
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated: {filename}")
    
    # Generate a summary file with all parameter combinations
    summary_file = os.path.join(output_dir, "grid_search_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Grid Search Parameter Combinations\n")
        f.write("==================================\n\n")
        f.write(f"Total configurations: {len(combinations)}\n\n")
        f.write("Parameters tested:\n")
        f.write(f"BETA: {beta_values}\n")
        f.write(f"ALPHA: {alpha_values}\n")
        f.write("-" * 60 + "\n")

        for beta, alpha in combinations:
            f.write(f"BETA: {beta:4.2f}, ALPHA: {alpha:5.3f}\n")

    print(f"\nGrid search configurations saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

def generate_batch_script():
    """Generate a batch script to run all grid search experiments"""
    
    # Grid search parameters (same as above)
    beta_values = [0.7, 0.8, 0.9, 0.95]
    alpha_values = [0.001, 0.01, 0.05, 0.1]

    combinations = list(product(beta_values, alpha_values))

    batch_script = """#!/bin/bash

# Grid Search Batch Script for IVLP_cSGHMC
# Run all parameter combinations

DATA=/home/ubuntu/omar/promptsrc/datasets
TRAINER=IVLP_cSGHMC
DATASET=eurosat
SEED=1
SHOTS=16

echo "Starting grid search with ${} configurations..."

""".format(len(combinations))

    for beta, alpha in combinations:

        config_name = f"vit_b16_beta{beta}_alpha{alpha}"
        
        batch_script += f"""
# Configuration: BETA={beta}, ALPHA={alpha}
echo "Running configuration: {config_name}"
CFG={config_name}
DIR=output/grid_search/${{DATASET}}/shots_${{SHOTS}}/${{TRAINER}}/${{CFG}}/seed${{SEED}}

python train.py \\
    --root ${{DATA}} \\
    --seed ${{SEED}} \\
    --trainer ${{TRAINER}} \\
    --dataset-config-file configs/datasets/${{DATASET}}.yaml \\
    --config-file configs/trainers/${{TRAINER}}/grid_search/${{CFG}}.yaml \\
    --output-dir ${{DIR}} \\
    DATASET.NUM_SHOTS ${{SHOTS}} \\
    DATASET.SUBSAMPLE_CLASSES base

echo "Completed: {config_name}"
echo "------------------------"

"""
    
    batch_script += """
echo "Grid search completed!"
echo "Results saved in output/grid_search/"
"""
    
    # Save batch script
    script_path = "scripts/ivlp_csghmc/grid_search_batch.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(batch_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"Batch script saved to: {script_path}")
    print("Run with: ./scripts/ivlp_csghmc/grid_search_batch.sh")

if __name__ == "__main__":
    generate_grid_search_configs()
    generate_batch_script()