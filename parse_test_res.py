"""
Goal
---
1. Read test results from log files (latest timestamped version)
2. Compute mean and std across different folders (seeds)
3. Optional: Log results to MLflow for experiment tracking

Usage
---
The script supports two directory structures:

1. Base2New Structure (recommended):
   output/base2new/train_base/{dataset}/shots_{shots}/{trainer}/{config}/seed{seed}/log.txt
   output/base2new/test_new/{dataset}/shots_{shots}/{trainer}/{config}/seed{seed}/log.txt

2. Legacy Structure:
   my_experiment/seed1/log.txt
   my_experiment/seed2/log.txt

Base2New Structure Usage
---
Parse base classes results:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16

Parse novel classes results:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16 --test-novel

Parse both base and novel:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16 --test-both

Legacy Structure Usage
---
$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16 --ci95

MLflow Support
---
Add --use-mlflow to log results to MLflow:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16 --use-mlflow

You can specify the experiment name and tracking URI:

$ python tools/parse_test_res.py --dataset imagenet --shots 16 --trainer CoOp --config vit_b16 \
    --use-mlflow --mlflow-experiment my_experiment --mlflow-uri http://localhost:5000
"""
import re
import numpy as np
import os
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def build_base2new_path(dataset, shots, trainer, config, split_type="train_base", root="output/base2new"):
    """
    Build path for base2new structure.
    
    Args:
        dataset: Dataset name (e.g., 'imagenet', 'caltech101')
        shots: Number of shots (e.g., 16)
        trainer: Trainer name (e.g., 'CoOp', 'CoCoOp')
        config: Config name (e.g., 'vit_b16', 'rn50')
        split_type: 'train_base' or 'test_new'
        root: Root directory for outputs
    
    Returns:
        Full path to the experiment directory
    """
    path = osp.join(root, split_type, dataset, f"shots_{shots}", trainer, config)
    return path


def get_latest_log_file(directory):
    """
    Get the latest log file from a directory.
    Priority: log.txt-YYYY-MM-DD-HH-MM-SS[-MS] > log.txt
    If multiple timestamped files exist, return the most recent one.
    Handles both formats: with and without milliseconds.
    """
    if not os.path.isdir(directory):
        return None
    
    files = os.listdir(directory)
    
    # Pattern for timestamped log files: log.txt-YYYY-MM-DD-HH-MM-SS or log.txt-YYYY-MM-DD-HH-MM-SS-MS
    timestamp_pattern = re.compile(r'^log\.txt-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:-\d+)?$')
    
    # Find all timestamped log files
    timestamped_files = []
    for file in files:
        match = timestamp_pattern.match(file)
        if match:
            timestamp_str = match.group(1)  # Get the main timestamp part (without milliseconds)
            try:
                # Parse timestamp: YYYY-MM-DD-HH-MM-SS
                timestamp_parts = timestamp_str.split('-')
                if len(timestamp_parts) == 6:
                    dt_str = '-'.join(timestamp_parts[:3]) + ' ' + ':'.join(timestamp_parts[3:6])
                    timestamp = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    timestamped_files.append((file, timestamp, timestamp_str))
            except ValueError as e:
                # If timestamp parsing fails, skip this file
                print(f"Warning: Could not parse timestamp from {file}: {e}")
                continue
    
    # If we have timestamped files, return the most recent one
    if timestamped_files:
        # Sort by timestamp (most recent first)
        timestamped_files.sort(key=lambda x: x[1], reverse=True)
        latest_file = timestamped_files[0][0]
        return osp.join(directory, latest_file)
    
    # If no timestamped files, check for regular log.txt
    log_txt_path = osp.join(directory, "log.txt")
    if check_isfile(log_txt_path):
        return log_txt_path
    
    return None
def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    
    if not osp.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        return OrderedDict()
    
    msg_dir = osp.join(directory, "parse.txt")
    subdirs = listdir_nohidden(directory, sort=True)
    
    # Filter for seed directories
    seed_dirs = [d for d in subdirs if d.startswith('seed')]
    
    if not seed_dirs:
        print(f"Warning: No seed directories found in {directory}")
        return OrderedDict()

    outputs = []

    for subdir in seed_dirs:
        subdir_path = osp.join(directory, subdir)
        fpath = get_latest_log_file(subdir_path)
        
        if not fpath:
            print(f"Warning: No log file found in {subdir_path}")
            continue
            
        print(f"Using log file: {fpath}")
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    if len(outputs) == 0:
        print(f"Warning: No results found in {directory}")
        return OrderedDict()

    metrics_results = defaultdict(list)
    msg = ""
    for output in outputs:
        msg_one = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg_one += f"{key}: {value:.2f}%. "
            else:
                msg_one += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg_one)
        msg += f"{msg_one}\n"

    output_results = OrderedDict()
    output_results["_used_files"] = []  # Track which files were actually used

    print("===")
    print(f"Summary of directory: {directory}")
    msg += "==="
    msg += f"\nSummary of directory: {directory}"
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        msg += f"\n* {key}: {avg:.2f}% +- {std:.2f}%"
        output_results[key] = avg
        output_results[f"{key}_std"] = std
        output_results[f"{key}_values"] = values
    
    # Store the list of files that were used for metrics
    for output in outputs:
        if "file" in output:
            output_results["_used_files"].append(output["file"])
    
    print("===")
    msg += "\n==="
    with open(msg_dir, 'w') as write:
        write.write(msg)
    return output_results



def log_to_mlflow(results, experiment_name, run_name, args, extra_params=None):
    """
    Log results to MLflow for experiment tracking.
    
    Args:
        results: Dictionary of metrics to log
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
        args: Command line arguments
        extra_params: Additional parameters to log (e.g., split_type)
    """
    try:
        import mlflow
        
        # Set tracking URI if specified
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
            print(f"MLflow tracking URI set to: {args.mlflow_uri}")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log base2new parameters if available
            if args.dataset:
                mlflow.log_param("dataset", args.dataset)
            if args.shots is not None:
                mlflow.log_param("shots", args.shots)
            if args.trainer:
                mlflow.log_param("trainer", args.trainer)
            if args.config:
                mlflow.log_param("config", args.config)
            
            # Log additional parameters
            if extra_params:
                for key, value in extra_params.items():
                    mlflow.log_param(key, value)
            
            # Log standard parameters
            mlflow.log_param("ci95", args.ci95)
            mlflow.log_param("keyword", args.keyword)
            
            # Log metrics
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                elif isinstance(value, list):
                    # Log individual seed results
                    for idx, val in enumerate(value):
                        mlflow.log_metric(f"{key}_seed_{idx}", val)
            
            print(f"Results logged to MLflow experiment: {experiment_name}")
            
    except ImportError:
        print("Warning: MLflow not installed. Install with: pip install mlflow")
        print("Skipping MLflow logging.")
    except Exception as e:
        print(f"Warning: Failed to log to MLflow: {e}")
        print("Continuing without MLflow logging.")


def main(args, end_signal):
    metrics = [{
        "name": keyword,
        "regex": re.compile(fr"\* {keyword}: ([\.\deE+-]+)%")
        } for keyword in ["accuracy", "macro_f1"]]

    # Determine which splits to parse
    splits_to_parse = []
    splits_to_parse.append(("train_base", "base"))
    splits_to_parse.append(("test_new", "novel"))
    all_results = {"base": {}, "novel": {}}
    
    # Start a single MLflow run for all results if enabled
    mlflow_run = None
    if args.use_mlflow:
        try:
            import mlflow
            
            # Set tracking URI if specified
            if args.mlflow_uri:
                mlflow.set_tracking_uri(args.mlflow_uri)
                print(f"MLflow tracking URI set to: {args.mlflow_uri}")
            
            # Set experiment
            experiment_name = args.mlflow_experiment
            mlflow.set_experiment(experiment_name)
            
            # Start single run
            run_name = f"{args.config}_shots{args.shots}_{args.trainer}"
            mlflow_run = mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("dataset", args.dataset)
            mlflow.log_param("shots", args.shots)
            mlflow.log_param("trainer", args.trainer)
            mlflow.log_param("config", args.config)
            mlflow.log_param("ci95", args.ci95)
            
        except ImportError:
            print("Warning: MLflow not installed. Install with: pip install mlflow")
            print("Skipping MLflow logging.")
        except Exception as e:
            print(f"Warning: Failed to start MLflow run: {e}")
            print("Continuing without MLflow logging.")
    
    # Track all used log files across all splits and metrics
    all_used_files = set()
    
    for split_type, split_name in splits_to_parse:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} classes ({split_type})")
        print(f"{'='*60}")
        
        directory = build_base2new_path(
            args.dataset, args.shots, args.trainer, args.config, 
            split_type=split_type, root=args.root_dir
        )
        
        if not osp.exists(directory):
            print(f"Warning: Directory does not exist: {directory}")
            continue
        for metric in metrics:
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )
            
            if results:
                all_results[split_name][metric['name']] = results
                
                # Collect the files that were used for this metric
                if "_used_files" in results:
                    for fpath in results["_used_files"]:
                        all_used_files.add((fpath, split_name))
                
                # Log metrics to the single MLflow run
                if mlflow_run is not None:
                    try:
                        import mlflow
                        
                        # Log mean accuracy
                        mlflow.log_metric(f"{split_name}_{metric['name']}", results[metric['name']])
                        
                        # Log std/ci95
                        std_key = f"{metric['name']}_std"
                        if std_key in results:
                            metric_name = f"{split_name}_{metric['name']}_ci95" if args.ci95 else f"{split_name}_{metric['name']}_std"
                            mlflow.log_metric(metric_name, results[std_key])
                        
                        # Log individual seed values
                        values_key = f"{metric['name']}_values"
                        if values_key in results:
                            for idx, val in enumerate(results[values_key]):
                                mlflow.log_metric(f"{split_name}_{metric['name']}_seed{idx+1}", val)
                        
                    except Exception as e:
                        print(f"Warning: Failed to log {split_name} metrics to MLflow: {e}")
    
    # Upload only the log files that were actually used for computing metrics
    if mlflow_run is not None and len(all_used_files) > 0:
        try:
            import mlflow
            import shutil
            print(f"\nUploading {len(all_used_files)} log files to MLflow...")
            
            for log_file, split_name in all_used_files:
                if osp.isfile(log_file):
                    # Extract seed directory name from the path
                    seed_dir = osp.basename(osp.dirname(log_file))
                    
                    # Create a temporary copy with standardized name
                    temp_dir = "/tmp/mlflow_logs"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = osp.join(temp_dir, "log.txt")
                    shutil.copy(log_file, temp_file)
                    
                    # Create artifact path with split name and seed
                    artifact_path = f"logs/{split_name}/{seed_dir}"
                    mlflow.log_artifact(temp_file, artifact_path)
                    
                    # Clean up temp file
                    os.remove(temp_file)
                    
                    print(f"Logged {log_file} to MLflow artifacts at {artifact_path}/log.txt")
                    
        except Exception as e:
            print(f"Warning: Failed to log files to MLflow: {e}")
    # Print summary if both splits were parsed
    if len(all_results) == 2 and "base" in all_results and "novel" in all_results:
        print(f"\n{'='*60}")
        print("SUMMARY: Base and Novel Classes")
        print(f"{'='*60}")
        for metric in metrics:
            base_results = all_results["base"].get(metric['name'], {})
            novel_results = all_results["novel"].get(metric['name'], {})
            
            # Extract the accuracy values from the OrderedDict
            base_acc = base_results.get(metric['name'], 0) if isinstance(base_results, dict) else 0
            novel_acc = novel_results.get(metric['name'], 0) if isinstance(novel_results, dict) else 0
            
            harmonic_mean = 2 * (base_acc * novel_acc) / (base_acc + novel_acc) if (base_acc + novel_acc) > 0 else 0
            
            print(f"Base classes {metric['name']}:  {base_acc:.2f}%")
            print(f"Novel classes {metric['name']}: {novel_acc:.2f}%")
            print(f"Harmonic mean (H):      {harmonic_mean:.2f}%")
            print(f"{'='*60}")
            
            # Log harmonic mean to the single MLflow run
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metric(f"{metric['name']}/harmonic_mean", harmonic_mean)
                except Exception as e:
                    print(f"Warning: Failed to log harmonic mean to MLflow: {e}")
    
    # End the single MLflow run
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.end_run()
            print(f"\nResults logged to MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"Warning: Failed to end MLflow run: {e}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse test results from log files with support for base2new structure"
    )
    
    # Base2New structure arguments
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (e.g., imagenet, caltech101). Required for base2new structure."
    )
    parser.add_argument(
        "--shots", type=int, default=None,
        help="Number of shots (e.g., 1, 2, 4, 8, 16). Required for base2new structure."
    )
    parser.add_argument(
        "--trainer", type=str, default=None,
        help="Trainer name (e.g., CoOp, CoCoOp, PLOT). Required for base2new structure."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config name (e.g., vit_b16, rn50). Required for base2new structure."
    )
    parser.add_argument(
        "--root-dir", type=str, default="output/base2new",
        help="Root directory for base2new outputs (default: output/base2new)"
    )
    parser.add_argument(
        "--test-novel", action="store_true",
        help="Parse novel classes (test_new) instead of base classes"
    )
    parser.add_argument(
        "--test-both", action="store_true",
        help="Parse both base and novel classes and compute harmonic mean"
    )
    
    parser.add_argument(
        "--multi-exp", action="store_true",
        help="Parse multiple experiments (legacy structure)"
    )
    
    # Common arguments
    parser.add_argument(
        "--ci95", action="store_true",
        help=r"Compute 95\% confidence interval instead of standard deviation"
    )
    parser.add_argument(
        "--test-log", action="store_true",
        help="Parse test-only logs"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str,
        help="Which keyword to extract from logs (default: accuracy)"
    )
    
    # MLflow arguments
    parser.add_argument(
        "--use-mlflow", action="store_true",
        help="Log results to MLflow for experiment tracking"
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default="Bayesian-Prompt-Learning",
        help="MLflow experiment name (default: auto-generated based on trainer/dataset or directory)"
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="http://172.16.4.3:5000",
        help="MLflow tracking URI (default: uses MLflow default, typically ./mlruns)"
    )
    
    args = parser.parse_args()

    end_signal = "=> result"

    main(args, end_signal)