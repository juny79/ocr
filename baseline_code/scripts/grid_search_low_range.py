#!/usr/bin/env python3
"""
Extended grid search with lower threshold values.
Based on results showing better performance at lower thresh/box_thresh values.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import subprocess

def run_prediction(thresh, box_thresh, exp_name, checkpoint_path):
    """Run prediction with given parameters."""
    cmd = [
        "python", "runners/predict.py",
        "preset=hrnet_w44_1024",
        f"exp_name={exp_name}",
        f"checkpoint_path={checkpoint_path}",
        f"models.head.postprocess.thresh={thresh}",
        f"models.head.postprocess.box_thresh={box_thresh}"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/data/ephemeral/home/baseline_code"
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def convert_to_csv(json_path, csv_path):
    """Convert JSON prediction to CSV submission format."""
    cmd = [
        "python", "ocr/utils/convert_submission.py",
        "-J", json_path,
        "-O", csv_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/data/ephemeral/home/baseline_code"
        )
        return result.returncode == 0
    except Exception:
        return False


def grid_search():
    """Run extended grid search with lower threshold ranges."""
    
    # Configuration
    PROJECT_ROOT = Path("/data/ephemeral/home/baseline_code")
    CHECKPOINT_PATH = "best_model.ckpt"
    SUBMISSIONS_DIR = PROJECT_ROOT / "grid_search_submissions_low"
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    
    # Extended grid search parameters - LOWER RANGE
    # Based on result: 0.225/0.425 performed better (0.9853) than baseline (0.9851)
    THRESH_RANGE = np.arange(0.210, 0.231, 0.005)  # 0.210, 0.215, 0.220, 0.225, 0.230 (5 values)
    BOX_THRESH_RANGE = np.arange(0.400, 0.436, 0.005)  # 0.400, 0.405, ..., 0.435 (8 values)
    
    # Reference results
    KNOWN_RESULTS = {
        "baseline": {"thresh": 0.231, "box_thresh": 0.432, "hmean": 0.9851},
        "t0.240_bt0.440": {"hmean": 0.9846},
        "t0.225_bt0.425": {"hmean": 0.9853}  # Best so far!
    }
    
    print("=" * 80)
    print("Extended Grid Search - Lower Threshold Range")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Thresh range: {THRESH_RANGE[0]:.3f} to {THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Box_thresh range: {BOX_THRESH_RANGE[0]:.3f} to {BOX_THRESH_RANGE[-1]:.3f} (step: 0.005)")
    print(f"Total combinations: {len(THRESH_RANGE) * len(BOX_THRESH_RANGE)}")
    print(f"\nKnown Results:")
    print(f"  - Baseline (0.231/0.432): 0.9851")
    print(f"  - Best found (0.225/0.425): 0.9853 ⭐")
    print(f"  - Testing lower ranges to find optimal...")
    print(f"\nSubmissions will be saved to: {SUBMISSIONS_DIR}")
    print("=" * 80)
    
    results = {
        'known_results': KNOWN_RESULTS,
        'experiments': [],
        'search_space': {
            'thresh_range': THRESH_RANGE.tolist(),
            'box_thresh_range': BOX_THRESH_RANGE.tolist()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    total_experiments = len(THRESH_RANGE) * len(BOX_THRESH_RANGE)
    pbar = tqdm(total=total_experiments, desc="Grid Search Progress")
    
    for i, thresh in enumerate(THRESH_RANGE):
        for j, box_thresh in enumerate(BOX_THRESH_RANGE):
            exp_name = f"grid_search_low_t{thresh:.3f}_bt{box_thresh:.3f}"
            
            pbar.set_description(f"Testing thresh={thresh:.3f}, box_thresh={box_thresh:.3f}")
            
            # Run prediction
            success, stdout, stderr = run_prediction(
                thresh, box_thresh, exp_name, CHECKPOINT_PATH
            )
            
            if not success:
                experiment_result = {
                    'id': len(results['experiments']) + 1,
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'success': False,
                    'error': stderr[:200] if stderr else 'Prediction failed'
                }
                results['experiments'].append(experiment_result)
                pbar.update(1)
                continue
            
            # Find the generated JSON file
            outputs_dir = PROJECT_ROOT / f"outputs/{exp_name}/submissions"
            if not outputs_dir.exists():
                experiment_result = {
                    'id': len(results['experiments']) + 1,
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'success': False,
                    'error': 'Output directory not found'
                }
                results['experiments'].append(experiment_result)
                pbar.update(1)
                continue
            
            # Get the latest JSON file
            json_files = list(outputs_dir.glob("*.json"))
            if not json_files:
                experiment_result = {
                    'id': len(results['experiments']) + 1,
                    'thresh': float(thresh),
                    'box_thresh': float(box_thresh),
                    'success': False,
                    'error': 'No JSON output found'
                }
                results['experiments'].append(experiment_result)
                pbar.update(1)
                continue
            
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            
            # Convert to CSV
            csv_filename = f"submission_t{thresh:.3f}_bt{box_thresh:.3f}.csv"
            csv_path = SUBMISSIONS_DIR / csv_filename
            
            convert_success = convert_to_csv(str(latest_json), str(csv_path))
            
            experiment_result = {
                'id': len(results['experiments']) + 1,
                'thresh': float(thresh),
                'box_thresh': float(box_thresh),
                'success': convert_success,
                'submission_file': str(csv_path) if convert_success else None,
                'json_file': str(latest_json)
            }
            
            results['experiments'].append(experiment_result)
            pbar.update(1)
    
    pbar.close()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = PROJECT_ROOT / "grid_search_results" / f"grid_search_low_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = [r for r in results['experiments'] if r['success']]
    failed = [r for r in results['experiments'] if not r['success']]
    
    print("\n" + "=" * 80)
    print("Extended Grid Search Complete!")
    print("=" * 80)
    print(f"\nTotal experiments: {total_experiments}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    print(f"\nResults saved to: {results_file}")
    
    if successful:
        print(f"\nSubmission files created in: {SUBMISSIONS_DIR}")
        print("\n🎯 Target: Find combinations better than 0.9853 (current best)")
        print("\nGenerated submission files:")
        print("-" * 80)
        
        # Group by thresh for better readability
        by_thresh = {}
        for result in successful:
            t = result['thresh']
            if t not in by_thresh:
                by_thresh[t] = []
            by_thresh[t].append(result)
        
        for thresh in sorted(by_thresh.keys()):
            print(f"\nthresh={thresh:.3f}:")
            for result in sorted(by_thresh[thresh], key=lambda x: x['box_thresh']):
                filename = Path(result['submission_file']).name
                print(f"  box_thresh={result['box_thresh']:.3f} → {filename}")
    
    if failed:
        print(f"\n⚠ {len(failed)} experiments failed. Check the results file for details.")
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("1. Submit generated CSV files to leaderboard")
    print("2. Record H-Mean scores for each combination")
    print("3. Identify the optimal parameters")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    os.chdir("/data/ephemeral/home/baseline_code")
    
    venv_activate = "/data/ephemeral/home/venv/bin/activate_this.py"
    if Path(venv_activate).exists():
        with open(venv_activate) as f:
            exec(f.read(), {'__file__': venv_activate})
    
    grid_search()
