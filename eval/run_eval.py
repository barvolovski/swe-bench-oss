#!/usr/bin/env python3
"""
SWE-Bench evaluation script.

Evaluates generated patches against SWE-Bench gold patches using Docker containers.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# Dataset mappings
DATASET_MAP = {
    'pro': 'ScaleAI/SWE-bench_Pro',
    'lite': 'princeton-nlp/SWE-bench_Lite',
    'verified': 'princeton-nlp/SWE-bench_Verified',
}

# Docker image prefix for evaluation
DOCKER_IMAGE_PREFIX = 'jefzda/sweap-images'


def load_predictions(preds_dir: Path) -> Dict[str, str]:
    """
    Load predictions from directory.
    
    Args:
        preds_dir: Directory containing prediction subdirectories
        
    Returns:
        Dictionary mapping instance_id -> patch
    """
    predictions = {}
    
    for pred_file in preds_dir.rglob('*.pred'):
        instance_id = pred_file.stem
        patch = pred_file.read_text()
        predictions[instance_id] = patch
        
    console.print(f"[green]Loaded {len(predictions)} predictions[/green]")
    return predictions


def load_gold_patches(dataset: str, instance_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load gold patches from HuggingFace dataset.
    
    Args:
        dataset: Dataset name (pro, lite, verified)
        instance_ids: List of instance IDs to load
        
    Returns:
        Dictionary mapping instance_id -> task data
    """
    from datasets import load_dataset
    
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    console.print(f"[blue]Loading {dataset} dataset...[/blue]")
    ds = load_dataset(DATASET_MAP[dataset], split='test')
    
    gold = {}
    for row in ds:
        instance_id = row['instance_id']
        if instance_id in instance_ids:
            gold[instance_id] = {
                'instance_id': instance_id,
                'repo': row['repo'],
                'base_commit': row.get('base_commit', ''),
                'patch': row.get('patch', ''),
                'test_patch': row.get('test_patch', ''),
                'problem_statement': row.get('problem_statement', ''),
            }
    
    console.print(f"[green]Loaded {len(gold)} gold patches[/green]")
    return gold


def evaluate_instance(
    instance_id: str,
    pred_patch: str,
    gold_data: Dict[str, Any],
    timeout_minutes: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a single instance.
    
    Args:
        instance_id: Instance ID
        pred_patch: Predicted patch
        gold_data: Gold data from dataset
        timeout_minutes: Timeout for evaluation
        
    Returns:
        Evaluation result dictionary
    """
    start_time = datetime.now()
    
    # Get Docker image for this instance
    repo = gold_data['repo']
    base_commit = gold_data['base_commit']
    
    parts = repo.split('/')
    if len(parts) != 2:
        return {
            'instance_id': instance_id,
            'resolved': False,
            'error': f"Invalid repo format: {repo}",
        }
    
    repo_base, repo_name = parts
    image_tag = f"{repo_base}.{repo_name}-{repo_base}__{repo_name}-{base_commit}"
    image = f"{DOCKER_IMAGE_PREFIX}:{image_tag}"
    
    # Create temporary directory for evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write prediction patch
        pred_path = Path(tmpdir) / 'pred.patch'
        pred_path.write_text(pred_patch)
        
        # Write test patch if available
        test_patch = gold_data.get('test_patch', '')
        test_path = Path(tmpdir) / 'test.patch'
        test_path.write_text(test_patch)
        
        try:
            # Run evaluation in Docker container
            eval_script = '''
set -e
cd /workspace

# Apply prediction patch
if [ -s /eval/pred.patch ]; then
    git apply /eval/pred.patch || exit 1
fi

# Apply test patch if present
if [ -s /eval/test.patch ]; then
    git apply /eval/test.patch || true
fi

# Run tests (this varies by project)
# Try common test commands
if [ -f "pytest.ini" ] || [ -f "setup.py" ]; then
    python -m pytest --tb=short -q || exit 1
elif [ -f "package.json" ]; then
    npm test || exit 1
elif [ -f "go.mod" ]; then
    go test ./... || exit 1
elif [ -f "Makefile" ]; then
    make test || exit 1
fi

echo "EVAL_SUCCESS"
'''
            
            result = subprocess.run(
                [
                    'docker', 'run', '--rm',
                    '-v', f'{tmpdir}:/eval:ro',
                    image,
                    'bash', '-c', eval_script,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
            )
            
            resolved = 'EVAL_SUCCESS' in result.stdout
            
            return {
                'instance_id': instance_id,
                'resolved': resolved,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
            }
            
        except subprocess.TimeoutExpired:
            return {
                'instance_id': instance_id,
                'resolved': False,
                'error': 'Evaluation timeout',
                'duration_seconds': timeout_minutes * 60,
            }
        except Exception as e:
            return {
                'instance_id': instance_id,
                'resolved': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
            }


def evaluate_simple(
    instance_id: str,
    pred_patch: str,
    gold_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simple evaluation: check if prediction patch is non-empty and syntactically valid.
    
    For full evaluation, use Modal or Docker-based evaluation.
    """
    has_patch = bool(pred_patch.strip())
    
    # Basic diff validation
    is_valid_diff = False
    if has_patch:
        lines = pred_patch.split('\n')
        has_diff_header = any(line.startswith('diff --git') for line in lines)
        has_changes = any(line.startswith('+') or line.startswith('-') for line in lines)
        is_valid_diff = has_diff_header and has_changes
    
    return {
        'instance_id': instance_id,
        'has_patch': has_patch,
        'is_valid_diff': is_valid_diff,
        'patch_lines': len(pred_patch.split('\n')) if has_patch else 0,
        # Full resolved status requires running tests
        'resolved': None,  # Unknown without running tests
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SWE-Bench predictions')
    parser.add_argument('--preds', type=str, required=True, help='Predictions directory')
    parser.add_argument('--dataset', type=str, default='pro', help='Dataset (pro, lite, verified)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--full-eval', action='store_true', help='Run full Docker-based evaluation')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout per instance (minutes)')
    args = parser.parse_args()
    
    preds_dir = Path(args.preds)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    predictions = load_predictions(preds_dir)
    
    if not predictions:
        console.print("[red]No predictions found![/red]")
        sys.exit(1)
    
    # Load gold data
    gold_data = load_gold_patches(args.dataset, list(predictions.keys()))
    
    # Evaluate
    results = {}
    resolved_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(predictions))
        
        for instance_id, pred_patch in predictions.items():
            if instance_id not in gold_data:
                console.print(f"[yellow]Warning: {instance_id} not found in gold data[/yellow]")
                results[instance_id] = {
                    'instance_id': instance_id,
                    'resolved': False,
                    'error': 'Not found in dataset',
                }
            elif args.full_eval:
                # Full Docker-based evaluation
                result = evaluate_instance(
                    instance_id=instance_id,
                    pred_patch=pred_patch,
                    gold_data=gold_data[instance_id],
                    timeout_minutes=args.timeout,
                )
                results[instance_id] = result
                if result.get('resolved'):
                    resolved_count += 1
            else:
                # Simple validation only
                result = evaluate_simple(
                    instance_id=instance_id,
                    pred_patch=pred_patch,
                    gold_data=gold_data[instance_id],
                )
                results[instance_id] = result
            
            progress.update(task, advance=1)
    
    # Save results
    eval_results_path = output_dir / 'eval_results.json'
    eval_results_path.write_text(json.dumps(results, indent=2))
    
    # Calculate summary
    total = len(results)
    with_patch = sum(1 for r in results.values() if r.get('has_patch') or r.get('resolved'))
    valid_diff = sum(1 for r in results.values() if r.get('is_valid_diff'))
    
    summary = {
        'total': total,
        'with_patch': with_patch,
        'valid_diff': valid_diff,
        'resolved': resolved_count if args.full_eval else None,
        'accuracy': resolved_count / total if args.full_eval and total > 0 else None,
        'patch_rate': with_patch / total if total > 0 else 0,
        'evaluated_at': datetime.utcnow().isoformat(),
        'full_eval': args.full_eval,
    }
    
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    
    # Print summary table
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Instances", str(total))
    table.add_row("With Patch", f"{with_patch} ({100*with_patch/total:.1f}%)" if total > 0 else "0")
    table.add_row("Valid Diff", f"{valid_diff} ({100*valid_diff/total:.1f}%)" if total > 0 else "0")
    
    if args.full_eval:
        table.add_row("Resolved", f"{resolved_count} ({100*resolved_count/total:.1f}%)" if total > 0 else "0")
    
    console.print(table)
    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == '__main__':
    main()

