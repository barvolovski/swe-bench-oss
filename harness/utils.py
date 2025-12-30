#!/usr/bin/env python3
"""
Shared utilities for SWE-Bench harness scripts.
"""
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from datasets import load_dataset
from rich.console import Console

console = Console()


# Dataset mappings
DATASET_MAP = {
    'pro': 'ScaleAI/SWE-bench_Pro',
    'lite': 'princeton-nlp/SWE-bench_Lite',
    'verified': 'princeton-nlp/SWE-bench_Verified',
}

# Docker image prefix for SWE-Bench Pro
DOCKER_IMAGE_PREFIX = 'jefzda/sweap-images'


@dataclass
class TaskResult:
    """Result of running an agent on a task."""
    instance_id: str
    success: bool
    patch: str = ""
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class RunMetadata:
    """Metadata for a benchmark run."""
    run_id: str
    agent: str
    model: str
    dataset: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    results: List[TaskResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'agent': self.agent,
            'model': self.model,
            'dataset': self.dataset,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'results': [
                {
                    'instance_id': r.instance_id,
                    'success': r.success,
                    'duration_seconds': r.duration_seconds,
                    'cost_usd': r.cost_usd,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens,
                    'cached_tokens': r.cached_tokens,
                }
                for r in self.results
            ],
            'summary': {
                'total': len(self.results),
                'successful': sum(1 for r in self.results if r.success),
                'total_cost_usd': sum(r.cost_usd for r in self.results),
                'total_input_tokens': sum(r.input_tokens for r in self.results),
                'total_output_tokens': sum(r.output_tokens for r in self.results),
            }
        }


def load_tasks(dataset: str, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load tasks from HuggingFace dataset.
    
    Args:
        dataset: Dataset name (pro, lite, verified)
        task_ids: Optional list of specific task IDs to load
        
    Returns:
        List of task dictionaries
    """
    if dataset not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset}. Use one of: {list(DATASET_MAP.keys())}")
    
    console.print(f"[blue]Loading {dataset} dataset from HuggingFace...[/blue]")
    ds = load_dataset(DATASET_MAP[dataset], split='test')
    
    tasks = []
    for row in ds:
        instance_id = row['instance_id']
        
        # Filter by task_ids if specified
        if task_ids and instance_id not in task_ids:
            # Also try without instance_ prefix
            if not instance_id.startswith('instance_') or instance_id[9:] not in task_ids:
                continue
        
        tasks.append({
            'instance_id': instance_id,
            'repo': row['repo'],
            'problem_statement': row['problem_statement'],
            'base_commit': row.get('base_commit', ''),
            'hints_text': row.get('hints_text', ''),
        })
    
    console.print(f"[green]Loaded {len(tasks)} tasks[/green]")
    return tasks


def get_docker_image(instance_id: str, repo: str, base_commit: str) -> str:
    """
    Get the Docker image name for a SWE-Bench instance.
    
    Args:
        instance_id: The instance ID
        repo: Repository name (e.g., "gravitational/teleport")
        base_commit: Base commit hash
        
    Returns:
        Docker image name
    """
    # Parse repo into base and name
    parts = repo.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid repo format: {repo}")
    
    repo_base, repo_name = parts
    
    # Build image tag
    # Format: jefzda/sweap-images:repo_base.repo_name-repo_base__repo_name-commit-version
    tag = f"{repo_base}.{repo_name}-{repo_base}__{repo_name}-{base_commit}"
    
    return f"{DOCKER_IMAGE_PREFIX}:{tag}"


def setup_workspace(task: Dict[str, Any], workspace_root: Path) -> Path:
    """
    Set up a workspace for a task by pulling and running the Docker container.
    
    Args:
        task: Task dictionary with instance_id, repo, base_commit
        workspace_root: Root directory for workspaces
        
    Returns:
        Path to the workspace directory
    """
    instance_id = task['instance_id']
    workspace = workspace_root / instance_id
    
    # Clean up existing workspace
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    
    # Get Docker image
    image = get_docker_image(
        instance_id=instance_id,
        repo=task['repo'],
        base_commit=task['base_commit']
    )
    
    console.print(f"[blue]Pulling Docker image: {image}[/blue]")
    
    try:
        client = docker.from_env()
        client.images.pull(image)
        
        # Run container and copy files to workspace
        container = client.containers.run(
            image,
            command='tar -cf - .',
            detach=True,
            remove=False,
        )
        
        # Wait for container to finish
        container.wait()
        
        # Get the tarball and extract
        bits, _ = container.get_archive('/')
        
        # Write tar to temp file and extract
        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as f:
            for chunk in bits:
                f.write(chunk)
            tar_path = f.name
        
        # Extract tar to workspace
        subprocess.run(['tar', '-xf', tar_path, '-C', str(workspace)], check=True)
        os.unlink(tar_path)
        
        # Cleanup container
        container.remove()
        
        console.print(f"[green]Workspace ready: {workspace}[/green]")
        return workspace
        
    except Exception as e:
        console.print(f"[red]Failed to setup workspace: {e}[/red]")
        raise


def setup_workspace_simple(task: Dict[str, Any], workspace_root: Path) -> Path:
    """
    Set up a workspace by cloning the repo directly (simpler alternative to Docker).
    
    Args:
        task: Task dictionary
        workspace_root: Root directory for workspaces
        
    Returns:
        Path to the workspace directory
    """
    instance_id = task['instance_id']
    repo = task['repo']
    base_commit = task['base_commit']
    
    workspace = workspace_root / instance_id
    
    # Clean up existing workspace
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    
    console.print(f"[blue]Cloning {repo} at {base_commit[:8]}...[/blue]")
    
    # Clone the repository
    clone_url = f"https://github.com/{repo}.git"
    subprocess.run(
        ['git', 'clone', '--depth', '1', clone_url, str(workspace)],
        check=True,
        capture_output=True
    )
    
    # Checkout the specific commit (if we have full history)
    if base_commit:
        try:
            subprocess.run(
                ['git', 'fetch', '--depth', '1', 'origin', base_commit],
                cwd=workspace,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ['git', 'checkout', base_commit],
                cwd=workspace,
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            console.print(f"[yellow]Could not checkout {base_commit[:8]}, using HEAD[/yellow]")
    
    console.print(f"[green]Workspace ready: {workspace}[/green]")
    return workspace


def gather_diff(workspace: Path) -> str:
    """
    Gather the git diff from a workspace.
    
    Args:
        workspace: Path to the workspace
        
    Returns:
        Git diff as string
    """
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[yellow]Failed to get diff: {e}[/yellow]")
        return ""


def save_prediction(output_dir: Path, instance_id: str, patch: str, metadata: Dict[str, Any]):
    """
    Save a prediction in SWE-Bench format.
    
    Args:
        output_dir: Output directory
        instance_id: Instance ID
        patch: The generated patch
        metadata: Additional metadata
    """
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Save .pred file
    pred_file = instance_dir / f"{instance_id}.pred"
    pred_file.write_text(patch)
    
    # Save metadata
    meta_file = instance_dir / "metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2))
    
    console.print(f"[green]Saved prediction: {pred_file}[/green]")


def save_run_metadata(output_dir: Path, metadata: RunMetadata):
    """
    Save run metadata.
    
    Args:
        output_dir: Output directory
        metadata: Run metadata
    """
    metadata.completed_at = datetime.utcnow().isoformat()
    
    # Save run.json
    run_file = output_dir / "run.json"
    run_file.write_text(json.dumps(metadata.to_dict(), indent=2))
    
    # Save timing.json
    timing = [
        {
            'instance_id': r.instance_id,
            'duration_seconds': r.duration_seconds,
        }
        for r in metadata.results
    ]
    timing_file = output_dir / "timing.json"
    timing_file.write_text(json.dumps(timing, indent=2))
    
    # Save cost.json
    costs = [
        {
            'instance_id': r.instance_id,
            'cost_usd': r.cost_usd,
            'input_tokens': r.input_tokens,
            'output_tokens': r.output_tokens,
            'cached_tokens': r.cached_tokens,
        }
        for r in metadata.results
    ]
    cost_file = output_dir / "cost.json"
    cost_file.write_text(json.dumps(costs, indent=2))
    
    console.print(f"[green]Saved run metadata to {output_dir}[/green]")

