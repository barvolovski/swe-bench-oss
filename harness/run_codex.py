#!/usr/bin/env python3
"""
Codex CLI harness for SWE-Bench.

Runs OpenAI's Codex CLI on SWE-Bench tasks and collects results.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def _normalize_openai_api_key(raw: Optional[str]) -> str:
    """
    Normalize OPENAI_API_KEY from CI secrets.

    We intentionally avoid printing the key anywhere. This function:
    - Strips whitespace/newlines (common when secrets are copy-pasted)
    - Removes a leading 'Bearer ' prefix if the user pasted a full Authorization header
    - Strips surrounding quotes
    """
    if not raw:
        return ""
    key = raw.strip()
    if key.lower().startswith("bearer "):
        key = key.split(" ", 1)[1].strip()
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    return key

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils import (
    TaskResult,
    RunMetadata,
    gather_diff,
    load_tasks,
    save_prediction,
    save_run_metadata,
    setup_workspace_simple,
)

console = Console()

def _run_codex_quiet(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_seconds: int,
) -> tuple[int, str]:
    """
    Run Codex CLI in --quiet mode (non-interactive, JSON output).
    
    This is the correct way to run codex in CI/automation:
    - --quiet flag makes it non-interactive
    - Output is JSON lines with the conversation
    - No TTY required!
    """
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    
    # Combine stdout and stderr
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    
    return result.returncode, output


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        'model': 'gpt-4.1',
        'approval_mode': 'full-auto',
        'timeout_minutes': 30,
        'cli_flags': [],
    }


def run_codex_cli(
    workspace: Path,
    problem_statement: str,
    model: str,
    config: Dict[str, Any],
) -> TaskResult:
    """
    Run Codex CLI on a workspace.
    
    Args:
        workspace: Path to the workspace
        problem_statement: The problem to solve
        model: Model to use
        config: Configuration dictionary
        
    Returns:
        TaskResult with patch and metrics
    """
    start_time = time.time()
    
    # Build the prompt
    prompt = f"""You are solving a software engineering task. Fix the following issue in this codebase.

## Problem Statement

{problem_statement}

## Instructions

1. Analyze the codebase to understand the issue
2. Make the necessary code changes to fix the problem
3. Do NOT run tests or build commands - just make the code changes
4. When done, the changes will be collected as a git diff
"""

    # Build command using codex exec subcommand for non-interactive mode
    # Based on CI help output: "codex exec - Run Codex non-interactively"
    # NOTE: exec subcommand only accepts: --model <MODEL> [PROMPT]
    cmd = [
        'codex',
        'exec',  # Non-interactive subcommand!
        '--model', model,
        prompt,
    ]
    
    console.print(f"[blue]Running Codex CLI (exec mode)...[/blue]")
    # Debug: print actual command (truncate prompt for readability)
    debug_cmd = cmd.copy()
    debug_cmd[-1] = "<prompt truncated>"
    console.print(f"[dim]Full command: {' '.join(debug_cmd)}[/dim]")
    
    try:
        # Set up environment
        env = os.environ.copy()
        env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'

        # Normalize + validate OPENAI_API_KEY (do NOT print the value)
        raw_key = env.get("OPENAI_API_KEY")
        norm_key = _normalize_openai_api_key(raw_key)
        # Set several common env var names used by different Codex/OpenAI CLIs.
        # This is safe (same secret) and improves compatibility across versions.
        env["OPENAI_API_KEY"] = norm_key
        env.setdefault("OPENAI_API_TOKEN", norm_key)
        env.setdefault("OPENAI_KEY", norm_key)
        env.setdefault("CODEX_API_KEY", norm_key)
        env.setdefault("CODEX_OPENAI_API_KEY", norm_key)

        # Ensure Codex has a writable config directory (some builds store auth here).
        # If the workflow set HOME/XDG_* already, keep those values.
        env.setdefault("HOME", os.environ.get("HOME", ""))
        if env.get("HOME"):
            env.setdefault("XDG_CONFIG_HOME", env["HOME"])
            env.setdefault("XDG_STATE_HOME", env["HOME"])

        # Debug: show safe diagnostics only
        if not norm_key:
            console.print("[red]ERROR: OPENAI_API_KEY is missing/empty after normalization[/red]")
            console.print("[red]Check GitHub Actions secret OPENAI_API_KEY (no quotes, no 'Bearer ' prefix)[/red]")
            return TaskResult(
                instance_id="",
                success=False,
                patch="",
                error="OPENAI_API_KEY missing/empty after normalization",
                duration_seconds=time.time() - start_time,
            )
        console.print(
            "[green]OPENAI_API_KEY is set[/green] "
            f"(length={len(norm_key)}, startswith_sk={norm_key.startswith('sk-')})"
        )
        
        # Prevent interactive editors from spawning
        env['EDITOR'] = '/usr/bin/true'
        env['VISUAL'] = '/usr/bin/true'
        env['GIT_EDITOR'] = '/usr/bin/true'
        
        # Add any environment variables from config
        for key, value in config.get('environment', {}).items():
            env[key] = value
        
        # Run Codex CLI in quiet mode (simple subprocess, no PTY needed!)
        timeout = config.get('timeout_minutes', 30) * 60
        returncode, output = _run_codex_quiet(cmd=cmd, cwd=workspace, env=env, timeout_seconds=timeout)
        duration = time.time() - start_time
        
        if returncode != 0:
            console.print(f"[red]Command failed with return code {returncode}[/red]")
            console.print(output)
        
        # Gather the diff
        patch = gather_diff(workspace)
        
        # Parse token usage from output if available
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        cost_usd = 0.0
        
        # Try to extract usage from output (format varies)
        for line in output.split('\n'):
            if 'input_tokens' in line.lower():
                try:
                    input_tokens = int(''.join(filter(str.isdigit, line.split(':')[-1])))
                except:
                    pass
            if 'output_tokens' in line.lower():
                try:
                    output_tokens = int(''.join(filter(str.isdigit, line.split(':')[-1])))
                except:
                    pass
        
        return TaskResult(
            instance_id="",  # Will be set by caller
            success=returncode == 0 and bool(patch),
            patch=patch,
            output=output,
            duration_seconds=duration,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        patch = gather_diff(workspace)
        return TaskResult(
            instance_id="",
            success=bool(patch),
            patch=patch,
            error="Timeout expired",
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        patch = gather_diff(workspace)
        return TaskResult(
            instance_id="",
            success=False,
            patch=patch,
            error=str(e),
            duration_seconds=duration,
        )


def main():
    parser = argparse.ArgumentParser(description='Run Codex CLI on SWE-Bench tasks')
    parser.add_argument('--task-ids', type=str, help='Comma-separated task IDs')
    parser.add_argument('--dataset', type=str, default='pro', help='Dataset (pro, lite, verified)')
    parser.add_argument('--model', type=str, default='gpt-4.1', help='Model to use')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to run')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Parse task IDs
    task_ids = None
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(',') if t.strip()]
    
    # Load tasks
    tasks = load_tasks(args.dataset, task_ids)
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    if not tasks:
        console.print("[red]No tasks to run![/red]")
        sys.exit(1)
    
    console.print(f"[green]Running {len(tasks)} tasks with Codex CLI[/green]")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create workspace root
    workspace_root = Path(tempfile.mkdtemp(prefix='swe-bench-codex-'))
    
    # Initialize run metadata
    run_id = f"codex-cli-{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}"
    metadata = RunMetadata(
        run_id=run_id,
        agent='codex-cli',
        model=args.model,
        dataset=args.dataset,
    )
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_progress = progress.add_task("Running tasks...", total=len(tasks))
            
            for i, task in enumerate(tasks):
                instance_id = task['instance_id']
                console.print(f"\n[bold]Task {i+1}/{len(tasks)}: {instance_id}[/bold]")
                
                try:
                    # Setup workspace
                    workspace = setup_workspace_simple(task, workspace_root)
                    
                    # Run Codex CLI
                    result = run_codex_cli(
                        workspace=workspace,
                        problem_statement=task['problem_statement'],
                        model=args.model,
                        config=config,
                    )
                    result.instance_id = instance_id
                    
                    # Save prediction
                    save_prediction(
                        output_dir=output_dir,
                        instance_id=instance_id,
                        patch=result.patch,
                        metadata={
                            'instance_id': instance_id,
                            'success': result.success,
                            'duration_seconds': result.duration_seconds,
                            'cost_usd': result.cost_usd,
                        }
                    )
                    
                    metadata.results.append(result)
                    
                    status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
                    console.print(f"{status} {instance_id} ({result.duration_seconds:.1f}s)")
                    
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    metadata.results.append(TaskResult(
                        instance_id=instance_id,
                        success=False,
                        error=str(e),
                    ))
                
                finally:
                    # Cleanup workspace
                    workspace_path = workspace_root / instance_id
                    if workspace_path.exists():
                        shutil.rmtree(workspace_path)
                
                progress.update(task_progress, advance=1)
        
        # Save run metadata
        save_run_metadata(output_dir, metadata)
        
        # Print summary
        successful = sum(1 for r in metadata.results if r.success)
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total: {len(metadata.results)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {len(metadata.results) - successful}")
        
    finally:
        # Cleanup workspace root
        if workspace_root.exists():
            shutil.rmtree(workspace_root)


if __name__ == '__main__':
    main()

