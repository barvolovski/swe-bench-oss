#!/usr/bin/env python3
"""
Codex CLI harness for SWE-Bench.

Runs OpenAI's Codex CLI on SWE-Bench tasks and collects results.
"""
import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

def _run_with_pty(cmd: List[str], cwd: Path, env: Dict[str, str], timeout_seconds: int, prompt: str) -> tuple[int, str]:
    """
    Run Codex CLI attached to a pseudo-terminal (PTY), send prompt interactively, and capture output.

    Why:
    - `codex` uses Ink (terminal UI) which expects a TTY and also sends cursor-position
      queries (ESC[6n). In CI there is no real terminal emulator to answer those, so
      Ink times out with:
        "Error: The cursor position could not be read within a normal duration"
    - By running the process on a PTY and replying with a dummy cursor position
      response (ESC[1;1R), we satisfy Ink without needing a real terminal emulator.
    - Codex CLI waits for interactive input, so we detect when it's ready and send
      the prompt via stdin, then press Enter to submit.
    """
    import pty
    import select

    master_fd, slave_fd = pty.openpty()
    start = time.time()
    output = bytearray()
    recent = bytearray()
    prompt_sent = False

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
    finally:
        try:
            os.close(slave_fd)
        except Exception:
            pass

    try:
        while True:
            # Timeout handling
            if time.time() - start > timeout_seconds:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

            # If process exited, drain remaining output then break
            exited = proc.poll() is not None

            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in rlist:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""

                if chunk:
                    output.extend(chunk)

                    # Keep a small rolling buffer to detect patterns
                    recent.extend(chunk)
                    if len(recent) > 2048:
                        recent = recent[-2048:]

                    # Respond to cursor position queries (ESC[6n)
                    if b"\x1b[6n" in recent:
                        try:
                            os.write(master_fd, b"\x1b[1;1R")
                        except Exception:
                            pass
                        recent = recent.replace(b"\x1b[6n", b"")

                    # Check if Codex is ready for input (look for common prompts)
                    # Codex shows "send a message" or similar when ready
                    recent_str = recent.decode("utf-8", errors="replace").lower()
                    if not prompt_sent and ("send a message" in recent_str or 
                                            "what would you like" in recent_str or
                                            "how can i help" in recent_str or
                                            # Also check for a simple prompt indicator after UI init
                                            (time.time() - start > 3 and len(output) > 100)):
                        # Give Codex a moment to fully initialize
                        time.sleep(0.5)
                        try:
                            # Send the prompt
                            os.write(master_fd, prompt.encode("utf-8"))
                            time.sleep(0.2)
                            # Press Enter to submit
                            os.write(master_fd, b"\r")
                            prompt_sent = True
                            console.print(f"[green]Prompt sent ({len(prompt)} chars)[/green]")
                        except Exception as e:
                            console.print(f"[red]Failed to send prompt: {e}[/red]")

                    continue

            if exited:
                break

        returncode = proc.wait(timeout=5)
        return returncode, output.decode("utf-8", errors="replace")
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        'model': 'gpt-4.1',
        'approval_mode': 'full-auto',
        'timeout_minutes': 30,
        'cli_flags': ['--full-auto', '--dangerously-auto-approve-everything'],
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

    # Build command
    cmd = ['codex']
    
    # Add CLI flags from config
    for flag in config.get('cli_flags', []):
        cmd.append(flag)
    
    # Add model
    cmd.extend(['--model', model])
    
    # Don't pass prompt as CLI arg - we'll send it interactively via PTY
    # (Codex CLI ignores CLI prompt when in interactive/Ink mode)
    
    console.print(f"[blue]Running Codex CLI (PTY mode)...[/blue]")
    
    try:
        # Set up environment
        env = os.environ.copy()
        env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
        # TERM can remain "xterm" style; we respond to cursor queries ourselves.
        env.setdefault('TERM', 'xterm-256color')
        
        # Prevent interactive editors
        env['EDITOR'] = '/usr/bin/true'
        env['VISUAL'] = '/usr/bin/true'
        env['GIT_EDITOR'] = '/usr/bin/true'
        
        # Add any environment variables from config
        for key, value in config.get('environment', {}).items():
            env[key] = value
        
        # Run Codex CLI - prompt is sent interactively after Codex is ready
        timeout = config.get('timeout_minutes', 30) * 60
        returncode, output = _run_with_pty(cmd=cmd, cwd=workspace, env=env, timeout_seconds=timeout, prompt=prompt)
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

