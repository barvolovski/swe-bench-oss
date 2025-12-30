#!/usr/bin/env python3
"""
Test codex with exec subcommand and other non-interactive approaches.
"""
import os
import subprocess
import sys

def run_test(name: str, cmd: list, timeout: int = 30):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    env = os.environ.copy()
    env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
    env['CI'] = '1'  # Maybe it has CI mode
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        print(f"Exit code: {result.returncode}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr[:1000]}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout[:1000]}")
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return -1
    except FileNotFoundError:
        print("codex not found")
        return -2
    except Exception as e:
        print(f"Error: {e}")
        return -3

def main():
    prompt = "Say hello"
    
    # First, let's see what subcommands/flags are available
    print("Checking codex help...")
    run_test("codex --help", ["codex", "--help"], timeout=10)
    
    tests = [
        # Try exec subcommand
        ("codex exec", [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model", "gpt-4.1",
            prompt,
        ]),
        
        # Try with -q/--quiet flag
        ("codex --quiet", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto",
            "--model", "gpt-4.1",
            "--quiet",
            prompt,
        ]),
        
        # Try with --no-interactive or similar
        ("codex --non-interactive", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto", 
            "--model", "gpt-4.1",
            "--non-interactive",
            prompt,
        ]),
        
        # Try with just the bare minimum
        ("codex minimal", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            prompt,
        ]),
    ]
    
    for name, cmd in tests:
        run_test(name, cmd, timeout=30)

if __name__ == "__main__":
    main()

