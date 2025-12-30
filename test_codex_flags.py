#!/usr/bin/env python3
"""
Test different ways to pass a prompt to codex CLI.

Run each test to see which method works.
"""
import os
import subprocess
import sys

def run_test(name: str, cmd: list, input_text: str = None):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    if input_text:
        print(f"Stdin: {input_text[:50]}...")
    print()
    
    env = os.environ.copy()
    env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        print(f"Exit code: {result.returncode}")
        if result.stderr and "error" in result.stderr.lower():
            print(f"STDERR: {result.stderr[:500]}")
        if result.stdout:
            print(f"STDOUT (first 500 chars): {result.stdout[:500]}")
            
        return result.returncode == 0 and ("TEST_OK" in result.stdout or "TEST_OK" in result.stderr)
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return False
    except FileNotFoundError:
        print("codex not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    prompt = "Reply with exactly: TEST_OK_12345"
    
    tests = [
        # Test 1: Positional argument
        ("Positional arg", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto",
            "--model", "gpt-4.1",
            prompt,
        ], None),
        
        # Test 2: Using -p flag
        ("Using -p flag", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox", 
            "--approval-mode", "full-auto",
            "--model", "gpt-4.1",
            "-p", prompt,
        ], None),
        
        # Test 3: Using --prompt flag
        ("Using --prompt flag", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto", 
            "--model", "gpt-4.1",
            "--prompt", prompt,
        ], None),
        
        # Test 4: Stdin with --quiet
        ("Stdin + --quiet", [
            "codex",
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto",
            "--model", "gpt-4.1",
            "--quiet",
        ], prompt + "\n"),
        
        # Test 5: Just stdin
        ("Just stdin", [
            "codex", 
            "--dangerously-bypass-approvals-and-sandbox",
            "--approval-mode", "full-auto",
            "--model", "gpt-4.1",
        ], prompt + "\n"),
    ]
    
    results = []
    for name, cmd, stdin in tests:
        success = run_test(name, cmd, stdin)
        results.append((name, success))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    return 0 if any(s for _, s in results) else 1

if __name__ == "__main__":
    sys.exit(main())

