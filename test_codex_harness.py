#!/usr/bin/env python3
"""
Test that mimics exactly what the harness will do.
Uses --quiet mode which we confirmed works!
"""
import os
import subprocess
import sys

def main():
    prompt = """You are solving a software engineering task. Fix the following issue in this codebase.

## Problem Statement

This is a test. Just say "TEST_PASSED_12345" and nothing else.

## Instructions

1. Just output TEST_PASSED_12345
"""

    # Build command exactly like the harness
    # Use npx to ensure we get the npm @openai/codex package
    cmd = [
        'npx', '--yes', '@openai/codex',
        '-q',  # Non-interactive mode! (short form)
        '--dangerously-bypass-approvals-and-sandbox',
        '--model', 'gpt-4.1',
        '--approval-mode', 'full-auto',
        prompt,  # Prompt as positional argument
    ]
    
    print(f"[TEST] Running codex in quiet mode...")
    print(f"[TEST] OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'NO!'}")
    print()
    
    env = os.environ.copy()
    env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
    env['EDITOR'] = '/usr/bin/true'
    env['VISUAL'] = '/usr/bin/true'
    env['GIT_EDITOR'] = '/usr/bin/true'
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        print(f"[TEST] Exit code: {result.returncode}")
        print()
        print("[STDOUT]:")
        print(result.stdout)
        print()
        if result.stderr:
            print("[STDERR]:")
            print(result.stderr)
        
        if result.returncode == 0:
            print()
            print("[RESULT] ✓ SUCCESS - Codex ran successfully in quiet mode!")
            return 0
        else:
            print()
            print("[RESULT] ✗ FAILED - Non-zero exit code")
            return 1
            
    except subprocess.TimeoutExpired:
        print("[TEST] TIMEOUT after 120s")
        return 1
    except FileNotFoundError:
        print("[ERROR] codex not found")
        return 1

if __name__ == "__main__":
    sys.exit(main())

