#!/usr/bin/env python3
"""
Simple test: run codex with a prompt passed as CLI argument.

This tests whether codex can accept a prompt directly on the command line
(which should work in full-auto mode without needing interactive input).
"""
import os
import subprocess
import sys

def main():
    # The prompt to send
    prompt = "Say exactly: HELLO_WORLD_123"
    
    # Build command - try passing prompt as positional arg
    cmd = [
        "codex",
        "--dangerously-bypass-approvals-and-sandbox",
        "--approval-mode", "full-auto", 
        "--model", "gpt-4.1",
        prompt,  # Prompt as positional argument
    ]
    
    print(f"[TEST] Command: {' '.join(cmd)}")
    print(f"[TEST] OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'NO!'}")
    print()
    
    env = os.environ.copy()
    env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
    
    print("[TEST] Running codex...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        print("-" * 60)
        print(f"[TEST] Exit code: {result.returncode}")
        print()
        print("[STDOUT]:")
        print(result.stdout[:2000] if result.stdout else "(empty)")
        print()
        print("[STDERR]:")
        print(result.stderr[:2000] if result.stderr else "(empty)")
        
        if "HELLO_WORLD_123" in result.stdout or "HELLO_WORLD_123" in result.stderr:
            print()
            print("[RESULT] ✓ SUCCESS - Got expected response!")
            return 0
        else:
            print()
            print("[RESULT] ✗ FAILED - No expected response found")
            return 1
            
    except subprocess.TimeoutExpired:
        print("[TEST] TIMEOUT after 120s")
        return 1
    except FileNotFoundError:
        print("[ERROR] codex not found. Install with: npm install -g @openai/codex")
        return 1

if __name__ == "__main__":
    sys.exit(main())

