#!/usr/bin/env python3
"""
Standalone test script for Codex CLI PTY interaction.

Run this locally to debug the PTY interaction before testing in CI.

Usage:
    python test_codex_pty.py
    
Requirements:
    - codex CLI installed: npm install -g @openai/codex
    - OPENAI_API_KEY environment variable set
"""

import os
import sys
import time
import subprocess
import select
import pty
import tty
import termios

# Simple test prompt (use a unique token so we can detect a real model response)
TEST_PROMPT = "Reply with exactly: OK_FROM_CODEX_12345"

# How long to wait for Codex to be ready (seconds)
READY_TIMEOUT = 30

# Total timeout (seconds)
TOTAL_TIMEOUT = 120


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes for cleaner logging."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def run_codex_with_pty():
    """Run codex attached to a PTY and send prompt interactively."""
    
    # Build command
    cmd = [
        'codex',
        '--dangerously-bypass-approvals-and-sandbox',
        '--model',
        'gpt-4.1',
        '--approval-mode',
        'full-auto',
    ]
    
    print(f"[TEST] Command: {' '.join(cmd)}")
    print(f"[TEST] Prompt: {TEST_PROMPT}")
    print(f"[TEST] OPENAI_API_KEY set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'NO!'}")
    print()
    
    # Set up environment
    env = os.environ.copy()
    env['CODEX_UNSAFE_ALLOW_NO_SANDBOX'] = '1'
    env['TERM'] = 'xterm-256color'
    env['EDITOR'] = '/usr/bin/true'
    env['VISUAL'] = '/usr/bin/true'
    env['GIT_EDITOR'] = '/usr/bin/true'
    
    # Create PTY
    master_fd, slave_fd = pty.openpty()
    try:
        tty.setraw(slave_fd, when=termios.TCSANOW)
    except Exception:
        pass
    
    start = time.time()
    output = bytearray()
    recent = bytearray()
    prompt_sent = False
    
    print("[TEST] Starting Codex with PTY...")
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
    except FileNotFoundError:
        print("[ERROR] codex not found. Install with: npm install -g @openai/codex")
        return False
    finally:
        os.close(slave_fd)
    
    try:
        while True:
            elapsed = time.time() - start
            
            # Timeout handling
            if elapsed > TOTAL_TIMEOUT:
                print(f"\n[TEST] TIMEOUT after {elapsed:.0f}s")
                proc.kill()
                break
            
            # Check if process exited
            exited = proc.poll() is not None
            
            # Read from PTY
            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in rlist:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""
                
                if chunk:
                    output.extend(chunk)
                    recent.extend(chunk)
                    if len(recent) > 2048:
                        recent = recent[-2048:]
                    
                    # Print chunk for debugging
                    decoded = chunk.decode("utf-8", errors="replace")
                    clean = strip_ansi(decoded)
                    if clean.strip():
                        for line in clean.strip().split('\n'):
                            if line.strip():
                                print(f"[CODEX] {line.strip()}")
                    
                    # Respond to cursor position queries (ESC[6n)
                    if b"\x1b[6n" in recent:
                        print("[TEST] Responding to cursor position query")
                        try:
                            os.write(master_fd, b"\x1b[1;1R")
                        except Exception:
                            pass
                        recent = recent.replace(b"\x1b[6n", b"")
                    
                    # Check if Codex is ready for input
                    # Use the STRIPPED text for detection (no ANSI codes)
                    recent_str = strip_ansi(recent.decode("utf-8", errors="replace")).lower()
                    ready_indicators = [
                        "send a message",
                        "what would you like",
                        "how can i help",
                        "type a message",
                    ]
                    
                    if not prompt_sent:
                        for indicator in ready_indicators:
                            if indicator in recent_str:
                                print(f"\n[TEST] Detected ready indicator: '{indicator}'")
                                time.sleep(1.0)  # Give UI time to fully render
                                
                                print(f"[TEST] Sending prompt: {TEST_PROMPT}")
                                # Clear any suggestion selection / partial input first.
                                os.write(master_fd, b"\x1b")   # ESC
                                os.write(master_fd, b"\x15")   # Ctrl+U (clear line)
                                time.sleep(0.1)
                                os.write(master_fd, TEST_PROMPT.encode("utf-8"))
                                # Give Ink time to insert the text into the input box before Enter
                                time.sleep(1.0)
                                
                                print("[TEST] Pressing Enter to submit...")
                                # Use CRLF + LF to behave like a real terminal Enter.
                                os.write(master_fd, b"\r\n")
                                time.sleep(0.15)
                                os.write(master_fd, b"\n")
                                prompt_sent = True
                                break
                        
                        # Fallback: if we've been running for a while and have output, try sending
                        if not prompt_sent and elapsed > READY_TIMEOUT and len(output) > 500:
                            print(f"\n[TEST] Fallback: sending prompt after {elapsed:.0f}s")
                            os.write(master_fd, TEST_PROMPT.encode("utf-8"))
                            time.sleep(0.3)
                            os.write(master_fd, b"\r")
                            prompt_sent = True
                    
                    continue
            
            if exited:
                print(f"\n[TEST] Process exited with code {proc.returncode}")
                break
        
        # Print summary
        print("\n" + "="*60)
        print("[TEST] SUMMARY")
        print("="*60)
        print(f"  Duration: {time.time() - start:.1f}s")
        print(f"  Prompt sent: {prompt_sent}")
        print(f"  Exit code: {proc.returncode}")
        print(f"  Output length: {len(output)} bytes")
        
        # Check if we got a response
        output_str = output.decode("utf-8", errors="replace").lower()
        if "ok_from_codex_12345" in output_str:
            print("  Response: SUCCESS - Got OK_FROM_CODEX_12345")
            return True
        else:
            print("  Response: No OK_FROM_CODEX_12345 found in output")
            return False
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted")
        proc.kill()
        return False
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass


if __name__ == "__main__":
    print("="*60)
    print("  Codex CLI PTY Test")
    print("="*60)
    print()
    
    success = run_codex_with_pty()
    
    print()
    if success:
        print("[RESULT] ✓ Test PASSED")
        sys.exit(0)
    else:
        print("[RESULT] ✗ Test FAILED")
        sys.exit(1)

