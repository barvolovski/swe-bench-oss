# SWE-Bench OSS Harness

Open-source harness for running AI coding agents (Codex CLI, Claude Code) on SWE-Bench benchmarks via GitHub Actions.

## Overview

This repository provides:
- **GitHub Actions workflows** for automated SWE-Bench runs
- **Harness scripts** for Codex CLI and Claude Code
- **Evaluation pipeline** compatible with SWE-Bench Pro/Lite/Verified

## Supported Agents

| Agent | Installation | Description |
|-------|-------------|-------------|
| `codex-cli` | `npm install -g @openai/codex` | OpenAI's Codex CLI agent |
| `claude-code` | `npm install -g @anthropic-ai/claude-code` | Anthropic's Claude Code agent |

## Quick Start

### 1. Fork this repository

### 2. Set up secrets

Add the following secrets to your GitHub repository:

| Secret | Required For | Description |
|--------|-------------|-------------|
| `OPENAI_API_KEY` | codex-cli | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | claude-code | Your Anthropic API key |

### 3. Run via GitHub Actions

Go to **Actions** → Select workflow → **Run workflow**

You can specify:
- `dataset`: `pro`, `lite`, or `verified`
- `task_ids`: Comma-separated instance IDs (or leave empty for all)
- `max_tasks`: Maximum number of tasks to run
- `model`: Model to use (e.g., `gpt-4.1`, `claude-sonnet-4-20250514`)

## Manual Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run Codex CLI on a single task
python harness/run_codex.py \
  --task-id "instance_navidrome__navidrome-7073d18b54da7e53274d11c9e2baef1242e8769e" \
  --output ./preds

# Run Claude Code on a single task
python harness/run_claude.py \
  --task-id "instance_navidrome__navidrome-7073d18b54da7e53274d11c9e2baef1242e8769e" \
  --output ./preds

# Evaluate results
python eval/run_eval.py \
  --preds ./preds \
  --dataset pro \
  --output ./eval_results
```

## Directory Structure

```
swe-bench-oss/
├── .github/workflows/     # GitHub Actions workflows
│   ├── codex-cli.yml      # Codex CLI benchmark workflow
│   ├── claude-code.yml    # Claude Code benchmark workflow
│   └── evaluate.yml       # Evaluation workflow
├── harness/               # Agent harness scripts
│   ├── run_codex.py       # Codex CLI runner
│   ├── run_claude.py      # Claude Code runner
│   └── utils.py           # Shared utilities
├── eval/                  # Evaluation scripts
│   └── run_eval.py        # SWE-Bench evaluation
├── configs/               # Agent configurations
│   ├── codex-cli.json     # Codex CLI settings
│   └── claude-code.json   # Claude Code settings
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

### Codex CLI (`configs/codex-cli.json`)

```json
{
  "model": "gpt-4.1",
  "approval_mode": "full-auto",
  "sandbox": "docker"
}
```

### Claude Code (`configs/claude-code.json`)

```json
{
  "model": "claude-sonnet-4-20250514",
  "permission_mode": "bypassPermissions",
  "dangerously_skip_permissions": true
}
```

## Evaluation

Results are automatically evaluated against SWE-Bench gold patches using the official evaluation harness. Results include:
- Pass/fail status per instance
- Generated patches
- Cost and token usage metrics

## License

MIT License - see [LICENSE](LICENSE) for details.

