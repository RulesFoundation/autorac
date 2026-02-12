# AutoRAC

AI-assisted RAC encoding infrastructure. Provides the feedback loop for automated statute encoding.

## Two Encoding Approaches

AutoRAC supports two ways to encode statutes:

### 1. Interactive (Claude Code Plugin) - Recommended

Uses Claude Code with Max subscription. No API billing.

```
┌─────────────────────────────────────────────────────────────┐
│  Claude Code Session (Max subscription)                      │
│    └── /encode "26 USC 32"                                  │
│          └── Task(rac:RAC Encoder)                          │
│                └── Write, Edit, Grep tools                  │
│                      └── rac-us/statute/26/32.rac           │
└─────────────────────────────────────────────────────────────┘
```

**How to use:**
1. Install rac-claude plugin
2. Run `/encode "26 USC 32"` in Claude Code
3. Agent encodes, validates, logs journey

### 2. Programmatic (Agent SDK) - For Batch/Parallel

Uses Claude Agent SDK with API key. Pay per token, but enables massive parallelization.

```
┌─────────────────────────────────────────────────────────────┐
│  Python Script / CI Pipeline                                 │
│    └── AgentSDKBackend(api_key=...)                         │
│          └── encode_batch(requests, max_concurrent=10)      │
│                └── 10 parallel encoding agents              │
│                      └── 10x faster for batch jobs          │
└─────────────────────────────────────────────────────────────┘
```

**How to use:**
```python
from autorac import AgentSDKBackend, EncoderRequest
from pathlib import Path

backend = AgentSDKBackend()  # Requires ANTHROPIC_API_KEY

# Encode 50 statutes in parallel
requests = [
    EncoderRequest(
        citation=f"26 USC {section}",
        statute_text=texts[section],
        output_path=Path(f"rac-us/statute/26/{section}.rac"),
    )
    for section in sections
]

responses = await backend.encode_batch(requests, max_concurrent=10)
```

**Install:** `pip install autorac[sdk]`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          AutoRAC                                 │
├─────────────────────────────────────────────────────────────────┤
│  Encoder Backends                                                │
│    ├── ClaudeCodeBackend (subprocess, Max subscription)         │
│    └── AgentSDKBackend (API, parallelization)                   │
├─────────────────────────────────────────────────────────────────┤
│  3-Tier Validator Pipeline (tiers run in order)                  │
│    ├── Tier 1: CI (rac pytest) - instant, catches syntax        │
│    ├── Tier 2: Oracles (PE/TAXSIM) - fast ~10s, comparison data │
│    └── Tier 3: LLM reviewers - uses oracle context to diagnose  │
├─────────────────────────────────────────────────────────────────┤
│  Experiment DB                                                   │
│    ├── encoding_id, file, timestamp                             │
│    ├── iterations, errors, fixes                                │
│    └── final_scores, session_transcript                         │
├─────────────────────────────────────────────────────────────────┤
│  Calibration Metrics                                             │
│    ├── MSE, MAE, bias per metric                                │
│    ├── Trend analysis over time                                 │
│    └── Confidence calibration                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 3-Tier Validation

Validation runs in order with oracle results feeding into LLM reviewers:

| Tier | What | Time | Cost | Purpose |
|------|------|------|------|---------|
| 1. CI | rac pytest | instant | free | Catches syntax/format errors |
| 2. Oracles | PE + TAXSIM | ~10s | free | Generates comparison data |
| 3. LLM Reviewers | 4 specialists | ~30s | API cost | Diagnoses WHY discrepancies exist |

Oracles run before LLM reviewers because they're fast/free and provide essential
diagnostic context. LLMs can then investigate *why* encodings differ from consensus.

## Components

- `src/harness/backends.py` - Encoder backends (ClaudeCode, AgentSDK)
- `src/harness/experiment_db.py` - SQLite experiment logging
- `src/harness/validator_pipeline.py` - Parallel validator execution
- `src/harness/encoder_harness.py` - Wraps encoder with prediction
- `src/harness/metrics.py` - Calibration computation

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .          # CLI backend only
pip install -e ".[sdk]"   # With Agent SDK for parallel encoding

# Run tests
pytest tests/ -v

# View calibration
python -m autorac.metrics --db experiments.db
```

## Related Repos

- **rac** - DSL parser, executor, runtime
- **rac-us** - US statute encodings
- **rac-validators** - External calculator validation (PolicyEngine, TAXSIM)
- **rac-claude** - Claude Code plugin with /encode command
