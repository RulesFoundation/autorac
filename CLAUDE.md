# AutoRAC

AI-assisted RAC encoding infrastructure. Provides the feedback loop for automated statute encoding.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          AutoRAC                                 │
├─────────────────────────────────────────────────────────────────┤
│  Encoder Harness                                                │
│    ├── encode() → .rac file                                     │
│    ├── predict_scores() → {rac: 8, formula: 7, param: 9}        │
│    └── suggest_improvements() → "add nesting pattern to docs"   │
├─────────────────────────────────────────────────────────────────┤
│  Validator Pipeline (parallel)                                   │
│    ├── CI (parse, lint, inline tests)                           │
│    ├── Reviewer agents (rac, formula, param, integration)       │
│    └── External oracles (PolicyEngine, TAXSIM)                  │
├─────────────────────────────────────────────────────────────────┤
│  Experiment DB                                                   │
│    ├── encoding_id, file, timestamp                             │
│    ├── predicted_scores, actual_scores                          │
│    ├── prediction_error (for calibration)                       │
│    └── agent_suggestions                                        │
├─────────────────────────────────────────────────────────────────┤
│  Calibration Metrics                                             │
│    ├── MSE, MAE, bias per metric                                │
│    ├── Trend analysis over time                                 │
│    └── Confidence calibration                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

- `src/harness/experiment_db.py` - SQLite experiment logging
- `src/harness/validator_pipeline.py` - Parallel validator execution
- `src/harness/encoder_harness.py` - Wraps encoder agent with prediction
- `src/harness/metrics.py` - Calibration computation

## Usage

```python
from autorac import EncoderHarness, EncoderConfig
from pathlib import Path

config = EncoderConfig(
    rac_us_path=Path("../rac-us"),
    rac_path=Path("../rac"),
)

harness = EncoderHarness(config)
run, result = harness.encode_with_feedback(
    citation="26 USC 1(h)(1)(E)",
    statute_text="...",
    output_path=Path("rac-us/statute/26/1/h/1/E.rac"),
)

print(f"Predicted: {run.predicted.rac_reviewer}")
print(f"Actual: {run.actual.rac_reviewer}")
print(f"All passed: {result.all_passed}")
```

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run tests
pytest tests/ -v

# View calibration
python -m autorac.metrics --db experiments.db
```

## Related Repos

- **rac** - DSL parser, executor, runtime
- **rac-us** - US statute encodings
- **rac-validators** - External calculator validation (PolicyEngine, TAXSIM)
