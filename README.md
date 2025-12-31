# AutoRAC

AI-assisted RAC encoding infrastructure. Provides the feedback loop for automated statute encoding.

## Installation

```bash
pip install -e ".[dev]"
```

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
    citation="26 USC 32(a)(1)",
    statute_text="...",
    output_path=Path("rac-us/statute/26/32/a/1.rac"),
)
```
