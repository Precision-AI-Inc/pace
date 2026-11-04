# Standard `logging`

Optional, and only when the user wants one log format. Call it once at startup:

```python
from precisionai.pace import capture_logging

capture_logging("INFO")
```

It routes the root logger's records (so PyTorch's, Transformers' and the repo's own)
through Pace's formatter. If the repo already called `logging.basicConfig(...)` or added a
`StreamHandler` to the root logger, every record would be written twice: remove that
handler in the same change, or skip `capture_logging()`.

Lightning and Transformers print through loggers of their own, so their lines would come
out twice or not reach Pace. When the repo uses them, name them, **after** their imports:

```python
import lightning.pytorch as pl

from precisionai.pace import capture_logging

capture_logging("INFO", take_over=["lightning", "transformers"])
```

(`"pytorch_lightning"` for the older package.) Do not hand-clear their handlers.

vLLM is different: set `VLLM_CONFIGURE_LOGGING=0` in the environment before vLLM is
imported, because its engine runs in another process that configures logging itself.

Loguru and `print` stay as they are. Pace does not bridge Loguru.

## Examples

- Pace: [`stdlib_logging.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/stdlib_logging.py),
  [`library_hooks.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/library_hooks.py)
  (`take_over` with a stand-in library).
- Tested: [`llm/generate_vllm.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/generate_vllm.py)
  (vLLM's records through Pace, image sets `VLLM_CONFIGURE_LOGGING=0`) and
  [`detection/train_lightning.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/detection/train_lightning.py)
  in OrchAI's example repo.
