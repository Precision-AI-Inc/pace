# A library call that takes a `tqdm` class (vLLM)

`LLM.generate(prompts, params)` and `LLM.chat(...)` run every prompt inside one call, so
nothing written around the call reports progress until it returns. Both accept a `tqdm`
class in `use_tqdm`; give them Pace's:

```python
from precisionai.pace.tqdm import PaceTqdm

outputs = llm.generate(prompts, params, use_tqdm=PaceTqdm)
```

- No dependency is added: `PaceTqdm` is part of `precisionai-pace` and does not import `tqdm`.
- vLLM opens two bars per call: `Rendering prompts` (it ends first), then
  `Processed prompts`, the one that matters. Neither is a root by default. When one
  `generate` or `chat` call is the whole job, mark the second as the root:

  ```python
  class Generation(PaceTqdm):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, root=kwargs.get("desc") == "Processed prompts", **kwargs)
  ```

  Never mark both: the first root decides, and its end would complete the job.
- Each call opens new scopes (`processed-prompts`, `processed-prompts-2`, ...). A job that
  calls it in a loop over batches should wrap the loop in one `progress(..., root=True)`
  instead, advance by the chunk, and pass `use_tqdm=False`:

  ```python
  with progress(total=len(prompts), description="Generate", unit="prompt", root=True) as run:
      for start in range(0, len(prompts), args.chunk):
          chunk = prompts[start : start + args.chunk]
          answers += llm.chat(chunk, sampling, use_tqdm=False)
          run.advance(len(chunk))
  ```

  Progress is then reported a chunk at a time; a smaller chunk gives a smoother estimate.
- vLLM's own log lines are separate: see `logging.md` (`VLLM_CONFIGURE_LOGGING=0`).

## Examples

- Pace: [`library_hooks.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/library_hooks.py)
  (`PaceTqdm` with a stand-in library).
- Tested, in OrchAI's example repo, with chunked progress:
  [`llm/generate_vllm.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/llm/generate_vllm.py)
  and, for a vision-language model with `track` over chunks of images,
  [`vlm/caption_vllm.py`](https://github.com/Precision-AI-Inc/orchai-examples/blob/main/vlm/caption_vllm.py).
