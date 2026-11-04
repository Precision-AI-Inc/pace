# Unknown totals, files and bytes

Find a real denominator before settling for "unknown": `len(dataset)`, a `wc -l`, the
number of files from a glob, `os.path.getsize`, the count a manifest holds:

```python
files = sorted(Path(args.input).glob("**/*.jsonl"))
with progress(total=len(files), description="Convert", unit="file", root=True) as run:
    for path in files:
        convert(path)
        run.advance()
```

```python
size = os.path.getsize(path)
with progress(total=size, description="Download", unit="byte") as run:
    for chunk in stream:
        out.write(chunk)
        run.advance(len(chunk))
```

- If the total only becomes known later (the first pass counts), start with `total=None`
  and call `run.set_total(n)` once it is known.
- A scope that never has a total is reported as unknown and the scheduler gets no estimate
  from it. Do not invent a total to make a loop estimable: an honest "unknown" beats a wrong
  number. List it as a remaining gap in the report.

## Examples

- Pace: [`unknown_total.py`](https://github.com/Precision-AI-Inc/pace/blob/main/examples/unknown_total.py)
  (`set_total` when the total appears).
