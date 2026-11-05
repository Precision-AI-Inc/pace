# Skills

Claude Code skills that ship with Pace.

## adopt-pace

Adds Pace (`precisionai-pace`) logging and progress reporting to a repository so its
jobs report completion estimates. Given a training, evaluation, inference or data
pipeline repo, Claude Code surveys it, shows you a plan, makes the smallest edits
(dependency declarations, a framework callback or a `progress`/`track` wrapper around
the main loop, one `root` scope), runs a short smoke test and checks the output with
`verify.py`.

For jobs that run on OrchAI, the same steps by hand are in
[Getting started on OrchAI](https://github.com/Precision-AI-Inc/pace/blob/main/docs/orchai.md).

### Install

The repository is public, so a shallow clone is enough. For you only, in every
repository:

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src"
mkdir -p ~/.claude/skills && rm -rf ~/.claude/skills/adopt-pace
cp -r "$src/skills/adopt-pace" ~/.claude/skills/ && rm -rf "$src"
```

Or in one repository, shared with everyone who clones it (commit the folder), from the
repository's root:

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src"
mkdir -p .claude/skills && rm -rf .claude/skills/adopt-pace
cp -r "$src/skills/adopt-pace" .claude/skills/ && rm -rf "$src"
```

Run the same commands again to update it. Then open Claude Code in the repository and ask:

> add Pace to this repo

Claude Code reads `SKILL.md`, shows its plan and asks before editing. The `patterns/`
folder (one small file of snippets per situation, read only when the repository needs it)
and `verify.py` (the output checker) sit beside it; copy the whole folder.

### Check a run yourself

`verify.py` needs only the standard library and `precisionai-pace`:

```bash
python ~/.claude/skills/adopt-pace/verify.py run.log
my-job --smoke 2>&1 | python ~/.claude/skills/adopt-pace/verify.py
```

Installed in the repository instead, it is `.claude/skills/adopt-pace/verify.py`.

It reports every scope, which one is the root, missing totals or `end` records, plan
names that match no scope, and what the estimator concludes at the end of the log. The
exit status is 0 when nothing is wrong, 1 on an error (or any warning with `--strict`),
and 2 when there was nothing to read.
