# Skills

Coding-agent skills that ship with Pace, for anyone adding Pace to their own repository.
They are written for Claude Code and work with other agents too (see below).

## adopt-pace

Adds Pace (`precisionai-pace`) logging and progress reporting to a repository so its
jobs report completion estimates. Given a training, evaluation, inference or data
pipeline repo, Claude Code surveys it, shows you a plan, makes the smallest edits
(dependency declarations, a framework callback or a `progress`/`track` wrapper around
the main loop, one `root` scope), runs a short smoke test and checks the output with
`verify.py`.

The same steps by hand, for a job that runs in a container or under a scheduler, are in
[Running Pace on a cluster or scheduler](https://github.com/Precision-AI-Inc/pace/blob/main/docs/orchai.md).

### Install

The repository is public, so a shallow clone is enough. For you only, in every
repository:

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src" &&
  mkdir -p ~/.claude/skills && rm -rf ~/.claude/skills/adopt-pace &&
  cp -r "$src/skills/adopt-pace" ~/.claude/skills/ && rm -rf "$src"
```

Or in one repository, shared with everyone who clones it (commit the folder), from the
repository's root:

```bash
src=$(mktemp -d) && git clone --depth 1 https://github.com/Precision-AI-Inc/pace.git "$src" &&
  mkdir -p .claude/skills && rm -rf .claude/skills/adopt-pace &&
  cp -r "$src/skills/adopt-pace" .claude/skills/ && rm -rf "$src"
```

Run the same command again to update it: it removes the old copy first, so no stale
files are left behind. Then open Claude Code in the repository and ask:

> add Pace to this repo

Claude Code reads `SKILL.md`, shows its plan and asks before editing. The `patterns/`
folder (one small file of snippets per situation, read only when the repository needs it)
and `verify.py` (the output checker) sit beside it; copy the whole folder.

### Other coding agents

`SKILL.md` is plain Markdown with a short front-matter header, so any coding agent that
can read files and run commands can follow it. Copy the folder into the repository as
above (any path works), then ask the agent to "follow `<path>/adopt-pace/SKILL.md` to add
Pace to this repo". To make it permanent, add that line to the repository's own agent
instructions (`AGENTS.md` or its equivalent). If your agent supports skills in the same
`SKILL.md` format, install the folder wherever it looks for them.

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
