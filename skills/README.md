# Skills

Claude Code skills that ship with Pace.

## adopt-pace

Adds Pace (`precisionai-pace`) logging and progress reporting to a repository so its
jobs report completion estimates. Given a training, evaluation, inference or data
pipeline repo, Claude Code surveys it, shows you a plan, makes the smallest edits
(dependency declarations, a framework callback or a `progress`/`track` wrapper around
the main loop, one `root` scope), runs a short smoke test and checks the output with
`verify.py`.

### Install

Copy the folder into the repository you want to adopt Pace in, or into your user-level
skills so it is available everywhere:

```bash
# in that repository: shared with everyone who clones it
mkdir -p .claude/skills
cp -r /path/to/pace/skills/adopt-pace .claude/skills/

# or for you only, in every repository
mkdir -p ~/.claude/skills
cp -r /path/to/pace/skills/adopt-pace ~/.claude/skills/
```

Then open Claude Code in that repository and ask:

> add Pace to this repo

Claude Code reads `SKILL.md`, shows its plan and asks before editing. `patterns.md`
(snippets for each situation) and `verify.py` (the output checker) sit beside it and are
read or run as needed; copy all three files.

### Check a run yourself

`verify.py` needs only the standard library and `precisionai-pace`:

```bash
python skills/adopt-pace/verify.py run.log
my-job --smoke 2>&1 | python skills/adopt-pace/verify.py
```

It reports every scope, which one is the root, missing totals or `end` records, plan
names that match no scope, and what the estimator concludes at the end of the log. The
exit status is 0 when nothing is wrong, 1 on an error (or any warning with `--strict`),
and 2 when there was nothing to read.
