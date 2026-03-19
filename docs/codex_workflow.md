# Codex Workflow for Project 1

## Daily Prompt Template
Use this exact prompt structure when we start:

```text
Goal for this session: <one concrete goal>
Current blockers: <if any>
Please do: inspect -> propose next command -> run -> explain output -> apply changes -> test.
```

## Session Rhythm
1. You set goal.
2. Codex inspects repo and context.
3. Codex proposes the next command.
4. We run it.
5. Codex explains what happened in plain English.
6. We iterate until goal is done.

## Command Playbook
- Explore files: `rg --files`, `find . -maxdepth 3 -type d | sort`, `sed -n '1,200p' <file>`
- Run app: `streamlit run app/streamlit_app.py`
- Run pipeline: `python scripts/run_pipeline.py`
- Run tests: `pytest -q`
- Debug imports quickly: `PYTHONPATH=src python -c "from job_intel.matching import HybridMatcher; print('ok')"`

## Learning Checkpoints
At the end of each milestone, ask Codex to include:
- What changed
- Why it changed
- How to run it
- How to verify it

Suggested prompt:

```text
Before next step, give me a learning checkpoint:
- What we changed
- Why this design is useful
- One thing I should practice now
```

## Review Checklist
- Can I run the feature locally without touching code?
- Do tests cover key behavior and edge cases?
- Is output cached in `data/cache/`?
- Are explanations visible to user in Streamlit?
