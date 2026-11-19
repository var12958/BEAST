# Phase 4 demo C — Find and fix a bug

- Mode: scripted model, real runtime
- Model: `scripted-model`
- Verdict: **completed**
- Request: Make the project's failing test pass without weakening it.

Everything below the request is captured from BEAST's own event stream.
The unattended harness approves file writes and rejects commands that do
not look like project validation.

```text
workspace: C:\Users\Varun\AppData\Local\Temp\beast-phase4-c-ai4co8ta
fixture files: ['calculator.py', 'pyproject.toml', 'test_calculator.py']

[PLAN] model=scripted-model request="Make the project's failing test pass without weakening it."
[PLAN] ready: 1 task(s)
[TASK] t1 started: Run the failing test, find the bug and fix it
[TOOL] write_file {'path': 'calculator.py', 'content': 'def add(left: int, right: int) -> int:\n    # The existing test says this should add; reproduce the failure first.\n    return left - right\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Updated calculator.py (3 lines). The file is now on disk.
[MODEL] scripted step 1: wrote 1 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['calculator.py']
[VALIDATION] FAILED — 1 command(s) failed
    [SYNTAX] calculator.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 1
[DEBUG] attempt 1/2 signature=26faa3b289f3
[TOOL] write_file {'path': 'calculator.py', 'content': 'def add(left: int, right: int) -> int:\n    return left + right\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Updated calculator.py (2 lines). The file is now on disk.
[MODEL] scripted step 2: wrote 1 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['calculator.py']
[VALIDATION] PASSED — 1 command
    [SYNTAX] calculator.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 0
[TASK] t1 completed (verified): validated: PASSED — 1 command
[RUN]
  Run completed — 1/1 tasks completed
  Validation: 1 verified, 0 completed without verification
  Changes: 0 created, 1 modified, 0 deleted
  Limits used: 1 debug attempt(s), 0 plan revision(s), 2 validation pass(es)
  Duration: 6s

approval decisions:
  approved [write] calculator.py
  approved [execute] python -m pytest -q
  approved [write] calculator.py
  approved [execute] python -m pytest -q

elapsed: 6.5s

checks:
  PASS — calculator.py exists
  PASS — test_calculator.py exists
  PASS — bug fixed on disk
  PASS — validation first failed
  PASS — validation later passed
  PASS — task completed
  PASS — task verified
  PASS — one debug attempt used
  PASS — change tracker recorded at least one file
file on disk: .pytest_cache/.gitignore
file on disk: .pytest_cache/CACHEDIR.TAG
file on disk: .pytest_cache/README.md
file on disk: .pytest_cache/v/cache/lastfailed
file on disk: .pytest_cache/v/cache/nodeids
file on disk: __pycache__/calculator.cpython-314.pyc
file on disk: __pycache__/test_calculator.cpython-314-pytest-9.1.1.pyc
file on disk: calculator.py
file on disk: pyproject.toml
file on disk: test_calculator.py
```
