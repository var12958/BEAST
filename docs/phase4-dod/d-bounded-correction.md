# Phase 4 demo D — Bounded correction

- Mode: scripted model, real runtime
- Model: `scripted-model`
- Verdict: **failed**
- Request: Make the failing test pass.

Everything below the request is captured from BEAST's own event stream.
The unattended harness approves file writes and rejects commands that do
not look like project validation.

```text
workspace: C:\Users\Varun\AppData\Local\Temp\beast-phase4-d-tr69labd
fixture files: ['calculator.py', 'pyproject.toml', 'test_calculator.py']

[PLAN] model=scripted-model request='Make the failing test pass.'
[PLAN] ready: 1 task(s)
[TASK] t1 started: Fix the failing test
[TOOL] write_file {'path': 'calculator.py', 'content': 'def add(left: int, right: int) -> int:\n    return left - right  # attempt 1: still subtracts\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Updated calculator.py (2 lines). The file is now on disk.
[MODEL] scripted step 1: wrote 1 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['calculator.py']
[VALIDATION] FAILED — 1 command(s) failed
    [SYNTAX] calculator.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 1
[DEBUG] attempt 1/2 signature=8e2e5f4e2381
[TOOL] write_file {'path': 'calculator.py', 'content': 'def add(left: int, right: int) -> int:\n    return left + right + 1  # attempt 2: still off by one\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Updated calculator.py (2 lines). The file is now on disk.
[MODEL] scripted step 2: wrote 1 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['calculator.py']
[VALIDATION] FAILED — 1 command(s) failed
    [SYNTAX] calculator.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 1
[DEBUG] attempt 2/2 signature=86b4aadba010
[TOOL] write_file {'path': 'calculator.py', 'content': 'def add(left: int, right: int) -> int:\n    return left + right + 2  # attempt 3: still wrong\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Updated calculator.py (2 lines). The file is now on disk.
[MODEL] scripted step 3: wrote 1 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['calculator.py']
[VALIDATION] FAILED — 1 command(s) failed
    [SYNTAX] calculator.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 1
[TASK] t1 FAILED: validation still fails after 2 correction attempt(s):
Validation failures:
- [TEST] python -m pytest -q (exit 1)
Exit code: 1
--- stdout ---
F                                                                        [100%]
================================== FAILURES ===================================
__________________________________ test_add ___________________________________

    def test_add() -> None:
>       assert add(2, 3) == 5
E       assert 7 == 5
E        +  where 7 = add(2, 3)

test_calculator.py:5: AssertionError
=========================== short test summary info ===========================
FAILED test_calculator.py::test_add - assert 7 == 5
1 failed in 0.28s
--- stderr ---
(no output)
[RUN]
  Run failed — 0/1 tasks completed, 1 failed
  Validation: 0 verified, 0 completed without verification
  Changes: 0 created, 1 modified, 0 deleted
  Limits used: 2 debug attempt(s), 0 plan revision(s), 3 validation pass(es)
  Duration: 8s

approval decisions:
  approved [write] calculator.py
  approved [execute] python -m pytest -q
  approved [write] calculator.py
  approved [execute] python -m pytest -q
  approved [write] calculator.py
  approved [execute] python -m pytest -q

elapsed: 7.9s

checks:
  PASS — calculator.py exists
  PASS — test_calculator.py exists
  PASS — run did not claim success
  PASS — task is failed
  PASS — no verified task
  PASS — correction limit respected
  PASS — failure explains the limit
  PASS — validation never passed
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
