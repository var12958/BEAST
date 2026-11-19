# Phase 4 demo A — Task-management app

- Mode: scripted model, real runtime
- Model: `scripted-model`
- Verdict: **completed**
- Request: Create a small task management app with tests.

Everything below the request is captured from BEAST's own event stream.
The unattended harness approves file writes and rejects commands that do
not look like project validation.

```text
workspace: C:\Users\Varun\AppData\Local\Temp\beast-phase4-a-_a37zvn9
fixture files: []

[PLAN] model=scripted-model request='Create a small task management app with tests.'
[PLAN] ready: 1 task(s)
[TASK] t1 started: Create the task manager and its tests
[TOOL] write_file {'path': 'pyproject.toml', 'content': '[tool.pytest.ini_options]\ntestpaths = ["."]\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Created pyproject.toml (2 lines). The file is now on disk.
[TOOL] write_file {'path': 'task_app.py', 'content': '"""Small in-memory task manager used by the Phase 4 demo."""\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass, field\n\n\n@dataclass\nclass Task:\n    title: str\n    done: bool = False\n\n\n@dataclass\nclass TaskStore:\n    tasks: list[Task] = field(default_factory=list)\n\n    def add(self, title: str) -> Task:\n        task = Task(title=title)\n        self.tasks.append(task)\n        return task\n\n    def complete(self, index: int) -> Task:\n        task = self.tasks[index]\n        task.done = True\n        return task\n\n    def pending(self) -> list[Task]:\n        return [task for task in self.tasks if not task.done]\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Created task_app.py (29 lines). The file is now on disk.
[TOOL] write_file {'path': 'test_task_app.py', 'content': 'from task_app import TaskStore\n\n\ndef test_add_and_complete() -> None:\n    store = TaskStore()\n    store.add("write demo")\n    store.add("run tests")\n    store.complete(0)\n\n    assert [task.title for task in store.pending()] == ["run tests"]\n    assert store.tasks[0].done is True\n'}
[APPROVAL] write_file
[TOOL] write_file -> ok: Created test_task_app.py (11 lines). The file is now on disk.
[MODEL] scripted step 1: wrote 3 file(s); BEAST will validate the result.
[VALIDATION] started for t1: ['pyproject.toml', 'task_app.py', 'test_task_app.py']
[VALIDATION] PASSED — 1 command
    [SYNTAX] pyproject.toml — ok
    [SYNTAX] task_app.py — ok
    [SYNTAX] test_task_app.py — ok
    [TEST] python -m pytest -q  (pyproject.toml) → exit 0
[TASK] t1 completed (verified): validated: PASSED — 1 command
[RUN]
  Run completed — 1/1 tasks completed
  Validation: 1 verified, 0 completed without verification
  Changes: 3 created, 0 modified, 0 deleted
  Limits used: 0 debug attempt(s), 0 plan revision(s), 1 validation pass(es)
  Duration: 4s

approval decisions:
  approved [write] pyproject.toml
  approved [write] task_app.py
  approved [write] test_task_app.py
  approved [execute] python -m pytest -q

elapsed: 3.7s

checks:
  PASS — task_app.py exists
  PASS — test_task_app.py exists
  PASS — pyproject.toml exists
  PASS — task manager has tests on disk
  PASS — validation ran and passed
  PASS — all tasks completed
  PASS — tasks were verified
  PASS — three files created
  PASS — change tracker recorded at least one file
file on disk: .pytest_cache/.gitignore
file on disk: .pytest_cache/CACHEDIR.TAG
file on disk: .pytest_cache/README.md
file on disk: .pytest_cache/v/cache/nodeids
file on disk: __pycache__/task_app.cpython-314.pyc
file on disk: __pycache__/test_task_app.cpython-314-pytest-9.1.1.pyc
file on disk: pyproject.toml
file on disk: task_app.py
file on disk: test_task_app.py
```
