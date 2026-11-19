# Phase 4 — Autonomous Planning, Execution and Self-Correction

Phase 4 turns BEAST's single-turn agent loop into a bounded autonomous coding
system. A request is decomposed by the **selected local model** into a task
plan; BEAST executes the plan one task at a time through the existing tool
stack, validates each result with the project's own commands, and repairs
failures within hard limits. Every write and every command still goes through
the Phase 2 approval protocol.

Nothing in this phase hardcodes a task sequence for a project type. The plan,
the order of work and the corrections are model output. The runtime only
enforces graph validity, permissions, validation evidence, cancellation and
budgets.

All Phase 1–3 behaviour is preserved: the chat interface, model picker, agent
loop, six original tools, `project_info`, project intelligence, approvals,
workspace containment, secret redaction and read-only analysis are unchanged.

---

## 1. Architecture

New modules:

```
beast/core/
├── planner.py         model → validated Plan (JSON tolerance, graph repair)
├── task_manager.py    immutable Task graph, statuses, retries, revisions
├── validation.py      command discovery, execution, evidence, signatures
├── changes.py         change tracker, unified diffs, conservative rollback,
│                      read-only Git inspection
└── autonomy.py        AutonomousRunner: bounded plan/execute/validate/repair

beast/ui/
├── commands.py        /auto /changes /diff /rollback /git /cancel /help
├── screens.py         slash-command handling, autonomous event rendering
└── widgets.py         TaskBoard, RunActivity, RunSummaryView, CommandOutput
```

Supporting changes live in existing modules:

| Module | Phase 4 role |
|---|---|
| `beast/core/agent.py` | unchanged THINK → ACT → OBSERVE loop, reused once per task |
| `beast/tools/base.py` | tracks approval decisions on the change tracker |
| `beast/tools/filesystem.py` | unchanged write/edit tools; snapshots are captured by the tracker before writing |
| `beast/tools/shell.py` | unchanged `run_command`; validation reuses it so commands remain approved |
| `beast/cli.py` / `beast/main.py` | `--auto "<request>"` starts a run from the command line |
| `beast/ui/app.py` | passes the initial autonomous request to the chat screen |

Data flow:

```
request
  │
  ▼
TaskPlanner ──► Plan (frozen, normalised, bounded)
  │
  ▼
TaskManager ──► ready task ──► fresh Agent + the normal tool registry
  │                                  │
  │                                  └─ READ automatic; WRITE/EXECUTE approval
  ▼
ValidationEngine
  ├─ syntax checks on changed files (.py/.json/.toml)
  ├─ project-declared commands (tests, lint, typecheck, build)
  └─ exit codes + stdout/stderr evidence (redacted)
  │
  ├─ passed ─────────────────────────────► COMPLETED (verified)
  ├─ failed, attempts remain ────────────► correction prompt (same agent)
  ├─ failed at the limit ────────────────► FAILED; dependants BLOCKED
  └─ unfinished work remains ────────────► model may revise the plan (bounded)
```

`AutonomousRunner` is the only orchestrator. It emits frozen events consumed by
the TUI and by tests; it is not tied to Textual.

---

## 2. Planning

`TaskPlanner` asks the **selected model** for a JSON plan and never invents a
task list itself. The default system prompt requests a goal and a list of tasks
with stable ids, concise titles, optional descriptions and explicit
dependencies.

Parsing is defensive because local models drift:

- JSON is accepted inside prose or Markdown fences; a bare list is accepted.
- Alternate field names (`plan`, `step`, `task`, `details`, `depends`) are
  tolerated.
- Missing ids are generated; duplicate, empty and unsafe ids are normalised.
- Unknown dependencies are dropped, self-dependencies are removed, and cycles
  are broken deterministically.
- Titles and descriptions are length-capped, and the plan is truncated to the
  configured maximum (default 12 tasks, configurable down to 1).
- Empty, oversized or non-JSON answers raise `PlanError`; the run reports the
  failure instead of guessing.
- Every repair is recorded in `plan.warnings` and shown to the user.

Plan revision uses the same model. Completed task ids are protected: a revised
plan may depend on completed work but may not reuse its ids. If a revision
would collide, the id is renamed (`t1 → t1-r2`) and the dependency is rewritten.

The planner sends only the request, a bounded project-context section (when
intelligence is available), completed/remaining tasks and the failure evidence
from the previous attempt. The full project is not sent.

---

## 3. Task management

`Task` is a frozen dataclass. Every transition returns a new value:

```
PENDING ──start──► IN_PROGRESS ──complete──► COMPLETED
   ▲                    │
   └──retry─────────────┼──fail──► FAILED / BLOCKED / CANCELLED
```

- **Dependencies.** A task starts only when every dependency is `COMPLETED`.
  Plan order breaks ties. Duplicate ids, unknown dependencies, self-dependencies
  and cycles are rejected before the manager changes anything.
- **Retries.** `max_attempts` bounds how many times a task may be started.
  A failure below the limit returns it to `PENDING`; at the limit it is final.
- **Blocked work.** When a task becomes `FAILED`, `BLOCKED` or `CANCELLED`,
  every transitive dependant becomes `BLOCKED` immediately; it is never run.
- **Verification.** `verified=True` only when validation actually ran project
  commands and every check passed. `verified=None` means the work could not be
  proven (no changes, no commands, rejected approval). `False` means failure.
- **Cancellation.** `cancel_all()` marks every unfinished task `CANCELLED` and
  leaves completed work intact.
- **Revisions.** A revision preserves completed tasks, archives unfinished ones
  into history, validates the new graph atomically and is itself bounded
  (default 2 revisions).
- **No infinite execution.** The manager also enforces a task-count ceiling
  (default 50), and the runner enforces run, task and command time budgets.

The TUI's task board shows every status with a glyph, the current task, attempt
count after a retry, progress, validation state, debug state and change count.
It is bounded to eight rows; longer plans report the hidden remainder.

---

## 4. Execution

For each ready task the runner creates a **fresh agent** with a task-specific
prompt containing:

- the original request and plan goal;
- completed tasks (with their validation detail);
- remaining tasks, explicitly marked "do NOT do these now";
- the current task id, title and description;
- files already changed in this run;
- the prior failure detail when retrying a task.

The agent uses the normal tool registry and the normal permissions. The model
decides which tools to call and in what order. The runtime adds no templates and
no file lists.

Corrections reuse the **same agent** for the task, so the fix is informed by the
work just done. Every correction receives the actual validation evidence.

A task's changed files are the tracker's cumulative changed paths for the run.
Validation therefore checks the whole set of files BEAST has touched, not only
the last write.

---

## 5. Validation

`ValidationEngine` discovers the checks a project already declares. It never
installs dependencies and never invents a test command.

| Ecosystem | Evidence inspected | Commands |
|---|---|---|
| Python | `pyproject.toml`, `pytest.ini`, `conftest.py`, `tox.ini`, `setup.cfg`, `tests/`, `test_*.py`, ruff/mypy config | `python -m pytest -q`, `ruff check .`, `mypy .` |
| Node/JS/TS | `package.json` scripts; lockfile selects npm/pnpm/yarn/bun | `npm test`, `npm run <lint\|build\|typecheck>`, `node --check` for changed JS |
| Go | `go.mod` | `go test ./...`, vet |
| Rust | `Cargo.toml` | `cargo test`, `cargo check` |
| Java | `pom.xml`, Gradle files (wrapper selection) | Maven/Gradle test/check |
| .NET | `*.sln`, `*.csproj` | `dotnet test`, `dotnet build` |
| Ruby / PHP / Make | Rakefile, phpunit config, Makefile | test/check targets |

Discovery is bounded (default three commands), deduplicated and ordered by
evidence strength: tests before lint, typecheck and build. In-process syntax
checks run for `.py`, `.json` and `.toml` changed files even when no command is
found.

**Execution.** Every discovered command runs through `RunCommandTool`, so it
requires the same explicit approval as any other command. The engine captures
stdout/stderr, the exit code, timeout and rejection state. A rejection stops the
validation pass; BEAST does not keep asking.

**Honest status.** A report is `verified` only when at least one command ran and
nothing failed. The user-facing status distinguishes:

- `PASSED — N commands`;
- `PASSED — N syntax check(s) only; no tests or build found`;
- `FAILED — N syntax error(s)` / `N command(s) failed`;
- `NOT VERIFIED — …` for no commands, rejected approval, no changes or
  interruption;
- `STOPPED — validation was interrupted`.

Failure evidence is redacted with the Phase 3 secret redactor before it reaches
the model or the transcript. A SHA-256 signature over the failing commands and
the tail of their output detects repeated failures.

---

## 6. Debugging and self-correction

When validation fails, the runner enters a bounded repair loop:

1. Build evidence from syntax errors and failed command output.
2. Compute the failure signature.
3. Stop immediately if the signature repeats the previous one — the model is
   not allowed to make the same ineffective fix forever.
4. Stop at `BEAST_MAX_DEBUG_ATTEMPTS` (default 2).
5. Otherwise ask the same agent to fix only the reported failures.
6. Re-run validation.
7. Complete the task only when validation passes.

A task that exhausts its corrections is `FAILED` with the real evidence in its
detail. The run summary reports `failed` or `partial`; BEAST never claims an
error was fixed unless the relevant validation succeeded.

If unfinished tasks remain, the planner may be asked for a bounded revision
(default 2). The revised graph is validated like the original. If revision
fails or the limit is reached, the remaining tasks are blocked or cancelled and
the run ends with an honest verdict.

---

## 7. Change tracking

`ChangeTracker` records BEAST-attributed work:

- files created, modified or deleted, with before/after hashes;
- the original bytes of a modified file, captured **after approval but before
  the write** in a private backup directory;
- commands with exit code, timeout and rejection state;
- approval decisions (tool, target, allowed/denied);
- failures such as model or validation errors.

It produces:

- a compact summary line (`3 created, 2 modified, 0 deleted`);
- a full report for `/changes` and the run summary;
- unified diffs (`/diff [path]`) including `/dev/null` for new files;
- read-only Git status/diff through `GitInspector`.

Git integration is optional and inspection-only. BEAST never commits, never
pushes, never resets, never checks out and never cleans. If `git` is missing or
the directory is not a repository, the command reports that plainly.

---

## 8. Rollback

Rollback restores only work BEAST can confidently attribute to itself:

- a modified file is restored only if its current hash still matches the content
  BEAST last wrote — if you edited it afterwards, it is skipped with a reason;
- a file BEAST created is deleted only if it still matches BEAST's content;
- an already-deleted file is restored only when a pre-deletion snapshot exists;
- untracked paths, paths outside the workspace and already-rolled-back paths are
  refused or skipped with an explanation.

`/rollback` asks for approval first. BEAST does **not** run `git reset`,
`git checkout`, `git clean` or any other blanket revert, and it never touches
unrelated uncommitted work. The rollback report lists every restored, deleted
and skipped path.

---

## 9. Security

Phase 4 keeps every Phase 2/3 boundary:

- **Workspace containment.** File paths are resolved (including symlinks and
  Windows junctions) and escapes are refused.
- **Credential protection.** Credential files stay excluded from analysis, and
  secret redaction is applied to validation evidence and context.
- **Permissions are not model-controlled.** READ tools run automatically.
  WRITE and EXECUTE tools always call the approval protocol. The model cannot
  pass a flag, choose an approver or skip the dialog. Anything other than an
  explicit approval counts as a refusal.
- **No dependency installation.** BEAST never runs `pip install`, `npm install`
  or equivalent on its own. If something is missing, the task must report it.
- **No automatic Git mutation.** No commits, pushes, resets, checkouts or
  cleans.
- **No silent model switching.** The planner and every task use the selected
  local model. Cloud models are filtered by Ollama capability metadata; BEAST
  has no cloud API path.
- **Bounded autonomy.** Task count, plan revisions, agent iterations, task
  retries, debug attempts, run time, task time and command time all have hard
  ceilings, configurable through environment variables and clamped to safe
  ranges.

### Approval is not an OS sandbox

**Command approval does not provide operating-system sandboxing.** An approved
command runs with your account's privileges and can read, write or delete
anything your account can reach, including paths outside the workspace, and can
make its own network connections. Workspace containment applies to BEAST's file
tools, not to arbitrary programs the model asks to run. BEAST does not
virtualize the file system, network, environment or child processes. Read the
command before approving it. Treat an approved action exactly as you would treat
running that command yourself.

---

## 10. Model compatibility and limitations

- Model discovery is runtime-based. No model name is hardcoded.
- Native tool calling is capability-detected. A model without the `tools`
  capability can chat but cannot run autonomous tasks; BEAST explains this and
  offers `Ctrl+N`.
- Unknown capability metadata (`?`) is treated as "tools unavailable", not as
  an optimistic guess.
- Planner quality varies. Small local models may emit malformed JSON, omit
  dependencies or produce shallow plans. BEAST repairs what it safely can and
  reports the rest; it does not silently replace the model.
- If a model cannot produce a usable plan, the run fails with a clear
  `PlanError` message. There is no fallback to a different model or a cloud API.
- Sampling is local and non-deterministic. Automated tests therefore use
  mocked model responses; real-model demos are reported separately.

---

## 11. Installation and usage

Requirements are unchanged: Python 3.11+, Ollama running locally, and at least
one model. A tool-capable model is required for autonomous mode.

```bash
python -m pip install -e .
cd C:\path\to\your-project
ASAP launch beast                       # interactive picker
ASAP launch beast --model qwen3:8b      # fixed model
ASAP launch beast --auto "Add a dark mode toggle to the settings page"
```

Inside the TUI:

| Command | Effect |
|---|---|
| `/auto <request>` | Plan and execute a request autonomously |
| `/changes` | Show files BEAST created, modified or deleted |
| `/diff [path]` | Unified diff of BEAST-attributed changes |
| `/rollback [path…]` | Ask for approval, then restore BEAST's changes |
| `/git` | Read-only Git status of the workspace |
| `/cancel` | Stop the running autonomous task |
| `/help` | Command list |

`Ctrl+O` toggles the task board. The board also appears automatically when a run
starts. The status bar moves through `PLANNING`, `TASK`, `VALIDATING`,
`DEBUGGING`, `TOOL`, `APPROVAL` and back to `IDLE`. `Esc` requests a cooperative
stop and cancels the active agent/validation work.

Environment limits (clamped when out of range):

| Variable | Default | Meaning |
|---|---|---|
| `BEAST_MAX_TASKS` | 12 | Tasks accepted from one plan |
| `BEAST_MAX_AGENT_ITERATIONS` | 12 | Tool rounds per agent run |
| `BEAST_MAX_DEBUG_ATTEMPTS` | 2 | Corrections per task |
| `BEAST_MAX_TASK_ATTEMPTS` | 2 | Starts per task |
| `BEAST_MAX_PLAN_REVISIONS` | 2 | Plan revisions per run |
| `BEAST_MAX_RUN_SECONDS` | 900 | Whole-run wall clock |
| `BEAST_TASK_TIMEOUT` | 300 | One task's wall clock |
| `BEAST_COMMAND_TIMEOUT` | 120 | One validation command |

---

## 12. Demonstrations

The required demos are checked in under
[`docs/phase4-dod/`](phase4-dod/). They are produced by:

```bash
python scripts/phase4_definition_of_done.py
```

By default the harness scripts the *model decisions* (a fake planner and
scripted agents) while running the **real** task manager, tools, approval
protocol, change tracker, validation engine, subprocess execution and failure
parsing. This makes the demos deterministic and safe to run in CI. The
transcripts say "scripted model, real runtime" and never claim a model wrote
the code.

| Demo | What it proves | Result |
|---|---|---|
| **A — Task-management app** | Model-style plan, multi-file generation, pytest discovery, verified completion | 3 files created, `python -m pytest -q` passed, 1/1 verified |
| **B — Dark-mode toggle** | Modify existing CSS/JS while preserving unrelated behaviour; honest weak-evidence reporting | 2 files modified, `node --check` passed; status explicitly says "syntax check only; no tests or build found" |
| **C — Find and fix a bug** | Real failing pytest output drives one correction, then passes | first validation `FAILED`, 1 debug attempt, second validation `PASSED`, task verified |
| **D — Bounded correction** | Endless-loop prevention and honest failure | 2 corrections used, third validation still failed, task `FAILED`, run `failed`, 0 verified |

The harness auto-approves file writes and accepts only validation-shaped
commands (pytest, node --check, lint/test/build commands) without shell
metacharacters; everything else is rejected and recorded. This is a demo
convenience, **not** a security sandbox.

A real-model run is opt-in:

```bash
python scripts/phase4_definition_of_done.py --real --model qwen3:8b --only a
```

Real-model results depend on the installed model and available memory. Any
skipped or timed-out real run is reported as such, not counted as a pass.

---

## 13. Testing

The Phase 4 suite uses mocked model streams, fake planners, fake agents, fake
command runners and fake Git subprocesses. It needs no Ollama service and no
interactive terminal.

Covered behaviour includes:

- planner JSON tolerance, normalisation, cycle breaking, truncation, revision
  id protection and bounded prompting;
- task graph validity, dependency ordering, retries, blocking, cancellation,
  revision atomicity and progress accounting;
- validation discovery for Python, Node, Go, Rust, Java, .NET, Ruby, PHP and
  Make; syntax checks; command outcomes; rejection, timeout, redaction,
  signatures and honest status labels;
- change tracking, first-snapshot preservation, unified diffs, conservative
  rollback, path refusal and read-only Git inspection;
- autonomous execution: success, failure → correction → success, debug cap,
  repeated-signature stop, retry, plan revision, planning failure, cancellation,
  run/task timeouts, model errors, environment limit parsing and validation
  limits;
- TUI slash commands, rollback approval/rejection, Git output, task-board
  toggle, autonomous event rendering, `initial_auto` plumbing and refusal for
  chat-only models;
- streamed Markdown mouse interaction during and after a long reply, including
  the detached-block selection race and terminal resizing (see
  [`TUI_MOUSE_CRASH.md`](TUI_MOUSE_CRASH.md)).

See the results section below for the exact executed counts and coverage.

---

## 14. Known limitations

- **Validation discovery is heuristic.** It recognises common manifests and
  script names; an unusual build system may produce no command. BEAST then
  reports `NOT VERIFIED` rather than pretending success.
- **Syntax-only evidence is weak.** `node --check`, JSON/TOML parsing and
  similar checks cannot prove behaviour; the status text says so.
- **No dependency installation.** If tests require missing packages, validation
  fails and BEAST reports the missing dependency instead of installing it.
- **Cumulative validation scope.** A task validates all files changed in the
  run so far, not a per-task diff. This is deliberate (later tasks must not
  break earlier work) but can attribute an earlier failure to a later task.
- **Rollback cannot recover unsnapshotted deletions.** A file deleted outside
  BEAST's capture path is tracked but cannot be restored.
- **Commands are not sandboxed.** See the security section.
- **Planner reliability depends on the model.** Very small models may not
  produce usable JSON or may propose shallow plans.
- **Windows-first.** Command discovery uses portable invocations; platform
  quirks (Gradle wrapper names, shell selection) are handled but not every
  exotic toolchain is covered.

---

## 15. Test results

Executed on Windows 11 with Python 3.14.7. `pyproject.toml` sets
`addopts = "-q"`, so the commands below override `addopts` to keep pytest's
final summary line visible.

```bash
python -m pytest -o addopts="" -q
python -m pytest --cov=beast --cov-report=term-missing -o addopts="" -q
```

### Automated suite

| Metric | Result |
|---|---|
| Collected | **553** |
| Passed | **551** |
| Failed | **0** |
| Skipped | **2** |
| Duration | 115.64 s (coverage run) |
| Statement coverage | **91%** (7250 statements, 679 missed) |

The total includes three streamed-Markdown mouse regressions added with the
TUI crash fix documented in [`TUI_MOUSE_CRASH.md`](TUI_MOUSE_CRASH.md).

The two skips are symlink tests: the platform refuses to create the link, so
the test is skipped rather than reported as a pass. No test was deselected or
ignored.

### Definition-of-done demos

`python scripts/phase4_definition_of_done.py` runs all four required demos
deterministically. All four passed:

| Demo | Result |
|---|---|
| A — task-management app | PASS — 3 files created, pytest passed, 1/1 task verified |
| B — dark-mode toggle | PASS — 2 files modified, `node --check` passed, weak evidence labelled honestly |
| C — find and fix a bug | PASS — first validation failed, one correction, second validation passed |
| D — bounded correction | PASS — stopped at the correction limit, task/run marked failed, 0 verified |

The full event stream and verification checks are in
[`phase4-dod/`](phase4-dod/). These runs script the model decisions while
exercising the real planner interface, task manager, tools, approval protocol,
change tracker, validation discovery, subprocess execution and failure parser.

### Real-model result

A real-model demo was attempted once:

```bash
python scripts/phase4_definition_of_done.py --real --model qwen3:8b --only a
```

It exceeded the 10-minute foreground window, was allowed to finish in the
background, and exited with code 1:

```text
== demo A: Task-management app ==
   FAILED: task_app.py exists
```

Demo A is therefore **not** claimed as a real-model pass. Per the resource
guidance the expensive run was not restarted. The real-model path is available
on a machine with more time or memory budget; it is opt-in and never required
by the automated suite.

### What is not claimed

- Real-model planning and code quality are not verified beyond the one attempt
  above.
- The deterministic demos prove the runtime's control flow and safety
  properties, not that a specific local model writes good code.
- Coverage is statement coverage from the commands above; it is not a
  correctness proof.
