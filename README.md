# BEAST — Local AI Coding Agent

BEAST is a terminal AI coding agent that runs entirely on your machine. It talks
to [Ollama](https://ollama.com) over `localhost`, streams replies into a
full-screen terminal interface, and never sends your code, prompts or replies to
a hosted service. No API keys, no accounts, no telemetry.

**Phase 2** adds the agentic execution loop: BEAST can inspect the workspace,
read and search files, write and edit code, and run commands — with every
change and every command gated behind explicit approval. The local model
decides which tools to call and in what order; nothing in BEAST hardcodes a
task sequence.

**Phase 3** gives BEAST a working understanding of the project before it acts.
It detects languages, frameworks, package managers, build systems, entry points
and test frameworks; analyses directories, symbols and dependencies; and
retrieves the files that matter for each question. Every conclusion is derived
from files actually inspected — there are no hardcoded project descriptions and
no invented intelligence. See [docs/PHASE3.md](docs/PHASE3.md).

**Phase 4** makes BEAST autonomous within hard bounds. The selected local model
plans the work as a dependency-aware task graph; BEAST executes one task at a
time through the existing tools, validates the result with the project's own
test/lint/build commands, and repairs failures within a bounded correction loop.
Every write and command still needs explicit approval, and nothing is committed,
pushed, installed or rolled back blindly. See [docs/PHASE4.md](docs/PHASE4.md).

```
█▀▄ █▀▀ ▄▀▄ █▀▀ ▀█▀
█▄█ ██▄ █▀█ █▄▄  █
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.14.7, Windows 11 |
| [Ollama](https://ollama.com/download) | Installed and running (`ollama serve` or the desktop app) |
| At least one chat model | Local (e.g. `ollama pull qwen3:8b`) or cloud (`ollama pull kimi-k2.7-code:cloud`) — BEAST never downloads models for you |
| A tool-capable model for agent mode | The model must advertise native tool calling. `ASAP launch beast --list-models` shows a `TOOLS` column (`yes` / `no` / `?`) |
| Ollama Cloud sign-in (cloud models only) | Run `ollama signin` once; BEAST reuses that account and never asks for an API key |

BEAST discovers whatever models you already have. The picker groups chat models
into **LOCAL MODELS** and **CLOUD MODELS**; both can be selected. Embedding-only
models are listed as hidden. Cloud models are proxied by your local Ollama
service to ollama.com using the credentials from `ollama signin`. Models that do
not support tool calling still work for chat; BEAST explains the limitation in
the transcript instead of silently switching to another model.

## Install

From the BEAST source directory:

```bash
python -m pip install -e .
```

This installs the dependencies (`httpx`, `pydantic`, `typer`, `rich`, `textual`)
and the `asap` console command.

> **Windows:** pip may place `asap.exe` in a per-user scripts directory that is
> not on `PATH` (for example `%APPDATA%\Python\Python314\Scripts`). If
> `asap` is not recognised, add that directory to your user `PATH`, then open a
> new terminal.

## Launch

BEAST is launched from **your project**, not from its own source tree:

```bash
cd C:\path\to\your-project
ASAP launch beast
```

The command works from any directory, including the VS Code integrated
terminal. The directory you launch from becomes the active workspace shown in
the header and status bar; BEAST never changes it and never asks you to `cd`
into the BEAST installation.

> On Windows the command is case-insensitive, so `ASAP launch beast` works as
> written. On Linux and macOS use `asap launch beast`.

Useful variations:

```bash
ASAP launch beast --list-models            # show what Ollama has, then exit
ASAP launch beast --model qwen3:8b         # skip the picker
ASAP launch beast --workspace .\service    # work in a sub-directory
ASAP launch beast --debug                  # tracebacks + verbose logs
ASAP launch beast --ollama-url http://192.168.1.5:11434
ASAP launch beast --project-summary        # detect stack/layout, print, exit
ASAP launch beast --auto "Add a dark mode toggle to the settings page"
```

## Using the interface

```
┌ BEAST AI CODING AGENT  v0.1.0 ──────────────────────────────────────────┐
│ Model  qwen3:8b  ·  Connection  LOCAL / OLLAMA  ·  Ollama v0.33.2       │
│ Workspace  ~/code/my-project                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ◆ BEAST   qwen3:8b                                                     │
│  Reading the project before I change anything…                          │
│  ▸ [TOOL] list_files  .                                                 │
│  ✓ [TOOL] list_files  · 4 entries under .                               │
│  … [TOOL] write_file  index.html  · awaiting approval                   │
│  ✓ [TOOL] write_file  · Created index.html (42 lines)                   │
│  Markdown reply with syntax-highlighted code blocks …                   │
│                                                                         │
│  ▌ You                                                                  │
│  Describe what you want to build...                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ ● IDLE   qwen3:8b   LOCAL / OLLAMA   ~/code/my-project   enter send …   │
└─────────────────────────────────────────────────────────────────────────┘
```

Tool calls show up inline in the transcript as `▸ [TOOL] name` while running,
`… [TOOL] name · awaiting approval` while the approval dialog is open, then
`✓` (success), `✕` (failure) or `⊘` (rejected). The status bar moves through
`THINKING`, `TOOL`, `APPROVAL` and back to `IDLE`.

When a tool needs permission, a modal shows the exact target and action:

```
BEAST wants to modify:
  index.html
Proposed action: Overwrite existing file
  - old line
  + new line
[ Approve ]  [ Reject ]
y / a approve · n / esc reject
```

Approving with the keyboard (`y`/`a`) or the button runs the action; `n`, `Esc`
or closing the screen rejects it. Reads never ask; writes and commands always
do. There is no default-yes: anything other than an explicit approval is a
refusal, and the model is told the action was rejected.

### Keyboard

| Key | Action |
|---|---|
| `Enter` | Send the prompt |
| `Shift+Enter` / `Ctrl+J` | Insert a newline (Ctrl+J covers terminals that cannot report Shift+Enter) |
| `Esc` | Stop the running turn (a running command is terminated); partial text stays in the transcript |
| `y` / `a` | Approve a pending write or command in the approval dialog |
| `n` / `Esc` | Reject a pending write or command |
| `Ctrl+N` | Open the model picker (`↑`/`↓`, `Enter` select, `Esc` back) |
| `Ctrl+R` | Refresh the model list (picker) |
| `Ctrl+O` | Show or hide the autonomous task board |
| `Ctrl+L` | Clear the conversation and start a fresh context |
| `Ctrl+Q` | Quit |

The header shows the ASCII wordmark, product name and version, the selected
model, `Connection: LOCAL / OLLAMA` and the workspace. The logo collapses on
narrow terminals so the header never wraps.

### Model selection

On launch BEAST asks Ollama for every installed model and shows a picker
listing name, size, quantization, family and parameter count. The list is
built at runtime — no model names are hardcoded, so any chat model you install
appears automatically. Local models appear under **LOCAL MODELS** and
cloud-hosted ones (`name:cloud`) under **CLOUD MODELS**; each row is also
tagged `local` or `cloud`.

Tool support is capability metadata, not a model name: entries whose
`capabilities` include `tools` get the full toolset. Models that report
`completion` without `tools` are shown as chat-only, and models whose
capabilities Ollama does not report are marked `?`. In both cases BEAST mounts
a visible notice and chats without tools; it never swaps in a different model
silently. Cloud failures (sign-in, connectivity, API errors) are reported
inline and the current selection stays as it was.

Switching models with `Ctrl+N` keeps the transcript on screen and starts a new
model context with a visible notice. BEAST never switches models silently: if a
model disappears or fails, you get an inline error and the current selection
stays as it was.

### Slash commands and autonomous mode

| Command | Effect |
|---|---|
| `/auto <request>` | The model plans the work, then BEAST executes and validates it |
| `/changes` | Show files BEAST created, modified or deleted |
| `/diff [path]` | Unified diff of BEAST-attributed changes |
| `/rollback [path…]` | Ask for approval, then restore only BEAST's changes |
| `/git` | Read-only Git status (never commits, pushes or resets) |
| `/cancel` | Stop the running autonomous task |
| `/help` | Show this list |

`Ctrl+O` toggles the task board. During a run it shows overall progress, one row
per task with its status, the current task, validation/debug state and the
change count; the final summary reports verified versus unverified tasks, the
changes made and the limits used.

Autonomous mode still uses the ordinary approval dialog for every write and
every validation command. It never installs packages, commits, pushes or runs
`git reset`/`checkout`/`clean`. Corrections are bounded by a debug-attempt limit
and a repeated-failure signature check, so a failing test cannot loop forever.
See [docs/PHASE4.md](docs/PHASE4.md) for the architecture and limits.

## The agent loop

```
prompt
  └─► send conversation + tool definitions to the selected model
        (local models run on this machine; cloud models are proxied by Ollama)
        ├─ model answers with text only ──► done, text streams to the TUI
        └─ model requests tool calls
              ├─ validate arguments against the tool's schema
              ├─ READ tools run immediately
              ├─ WRITE / EXECUTE tools wait for the approval dialog
              ├─ append the assistant tool call and the tool result to history
              └─ loop back to the model (bounded by a 12-round safety cap)
```

The model owns the sequence: it can list, search, read, write, edit, run a
command, verify its own work, and stop. The loop only enforces schema
validation, permissions, message consistency, cancellation and the iteration
cap. Unknown tool names and malformed arguments are answered with an error
observation instead of crashing, so the model can correct itself.

### Tools

| Tool | Permission | What it does |
|---|---|---|
| `list_files` | READ | Lists files/directories, optionally recursive, with depth and entry limits. Skips `.git`, `node_modules`, virtualenvs, caches and build output. |
| `read_file` | READ | Reads a text file with a size ceiling; refuses binary files; truncation is reported. |
| `search_files` | READ | Finds text (with line numbers) or filenames, with extension filtering and ignored directories. |
| `write_file` | WRITE | Creates a file with complete contents (creating parent directories) or overwrites an existing one. Always asks. |
| `edit_file` | WRITE | Replaces one exact passage. Refuses missing text and ambiguous (multiple-match) edits. Always asks. |
| `run_command` | EXECUTE | Runs one shell command in the workspace with stdin closed, captured stdout/stderr, an exit code and a timeout. Always asks; never runs silently. |
| `project_info` | READ | Inspects the indexed project: overview, structure, dependencies, symbols, one file's symbols/imports, symbol search, or an incremental refresh. Read-only, redacted, and mounted only when project intelligence is available. |

## What's implemented

- `asap → launch → beast` CLI hierarchy, installable with pip, launchable from
  any directory.
- Ollama detection: installed (`ollama` on `PATH`), running (`/api/version`,
  falling back to `/api/tags`), version, and friendly guidance for each failure.
- Dynamic model discovery from `GET /api/tags`, covering local and Ollama
  Cloud chat models (capability metadata, not names); embedding-only models are
  hidden with an explanation. The picker groups models by location and the
  `--list-models` table includes a `TOOLS` column.
- Full-screen Textual TUI: branded header, scrollable Markdown transcript,
  syntax-highlighted code, multiline prompt, streaming replies, thinking
  indicator with elapsed time, responsive status bar, and inline tool activity.
- Streaming chat through `POST /api/chat` (NDJSON), with conversation history
  sent on every turn.
- Native Ollama tool calling: tool definitions are sent on each request,
  `tool_calls` are parsed (including JSON-string arguments), and every result
  is appended as a `tool` message with `tool_name` so the conversation stays
  valid across rounds.
- The THINK → ACT → OBSERVE loop, driven entirely by the model, with a
  12-round cap, cancellation, malformed-call recovery and no lost partial
  output.
- Approval dialogs for writes and commands, with the exact target, action and
  a content/preview detail. Only an explicit `True` approval is accepted; a
  broken or missing approval UI is treated as a refusal.
- Workspace containment for all file tools (`../`, absolute escapes and
  symlinks that resolve outside the project are refused).
- Command execution with a platform shell, a workspace working directory,
  captured output limits, a timeout, process-tree termination and
  cancellation that kills the command.
- Interrupting a turn (`Esc`) keeps the partial answer in the transcript and
  in the agent's history. Cancelling mid-stream also records any tool calls the
  model had already requested, and cancelling mid-tool records an observation,
  so the next request never contains an unanswered call.
- Graceful handling of: Ollama missing, service stopped, no models, deleted
  model, model runner crash (friendly GPU/memory hint), timeout, malformed or
  empty responses, dropped connections, terminal resize, Ctrl+C and unexpected
  errors. Raw tracebacks only with `--debug`.
- Detailed logs at `%LOCALAPPDATA%\BEAST\logs\beast.log` (or
  `$XDG_STATE_HOME/beast/logs/beast.log` on Linux/macOS).

### Project intelligence (Phase 3)

- Runtime project detection from real evidence: languages and file counts,
  frameworks (manifest declarations plus framework-specific files), package
  managers (manifests + lockfiles), build systems, test frameworks, entry
  points, manifests and configuration files. Python, JavaScript/TypeScript,
  Java/Kotlin, C/C++, Rust, Go, Ruby, PHP, Dart/Flutter, Swift, .NET, Scala,
  Elixir and vcpkg projects are recognised; multi-language repositories are
  reported component by component.
- Directory structure analysis with source/test/documentation directories, key
  files and a bounded tree. `.gitignore` (including negation and nested files)
  and generated directories are respected; scans have configurable file,
  depth and size limits.
- AST-based symbol extraction for Python and JavaScript/TypeScript — classes,
  functions, methods, constants, imports, signatures, line ranges and
  docstrings — with an analyser registry for future languages.
- Declared-dependency analysis across ~20 manifest formats, grouped by scope,
  with warnings for malformed or out-of-workspace manifests. Nothing is ever
  installed, resolved or executed.
- Query-relevant context retrieval: lexical/identifier scoring, line-numbered
  excerpts, character/file budgets, and structure/dependency sections only
  when the question calls for them. The whole project is never sent blindly.
- A lightweight, incremental JSON index (size + mtime reuse, added/modified/
  deleted tracking, atomic writes, tamper-resistant loading, automatic
  invalidation after BEAST writes a file). No vector database.
- `[ANALYSIS]` transcript lines report what was retrieved and injected; the
  welcome panel shows Language, Framework, Package manager, Source files and
  Test files; `--project-summary` prints the overview without starting the UI.
- Security throughout: workspace confinement (symlinks **and** Windows
  junctions refused), credential-file exclusion, and secret redaction before
  any text reaches the model.

### Autonomous execution (Phase 4)

- Model-generated JSON plans with tolerant parsing and deterministic repair:
  missing/duplicate ids, unknown and self dependencies, cycles, overlong plans
  and malformed answers. No task sequence is hardcoded.
- A dependency-aware task graph with `PENDING`, `IN_PROGRESS`, `COMPLETED`,
  `FAILED`, `BLOCKED` and `CANCELLED` states, bounded retries, transitive
  blocking and atomic plan revisions.
- One fresh agent per task, with the original request, plan position, completed
  and remaining tasks, prior failure detail and files already changed. Later
  tasks are explicitly excluded.
- Project-declared validation discovery for Python, Node/JS/TS, Go, Rust, Java,
  .NET, Ruby, PHP and Make, plus in-process syntax checks for `.py`, `.json` and
  `.toml`. Commands run through `run_command`, so they still require approval.
- Exit-code interpretation, captured stdout/stderr, timeout and rejection
  handling, secret-redacted failure evidence and a repeated-failure signature
  check.
- Bounded self-correction (default 2 attempts per task) using the actual failure
  output; a task is marked verified only when validation ran and passed.
- Change tracking for created/modified/deleted files, commands, tests, failures
  and approvals, with unified diffs and read-only Git inspection.
- Conservative snapshot rollback that refuses to touch files the user edited
  after BEAST, never uses blanket Git commands and never affects unrelated work.
- Hard limits for tasks, agent rounds, retries, debug attempts, revisions, run
  time, task time and command time, configurable through `BEAST_*` variables.
- TUI task board, `[PLAN]` / `[VALIDATION]` / `[DEBUG]` activity lines, run
  summary, slash commands and `--auto` launch option.

## Security limitations

BEAST's file tools keep every path inside the workspace you launched it in:
paths are resolved (including symlinks) and anything outside is refused. Writes
and edits also refuse generated/tool-owned directories such as `.git` and
`node_modules`.

**This is a path check, not a sandbox.** In particular:

- An approved command runs with your account's privileges and can read, write
  or delete anything your account can reach. The workspace check does not apply
  to arbitrary programs.
- A process could change a path between the check and the file operation. The
  containment check is not a substitute for OS-level isolation.
- BEAST does not virtualize the file system, network, environment variables or
  child processes. Treat an approved action exactly as you would treat running
  that command or editing that file yourself.
- The approval dialog is the security boundary: read the command or change
  before approving it.

## Architecture

```
beast/
├── cli.py                 asap → launch → beast (Typer)
├── main.py                launch orchestration, discovery, error rendering
├── errors.py              BeastError hierarchy with messages + hints
├── core/
│   ├── agent.py           Agent: chat streaming + THINK/ACT/OBSERVE loop
│   ├── autonomy.py        AutonomousRunner: bounded plan/execute/validate/repair
│   ├── planner.py         model → normalised Plan (JSON tolerance, graph repair)
│   ├── task_manager.py    immutable Task graph, retries, blocking, revisions
│   ├── validation.py      command discovery, execution, evidence, signatures
│   ├── changes.py         change tracker, unified diffs, safe rollback, Git
│   ├── conversation.py    immutable Message/Conversation values
│   └── events.py          streaming, tool, plan, task, validation and run events
├── tools/
│   ├── base.py            Tool, Permission, ToolResult, ApprovalRequest, ToolContext
│   ├── registry.py        built-in tools, lookup + Ollama definitions
│   ├── schemas.py         compact JSON Schema from pydantic arg models
│   ├── filesystem.py      list_files, read_file, search_files, write_file, edit_file
│   ├── project.py         project_info (overview/structure/symbols/search/refresh)
│   └── shell.py           run_command (approval, timeout, output capture)
├── intelligence/          Phase 3: project understanding
│   ├── __init__.py        ProjectIntelligence facade (incremental, thread-safe)
│   ├── models.py          detection, structure, symbols, index, context values
│   ├── index_manager.py   scan → index → JSON cache, incremental refresh
│   ├── code_analyzer.py   Python and JS/TS AST analysers + registry
│   ├── project_detector.py languages, frameworks, managers, build/test tools
│   ├── structure_analyzer.py directories, key files, bounded tree
│   ├── dependency_analyzer.py declared dependencies from ~20 manifest formats
│   └── context_builder.py relevance scoring, excerpts, budgets, redaction
├── ollama/
│   ├── client.py          async /api/tags + streamed /api/chat (with tools)
│   ├── connection.py      install/running detection, OllamaStatus
│   └── models.py          pydantic models + ModelInventory split + tool metadata
├── ui/
│   ├── app.py             Textual App (theme, screens, approvals, client lifetime)
│   ├── screens.py         model picker, chat screen, approval dialog, auto runs
│   ├── commands.py        slash-command parsing and help text
│   ├── widgets.py         header, prompt, messages, tool activity, task board
│   ├── branding.py        name, ASCII art, palette, copy
│   └── styles.tcss        dark, professional theme
├── workspace/
│   ├── manager.py         validated project root + containment check
│   ├── scanning.py        shared walker: ignores, .gitignore, link refusal, limits
│   └── secrets.py         credential-file classification + redaction
└── utils/logger.py        file + optional stderr logging
```

Design choices worth knowing:

- **Layering.** The only module that knows the Ollama HTTP shape is
  `ollama/client.py`; only `core/agent.py` owns conversation state and the
  loop; only `tools/` mutates the file system or starts processes; only `ui/`
  imports Textual. The agent depends on a `ChatStreamer` protocol, so the loop
  is exercised headlessly in the tests.
- **One schema, two uses.** Each tool declares a pydantic argument model. The
  same model produces the JSON Schema sent to Ollama and validates the
  arguments that come back, so the advertised and enforced contracts cannot
  drift.
- **Immutable conversation and events.** Appending returns a new
  `Conversation`; the loop yields small frozen events that the TUI renders.
- **Capability-based tooling.** Tools are offered only when Ollama reports the
  `tools` capability. No model name is special-cased, and BEAST never switches
  models to get tool support.
- **Approval is data, not a tool argument.** The model cannot pass a flag to
  skip a prompt. The tool asks the application directly; only an explicit
  `True` proceeds.
- **Streaming throttle.** Markdown is re-rendered at most every 70 ms while
  tokens arrive, which keeps CPU use low during long replies.

### Extension points

| Area | Where to build next |
|---|---|
| New tools | Subclass `Tool`, add an args model, register it in `tools/registry.py` — the loop and the UI pick it up unchanged |
| Tool permissions | `Tool.permission` (`READ` / `WRITE` / `EXECUTE`) drives the approval dialog |
| Config | A config module feeding `Agent(options=...)`, `ToolContext` limits and `OllamaClient(...)` |
| More surfaces | `beast/ui/screens.py` can host additional screens without touching the client |

## Verified end-to-end

The Phase 2 acceptance task was run against a real local model
(`qwen3:8b` on Ollama 0.33.2) in an empty directory:

```
prompt: Create a simple login page using HTML, CSS and JavaScript.
[TOOL] write_file {'path': 'login.html', ...}
[APPROVAL] write_file (write) target='login.html'
  action: Create new file
  -> approved
[OK] Created login.html (71 lines)
I've created a simple login page with the following features: …
FILES ON DISK:
  login.html
```

The model chose `write_file` itself, supplied the complete page (HTML with
inline CSS and JavaScript), waited for the approval step, received the tool
result, and wrote the summary. The harness approved file writes automatically
because it was unattended and was configured to reject shell commands; the
interactive TUI asks for both. The resulting `login.html` is 1,985 bytes with
a centred form, styled inputs and a submit handler, and the summary matches
what is on disk.

The harness is checked in as `demo_login.py` so the run is reproducible:

```bash
python demo_login.py "%TEMP%\beast-login-demo" qwen3:8b
```

It drives the real agent loop (no mocks) and prints every tool call, approval
decision and observation. Use any tool-capable model as the second argument.

Local sampling means the exact page varies between runs. The authoritative
record is the `[TOOL]` / `[OK]` lines, which report what the tools actually did;
a small local model can describe more than it wrote, so read those lines rather
than the prose summary.

### Phase 3 acceptance

`scripts/phase3_definition_of_done.py` runs the two required questions against
three fixtures — a Python Flask service, a React/TypeScript app and a
multi-language repository — through the real agent loop and a local
`qwen3:8b`, with writes and commands denied:

```bash
python scripts/phase3_definition_of_done.py --model qwen3:8b
python scripts/phase3_benchmark.py
```

The transcripts are checked in under [`docs/phase3-dod/`](docs/phase3-dod/).
Each answer cites files, symbols and versions that exist in the fixture; the
React and multi-language fixtures contain no authentication, and BEAST says so
instead of inventing one. Measured performance, per-module coverage and the
full design are in [docs/PHASE3.md](docs/PHASE3.md).

### Phase 4 acceptance

`scripts/phase4_definition_of_done.py` runs the four required demonstrations —
a task-management app, a dark-mode change, finding/fixing a bug, and a
validation failure that reaches the correction limit:

```bash
python scripts/phase4_definition_of_done.py
```

By default the model decisions are scripted while the task graph, tools,
approval protocol, change tracker, validation engine, subprocesses and failure
parsing are real, so the demos are deterministic. The transcripts are checked in
under [`docs/phase4-dod/`](docs/phase4-dod/); each header says whether it came
from a scripted or a real model. The real-model path is opt-in:

```bash
python scripts/phase4_definition_of_done.py --real --model qwen3:8b --only a
```

Real-model results depend on the installed model and available memory; failed or
skipped real runs are reported honestly and never counted as passes.

## Testing with different Ollama models

Any locally installed chat model works. To try another one:

```bash
ollama pull llama3.2:3b
cd C:\path\to\your-project
ASAP launch beast
```

Pick it in the list, or skip the picker:

```bash
ASAP launch beast --model llama3.2:3b
ASAP launch beast --list-models
```

Because discovery is runtime-based, models installed while BEAST is running
show up after `Ctrl+R` in the picker. A cloud variant (`name:cloud`) is
selectable like any local model, as long as `ollama signin` has been run.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m pytest --cov=beast --cov-report=term-missing
```

The suite (553 tests) uses `httpx.MockTransport` for every HTTP interaction and
Textual's headless `run_test` harness for the UI, so it needs neither a running
Ollama nor an interactive terminal. Tool tests run against real temporary
directories and real subprocesses. `tests/data/tags_real.json` is a captured
`/api/tags` response from Ollama 0.33.2, including its `"families": null`
quirk.

Current result: **553 collected: 551 passed, 0 failed, 2 skipped (symlinks
where the platform refuses to create links), 91% statement coverage** (7250
statements, 679 missed). The Phase 3 security regressions run on the real
filesystem: the junction test creates an actual `mklink /J` and proves the
outside file is never indexed, and the cache tests write tampered JSON and
assert it is discarded.

The Phase 4 definition-of-done demos A–D pass against the real runtime with
scripted model decisions; see [`docs/phase4-dod/`](docs/phase4-dod/). A
real-model attempt was made once and did not pass (see that directory's README
for the exact output); it was not retried.

The mouse-selection crash that could occur while a long reply streamed is
documented, with its root cause and regression tests, in
[`docs/TUI_MOUSE_CRASH.md`](docs/TUI_MOUSE_CRASH.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `asap` is not recognised | Add pip's per-user scripts directory to `PATH`, open a new terminal |
| "Ollama doesn't appear to be installed" | Install from https://ollama.com/download and open a new terminal |
| "Could not reach the Ollama service" | Start Ollama: `ollama serve` (or the desktop app) |
| "No chat-capable models are available" | `ollama pull <model>` |
| "Ollama Cloud sign-in required" | Run `ollama signin` in another terminal, then retry |
| The model chats but never changes files | Check the `TOOLS` column (`ASAP launch beast --list-models`) or the picker label. `no`/`?` means Ollama does not report tool calling for that model; BEAST explains this and chats without tools. Select a model that advertises `tools`. |
| A command keeps running | Every command has a timeout (the model may ask for one; BEAST caps it at 120 s). Press `Esc` to stop the turn — the process tree is terminated. |
| A write or command was skipped | The approval dialog was rejected, dismissed or never answered; only an explicit `y`/`a`/Approve counts. The model is told the action was rejected. |
| "Ollama could not start the model" | The runner crashed (usually GPU driver or RAM). Try `ollama run <model>` in another terminal, or a smaller model |
| Reply timed out | The model may still be loading or too large for available memory |
| Interface refuses to start in CI / redirected output | BEAST needs an interactive terminal; use `--list-models` otherwise |
| Need details | `ASAP launch beast --debug`, and read the log file path printed on errors |
| "Python package … is missing" | The install is incomplete; run `python -m pip install -e .` again from the BEAST directory |

## Privacy

All inference happens through the Ollama service on your machine. BEAST
contains no cloud endpoints, no API-key handling and no analytics. The only
network traffic BEAST itself generates is to the Ollama URL you configure
(loopback by default, overridable with `--ollama-url` or `BEAST_OLLAMA_URL`).
A shell command you approve runs as an ordinary process and may make its own
network connections, just as it would from your terminal.

## License

MIT — see [LICENSE](LICENSE).
