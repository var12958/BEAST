# Phase 3 — Code Intelligence & Project Understanding

Phase 3 gives BEAST an evidence-based understanding of the project it is
working in. Every statement BEAST makes about a project comes from a file it
actually read: a manifest, a configuration file, a source file, or a directory
entry. There is no template text, no per-framework prose and no project-name
heuristics. If the evidence is missing, BEAST says so.

All previous functionality is preserved: the Phase 1 chat interface, the Phase 2
agent loop, the six original tools, approval dialogs and workspace containment
behave exactly as before. Phase 3 adds a seventh, read-only tool
(`project_info`) that is mounted only when project intelligence is present.

---

## 1. Architecture

```
beast/intelligence/
├── __init__.py            ProjectIntelligence facade (thread-safe, incremental)
├── models.py              frozen values: detection, structure, symbols, index, context
├── index_manager.py       scan → index → JSON cache; incremental refresh
├── code_analyzer.py       Python and JavaScript/TypeScript AST analysers + registry
├── project_detector.py    languages, frameworks, managers, build/test tools, entry points
├── structure_analyzer.py  directories, source/test/doc dirs, key files, tree rendering
├── dependency_analyzer.py declared dependencies from ~20 manifest formats
├── context_builder.py     relevance scoring, excerpts, budgets, redaction
└── text.py                tokenisation and identifier splitting
```

Supporting changes live in existing modules rather than new trees:

| Module | Phase 3 role |
|---|---|
| `beast/workspace/scanning.py` | the single walker used by the tools **and** the index: ignored dirs, `.gitignore`, symlink/junction refusal, scan limits |
| `beast/workspace/secrets.py` | credential-file classification and the redactor applied before any text reaches the model |
| `beast/tools/project.py` | the `project_info` tool (seven actions) |
| `beast/tools/registry.py` | mounts `project_info` only when intelligence was supplied |
| `beast/core/agent.py` | optional `ContextProvider`; retrieves context before the THINK → ACT → OBSERVE loop |
| `beast/main.py` / `beast/ui/` | `--project-summary`, project rows in the welcome panel, `[ANALYSIS]` lines |
| `beast/cli.py` | `--project-summary` flag with a stable call contract |

Data flow:

```
workspace files
   │  scan_project (bounded, gitignore-aware, link-safe)
   ▼
IndexManager ──► ProjectIndex (metadata + FileAnalysis per source file)
   │                 │
   │                 └─ JSON cache under BEAST_CACHE_DIR / ~/.beast/cache
   ▼
ProjectProfile = detection + structure + dependency report
   ├──► Agent: context_for(prompt) ──► copied system message, this turn only
   ├──► project_info tool actions
   └──► TUI rows / [ANALYSIS] lines
```

`ProjectIntelligence` is the only object the agent and tools know. Analysis
runs in a worker thread for the TUI; the facade serialises access with an
`RLock` and renders context from immutable snapshots outside the lock.

---

## 2. Project detection

`ProjectDetector.detect(scan_files, dependencies, root=…)` combines three
independent evidence sources and never guesses:

1. **Language counts** from scanned extensions (`scanning.LANGUAGES`). Only
   production files are counted; test files are reported separately.
2. **Dependency names** from the dependency report: exact matches and prefixes
   in `_FRAMEWORKS` (Flask, Django, FastAPI, React, Next.js, Vue, Angular,
   Express, Rails, Laravel, Spring Boot, ASP.NET Core, Gin, Actix Web, …).
3. **Framework-specific files** when a manifest is absent or silent
   (`manage.py` → Django, `next.config.*` → Next.js, `angular.json` → Angular,
   `pubspec.yaml` with `sdk: flutter` → Flutter, `CMakeLists.txt` with
   `find_package(Qt…` → Qt, `<Project Sdk="Microsoft.NET.Sdk.Web">` → ASP.NET
   Core).

Package managers come from root manifests (`requirements*.txt`, `pyproject.toml`,
`setup.py`, `Pipfile`, `package.json`, `pom.xml`, `build.gradle`, `Cargo.toml`,
`go.mod`, `Gemfile`, `composer.json`, `pubspec.yaml`, `Package.swift`,
`build.sbt`, `mix.exs`), with lockfiles overriding a manifest default
(`yarn.lock` makes the manager Yarn, not npm). Build systems cover CMake, Make,
Meson, Bazel, Maven, Gradle, Cargo, Go modules, sbt, SwiftPM, Ninja, Autotools,
Vite/webpack/Rollup/esbuild, npm scripts and the declared Python build backend.
Test frameworks come from declared dependencies and from test config files
(`pytest.ini`, `conftest.py`, `jest.config.*`, `vitest.config.*`, `cypress.config.*`,
`.rspec`, `phpunit.xml`, …).

Entry points are conventional names plus `package.json` `main`/`module`/`browser`
/`bin` fields, each checked for existence and workspace containment, and
credential-bearing paths are filtered out.

Output is a frozen `ProjectDetection`: name (the directory name), language
statistics, primary language, frameworks, package managers, build systems, test
frameworks, manifests, entry points, config files and source/test file counts.

## 3. Structure analysis

`StructureAnalyzer.build(files, root=…)` walks the indexed relative paths and
produces:

- **directories**: a bounded hierarchy (depth and per-directory file caps),
- **source directories**: parents of source files, ordered by file count,
- **test directories**: from the test-path classifier,
- **documentation**: README-style files and `docs/`-style directories,
- **key files**: entry points, manifests and important configuration first,
- **truncation flags**, so the renderer can say the tree is incomplete.

The tree renderer hides credential-bearing paths and marks elided branches.
`.gitignore` is respected by the shared walker, including negation and nested
files whose patterns are anchored to their own directory; generated directories
are matched case-insensitively (`NODE_MODULES` is ignored on Windows too).

## 4. Code intelligence

`code_analyzer.py` parses Python with `ast` and JavaScript/TypeScript with a
dedicated tokenizer-aware parser, producing, per file:

- language and line count,
- imports (module, names, line, resolved display),
- symbols: classes, functions, methods, constants — with line ranges,
  signatures and the first line of the docstring,
- a parse result (`ok` / error text) so a broken file is reported rather than
  silently omitted.

The analyser registry (`register_analyzer`, `analyzer_for`,
`supported_extensions`) makes new languages additive: a new analyser that
implements the `LanguageAnalyzer` protocol is picked up by the index, the
search and the context builder without touching them. Files in languages with
no analyser are indexed as source with `skipped="no analyser"`.

Symbols are queryable through `ProjectIndex.search`, which matches names,
paths, docstrings and identifier fragments, and through the `project_info`
`symbols`, `search` and `file_symbols` actions.

## 5. Dependency analysis

`DependencyAnalyzer.analyze(root, files=… | relative_paths=…)` reads declared
dependencies only; it never installs, resolves or executes anything. Supported
inputs include:

- Python: `requirements*.txt` (including `-r` includes inside the same tree),
  `pyproject.toml` (PEP 621, optional groups, dependency groups, Poetry,
  Poetry groups, build requirements), `setup.py` literals, `setup.cfg`,
  `Pipfile`, `Pipfile.lock`
- JavaScript/TypeScript: `package.json` runtime/dev/peer/optional dependencies
- Java/Kotlin: `pom.xml` and `build.gradle`/`build.gradle.kts`
- Rust: `Cargo.toml` (runtime, dev, build, workspace)
- Go: `go.mod`
- Ruby: `Gemfile`
- PHP: `composer.json` (runtime and dev)
- Dart/Flutter: `pubspec.yaml`
- Swift: `Package.swift` (named targets and URLs)
- .NET: `*.csproj`/`*.fsproj` package references
- Scala: `build.sbt`
- Elixir: `mix.exs`
- C/C++: `vcpkg.json`

Each dependency records name, version text, scope (runtime/dev/build/peer/
optional), ecosystem and the manifest that declared it. Report rendering groups
by scope and path; unparseable or out-of-workspace manifests produce a warning
instead of a crash or invented data. Malformed container shapes (a string where
a list of dependencies belongs) are ignored rather than iterated character by
character.

## 6. Project context engine

`ContextBuilder.build(query, index=…, profile=…)` decides what the model needs
to know **for this prompt**; it never sends the project wholesale.

1. Tokenise the prompt and split identifiers (`auth_handler` → `auth`, `handler`).
2. Score up to `candidate_pool` files on path, symbol names, imports, docstrings
   and content, with prefix/stem matching and small boosts for entry points.
3. Select files under `max_files`, emit line-numbered excerpts under
   `max_file_characters` and `excerpt_lines`, and stop at `max_characters`.
4. Include structure and dependency sections only when the question asks about
   architecture, technologies or dependencies.
5. Redact every excerpt before it leaves the process.

Defaults (`ContextLimits`): 14,000 characters, 8 files, 4,000 characters per
file, 60 excerpt lines, 40 candidates, 10 symbols per file, 24 dependencies.
The result is a frozen `ContextBundle` (text, files, symbol count, characters,
redactions, truncation) rendered as a `## Project context` section.

The agent merges the bundle into a **copy** of the system message for that turn
only, so conversation history is not polluted and later turns retrieve fresh,
query-specific context. A failed or empty retrieval never blocks the turn.

## 7. Project indexing

`IndexManager` turns a scan into a `ProjectIndex`:

- one `IndexedFile` per scanned file: size, mtime, language, source/binary
  flags, optional `FileAnalysis`, and a skip reason;
- a versioned JSON cache keyed by the workspace root and a limits fingerprint;
- incremental refresh: unchanged files (size + `mtime_ns`) reuse their stored
  analysis; modified files are re-analysed; deleted files disappear; added
  files are analysed once;
- atomic writes and defensive loading: a corrupt or tampered cache is ignored
  and rebuilt, unsafe keys (`../`, absolute paths, drive letters, backslashes,
  empty segments) and credential-bearing paths are dropped;
- no vector database and no external service. `BEAST_CACHE_DIR` selects the
  location; the default is `~/.beast/cache`.

`ProjectIntelligence` throttles refresh to once per `refresh_interval` (5 s) and
invalidates it immediately after the agent writes or edits a file through the
tools, so the next question sees the change without re-scanning on every call.

## 8. Agent integration

The agent accepts an optional `ContextProvider` (structural typing). Before a
turn it calls `context_for(prompt)` in the worker thread, records the result as
`[ANALYSIS] Project context: N files, M symbols, C characters.`, and injects the
rendered text into a copied system message. Nothing about any framework is
hardcoded in the loop; the model still decides which tools to call, in what
order, and when to stop. If no intelligence is configured — for example in the
existing test doubles — behaviour is byte-for-byte the Phase 2 loop.

The `project_info` tool gives the model on-demand access to the same evidence:

| Action | Returns |
|---|---|
| `overview` | languages, frameworks, managers, build/test tools, counts, entry points, manifests, dependency groups, structure tree |
| `structure` | source/test/doc directories, entry points, key files, bounded tree |
| `dependencies` | package managers, manifests, grouped dependencies, warnings |
| `symbols` | every indexed symbol (optionally filtered by `query`) |
| `file_symbols` | one file's language, imports, declarations, docstrings |
| `search` | symbol search across the index |
| `refresh` | incremental re-scan with added/modified/removed/analysed counts |

It is read-only, requires no approval, refuses credential-bearing files and is
redacted like every other path to the model.

## 9. TUI integration

The interface was extended, not redesigned:

- the welcome panel gains five rows — **Language**, **Framework**,
  **Pkg manager**, **Source files**, **Test files** — filled asynchronously, so
  startup is never blocked by a scan;
- one `[ANALYSIS]` line per retrieval reports what was selected (files,
  symbols, characters); a second line reports the same figures after the turn;
- `--project-summary` prints the overview non-interactively and exits, matching
  the existing `--list-models` shape.

No new screens, key bindings or layouts were introduced.

## 10. Performance

Measured on this machine against the three acceptance fixtures (Windows 11,
Python 3.14, warm file cache):

| Fixture | Files | Cold index | Warm index | Overview context | Auth context |
|---|---|---|---|---|---|
| Flask | 6 | 18 ms | 32 ms | 4,203 chars / 21 ms | 3,644 chars / 13 ms |
| React/TS | 8 | 17 ms | 13 ms | 2,926 chars / 14 ms | 2,178 chars / 6 ms |
| Multi-language | 10 | 20 ms | 17 ms | 2,680 chars / 17 ms | 1,953 chars / 9 ms |

Reproduce with `python scripts/phase3_benchmark.py`. Earlier runs on the same
fixtures ranged from 49–89 ms for indexing and 18–71 ms for context builds;
both are dominated by file-system metadata and remain far below a noticeable
delay. Warm refreshes report `added=0, changed=0, removed=0, analysed=0` —
the cache is working, not being rebuilt.

Guards that keep large repositories cheap: 5,000-file / depth-15 scan defaults,
512 KB per analysed source file, 256 KB per config read, 40-file candidate pool,
8-file / 14,000-character context budget, a 5-second refresh interval and a
10-entry refresh throttle in the profile. A monorepo is described usefully
quickly rather than perfectly slowly; truncation is surfaced, never hidden.

## 11. Security

The Phase 3 constraints are enforced at the layer that reads bytes, not only at
the tool boundary:

- **Workspace confinement.** Every path is resolved; symlinks and Windows
  directory junctions are rejected by the walker, the index, the dependency
  analyser, config reads and the tools. Junctions matter because
  `Path.is_symlink()` is `False` for them and `is_dir()` follows them — the
  regression test creates one and proves the outside file is never scanned.
- **Credential files never enter context.** `.env*` (including `*.env.*`
  variants such as `prod.env.local`), `*.pem`, `*.key`, `*.p12`, `*.pfx`,
  `*.jks`, `*.kdbx`, `credentials*`, `secrets.*`, `auth.json`, `.netrc`,
  `.npmrc`, `.git-credentials`, service-account JSON (hyphen or underscore),
  `application_default_credentials.json`, `.docker/config.json`,
  `appsettings*.json`, `local.settings.json`, `terraform.tfstate` and
  `*.tfvars` are classified before content is read. `.env.example`-style
  templates stay readable.
- **Redaction before the LLM.** AWS/GitHub/GitLab/Slack/Stripe/Anthropic/Google/
  SendGrid/npm/JWT token formats; `password = "literal"` assignments in any
  language, including JSON quoted keys (`"client_secret": "…"`),
  prefixed/camelCase keys (`dbPassword`, `this.apiToken`) and shell exports
  (`export DB_PASSWORD=…`); env/YAML/compose assignments; credentials in URLs of
  any scheme (`postgresql://`, `mongodb+srv://`, `redis://`, …); complete **and
  truncated** private-key blocks. Applied to file reads, search hits (the whole
  line is redacted before the 240-character snippet is cut), automatic context
  and `project_info` output, including indexed docstrings and signatures.
  Redaction counts are reported; files on disk are never modified.
- **No lossy rewrites.** `edit_file` refuses a file that is not valid UTF-8
  instead of decoding with replacement and writing the corrupted text back.
- **No execution, no installation.** Nothing in the intelligence layer parses
  with `eval`/`exec`, imports discovered modules, runs discovered scripts or
  invokes a package manager. `setup.py` is read with `ast`, manifests with
  `json`/`tomllib`/`ElementTree`.
- **Bounded scans.** A single directory is listed through a capped iterator
  before it is sorted, so one huge folder cannot defeat the file/depth limits.
- **Cache trust.** The JSON cache is treated as untrusted input: unsafe relative
  keys, sensitive paths and wrong shapes are discarded; a failed load falls back
  to a fresh scan.
- **Failure containment.** Unreadable directories and files, malformed
  manifests, binary look-alikes and unsupported languages degrade to warnings,
  skip reasons and error counts. One bad file cannot abort a scan or a turn.

Redaction is heuristic. It removes recognisable leaks; it cannot prove a file
contains no secret. That limitation is unchanged from Phase 2 and is documented
in the README.

## 12. Testing

```
429 collected: 427 passed, 2 skipped (symlink creation refused by the platform)
TOTAL 93% statement coverage
```

Phase 3 modules:

| Module | Coverage |
|---|---|
| `intelligence/code_analyzer.py` | 91% |
| `intelligence/context_builder.py` | 91% |
| `intelligence/dependency_analyzer.py` | 91% |
| `intelligence/index_manager.py` | 88% |
| `intelligence/models.py` | 94% |
| `intelligence/project_detector.py` | 85% |
| `intelligence/structure_analyzer.py` | 94% |
| `intelligence/text.py` | 95% |
| `workspace/scanning.py` | 85% |
| `workspace/secrets.py` | 95% |
| `tools/project.py` | 97% |

Test files cover Python detection, JS/TS detection, multi-language projects,
framework evidence, directory analysis, symbol extraction for both ASTs,
dependency analysis across every supported manifest (including malformed ones),
context relevance and budgets, incremental indexing (added/modified/deleted),
ignored directories, `.gitignore` anchoring, secret exclusion, JSON and
prefixed-key redaction, workspace and junction containment, cache tampering,
`project_info` redaction, non-UTF-8 edit refusal, agent integration and TUI
rows. All Phase 1/2 tests still pass unchanged.

```
python -m pytest
python -m pytest --cov=beast --cov-report=term-missing
```

## 13. Definition-of-Done evidence

The harness in `scripts/phase3_definition_of_done.py` writes each fixture to a
temporary directory, runs the two required questions through the **real**
agent loop against a local `qwen3:8b`, denies every write and command, and
records the transcript. Fixtures are the same dictionaries the test suite uses.

| Fixture | Overview detected | Q1 context | Q2 context | Transcript |
|---|---|---|---|---|
| Python Flask | Python · Flask · pip · pytest · 2 source / 1 test · `app.py` · Flask 3.0.0, pytest 8.0.0 | 6 files, 11 symbols, 4,203 chars | 3 files, 11 symbols, 3,644 chars | `docs/phase3-dod/flask.md` |
| React/TypeScript | TypeScript · React · npm · Vitest · Vite · 5 source / 1 test · `src/main.tsx` | 8 files, 5 symbols, 2,926 chars | 6 files, 5 symbols, 2,178 chars | `docs/phase3-dod/react.md` |
| Multi-language | Python, C++, Go, TypeScript, Vue · FastAPI, Gin, Vue · pip, npm, Go modules · CMake | 8 files, 1 symbol, 2,680 chars | 7 files, 1 symbol, 1,953 chars | `docs/phase3-dod/multilang.md` |

Representative grounded answers (full text in the transcripts):

- **Flask, Q1** names `app.py`, `auth.py`, `tests/test_auth.py`, the
  `CredentialStore`, `hash_password`, `login_required` and the `X-User` header —
  all of which exist in the files.
- **Flask, Q2** walks the actual login flow (`/login` → `authenticate` →
  `hash_password` → `CredentialStore`), and describes `login_required` checking
  `X-User`. No invented middleware or database.
- **React, Q1** correctly reports the component chain `main.tsx → App.tsx →
  Button.tsx`, the `fetchUser` client that is not wired into the UI, and the
  declared versions.
- **React, Q2** examines the retrieved files (UI components plus a `fetchUser`
  client), states that the project does not implement authentication, and
  explains why each candidate is not auth — rather than inventing any.
- **Multi-language, Q1** separates the FastAPI service, Gin stub, Vue client and
  CMake-built C++ engine, including the fact that they are not yet integrated.
- **Multi-language, Q2** calls `project_info search` for "authentication",
  finds nothing, then correctly reports no authentication anywhere, citing the
  health-only router and the placeholder Go/C++ files.

The model sometimes ends with a follow-up question. The evidence sections above
come from the tool output and context headers, not from prose the model could
have guessed.

## 14. Known limitations

- **Redaction is pattern-based.** Encoded secrets, bespoke token formats and
  secrets split across lines can survive. It is a safety net, not a guarantee.
- **Heuristic detection.** A dependency added only for a test can still make its
  framework appear; a project whose manifest is outside the scan root will not
  be detected from that manifest. File-based fallbacks reduce but do not remove
  ambiguity.
- **Only Python and JS/TS are parsed into symbols.** Other languages are still
  detected, counted, searched by name and read as text; the registry makes more
  analysers additive.
- **TOCTOU.** A file or link can change between the containment check and the
  read. The checks are defensive, not an OS sandbox.
- **`dependency_analyzer.py` is large** (~1,000 lines) because each ecosystem's
  manifest has a different shape. It is organised by ecosystem and is a
  candidate for splitting if it grows further; behaviour is covered by
  ecosystem-specific tests.
- **Context selection is lexical.** It is fast, local and explainable; it does
  not build a call graph or embed vectors, so a question that names nothing
  present in a file may rank a less relevant file first. The `project_info`
  tool lets the model correct that by searching directly.
- **Scan limits are deliberately conservative.** Very large monorepos are
  described from a bounded sample, with truncation reported.
- **Cache location.** `BEAST_CACHE_DIR` is trusted as an operator setting; it is
  not sandboxed.
