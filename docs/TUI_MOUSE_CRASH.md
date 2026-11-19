# TUI crash on mouse interaction while a reply streams

## Symptom

BEAST could abort while a long reply was being streamed from a local Ollama
model. The failure only occurred when the terminal also sent a mouse event:

```text
AttributeError: 'NoneType' object has no attribute 'region'
  textual/app.py:4082          App._on_event / event dispatch
  textual/screen.py:1914       Screen._forward_event
  beast/ui/screens.py          ChatScreen (inherits BeastScreen)
  textual/widgets/_markdown.py MarkdownParagraph
  textual/events.py            MouseDown
```

## Environment

- BEAST on Windows 11, Python 3.14.7.
- Textual **8.2.8** (the installed version; its source was not modified).
- A long, growing assistant reply rendered as Markdown, streamed in throttled
  chunks by `ChatScreen._stream_turn`.

## Root cause

Two behaviours combined.

1. **Every stream tick rebuilt the whole Markdown document.**
   `AssistantMessage.update_body()` called `Markdown.update(cumulative_text)`.
   In Textual 8.2.8 `Markdown.update()` removes *all* existing
   `MarkdownBlock` widgets and mounts a fresh set. During that remove/mount
   window the compositor can still hold geometry for a block that has already
   been detached from the DOM.

2. **Textual's selection path assumes a hit widget is still attached.**
   On `MouseDown`, `Screen._forward_event` calls
   `get_widget_and_offset_at()`. If the compositor returns a detached
   `MarkdownParagraph` with a content offset, the screen does:

   ```python
   content_widget = select_widget
   container = content_widget.parent      # None: the block was detached
   ...
   SelectStart(container, ..., container.region.offset, ...)
   ```

   `container` is `None`, so `container.region` raises the `AttributeError`.
   This is upstream Textual issue **#6643** (detached widget returned by the
   compositor during a selection hit test), observed in 8.2.8.

Re-rendering the entire document on every tick is also what made the race
common: a long reply triggers many remove/mount cycles per second, and a user
moving or clicking the mouse has many chances to land in one.

## Fix

Both layers are addressed. No Textual source was patched, no exception is
swallowed, and mouse selection is still enabled.

### 1. Stream incrementally instead of rebuilding

`beast/ui/widgets.py` — `AssistantMessage` now remembers what it has already
handed to the Markdown widget (`_rendered`) and appends only the new suffix:

```python
fragment = markdown_text[len(self._rendered):]
self._rendered = markdown_text
if fragment:
    await markdown.append(fragment)
```

`Markdown.append()` parses only the new lines and keeps already-mounted blocks,
so ordinary paragraphs no longer detach on every tick. A full `update()` is
still used for the defensive case where the caller sends replacement text
rather than a longer cumulative string. Markdown rendering, fenced code
highlighting and bottom-follow scrolling are unchanged.

### 2. Treat a detached widget hit as empty space

`beast/ui/screens.py` — `BeastScreen` (the base of `ChatScreen` and every other
BEAST screen) overrides the hit test:

```python
def get_widget_and_offset_at(self, x, y):
    widget, offset = super().get_widget_and_offset_at(x, y)
    if widget is not None and not widget.is_attached:
        return None, None
    return widget, offset
```

A widget that is no longer in the DOM cannot participate in a selection.
Returning `(None, None)` routes Textual through its existing "nothing
selectable here" branch (`_select_state = None`) instead of constructing an
invalid `SelectStart`. This is a validity check on the hit test, not an
exception handler, and it applies only to the stale window; clicks on attached
widgets still start selections normally.

## Regression tests

`tests/test_ui_markdown_stream.py`:

| Test | What it proves |
|---|---|
| `test_mouse_down_on_a_detached_markdown_block_is_ignored` | Deterministically waits for the compositor to return a just-detached `MarkdownParagraph`, feeds the `MouseDown` that previously crashed, then verifies a click on the rebuilt document still starts a selection. |
| `test_streaming_keeps_earlier_markdown_blocks_mounted` | Incremental streaming leaves the first paragraph mounted and the Markdown source equal to the cumulative reply. |
| `test_mouse_events_during_and_after_a_streamed_reply` | A long streamed reply tolerates mouse down/up during streaming and after completion, renders its fenced code block, and still handles clicks after a terminal resize. |

The first test fails with the exact `AttributeError` against the unguarded hit
test, and passes with the fix. The selection assertion in the same test guards
against "fixing" the crash by disabling mouse interaction.

Full suite after the fix (Windows 11, Python 3.14.7):

```text
551 passed, 2 skipped in 115.64s
statement coverage: 91% (7250 statements, 679 missed)
```

The two skips are the pre-existing symlink tests that the platform refuses to
create. No test is deselected or ignored.

## Limits

- The guard handles the general stale-compositor case but does not make
  Textual's compositor snapshot atomic; upstream #6643 still belongs in
  Textual. If a future Textual release fixes it, the guard becomes a harmless
  extra validity check.
- Incremental `Markdown.append()` assumes callers send a growing cumulative
  string. `AssistantMessage` keeps the replacement path for any other input.
- Manual testing used Textual's headless harness plus scripted chunk streams;
  the interaction was not re-run against a live Ollama model in a real
  terminal.
