"""Security regressions for discovery, redaction and cache handling.

These tests pin behaviour that would otherwise be easy to regress:

* a Windows directory junction is not a symlink, so it must be rejected
  explicitly or the walker reads files outside the workspace;
* every piece of text that leaves the intelligence layer is redacted,
  including symbol docstrings rendered by ``project_info``;
* credential-bearing file names are recognised in their real-world spellings;
* a tampered cache cannot inject paths that are not plain relatives.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from beast.intelligence import ProjectIntelligence
from beast.intelligence.dependency_analyzer import DependencyAnalyzer
from beast.intelligence.index_manager import IndexManager, _safe_relative
from beast.tools import ApprovalRequest, ToolContext
from beast.tools.project import ProjectInfoTool
from beast.workspace import Workspace
from beast.workspace.scanning import is_junction, scan_project
from beast.workspace.secrets import is_sensitive_path, redact_secrets


class Approver:
    async def approve(self, request: ApprovalRequest) -> bool:
        return True


def context_for(root: Path, intelligence=None) -> ToolContext:
    return ToolContext(
        workspace=Workspace(root),
        approver=Approver(),
        intelligence=intelligence,
    )


def synthetic_stripe_key() -> str:
    """Return a Stripe-shaped fake key assembled at runtime.

    The string is built from fragments so no literal credential token appears
    in the source tree: GitHub push protection flags live Stripe key shapes and
    would reject this test file.  The assembled value still matches the
    detector's ``sk_live_`` pattern, so redaction is exercised unchanged.  It
    is not a credential and was never issued by Stripe.
    """
    return "".join(("sk", "_li", "ve_", "51H", "abcdefghijklmnopqrstuv"))


def make_junction(link: Path, target: Path) -> None:
    """Create a Windows directory junction, or skip when unavailable."""
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not is_junction(link):
        pytest.skip("directory junctions are not available on this platform")


# ------------------------------------------------------------------ junctions


def test_a_junction_out_of_the_workspace_is_not_scanned_or_indexed(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text(
        "TOKEN = 'OUTSIDE_MARKER_123'\n", encoding="utf-8"
    )

    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    make_junction(root / "linked", outside)

    assert {file.relative for file in scan_project(root).files} == {"app.py"}

    index, update = IndexManager(root).refresh()

    assert set(index.files) == {"app.py"}
    assert update.total == 1

    intelligence = ProjectIntelligence(root)
    try:
        bundle = intelligence.context_for("where is the outside secret token")
        assert "linked/secret.py" not in bundle.files
        assert "OUTSIDE_MARKER_123" not in bundle.text
    finally:
        intelligence.close()


def test_a_junctioned_manifest_is_not_analysed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "package.json").write_text(
        '{"dependencies": {"left-pad": "1.0.0"}}', encoding="utf-8"
    )

    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    make_junction(root / "vendor", outside)

    report = DependencyAnalyzer().analyze(root)

    assert all("left-pad" != dependency.name for dependency in report.dependencies)
    assert report.manifests == ()


def test_dependency_analyzer_refuses_paths_that_escape_the_workspace(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "package.json").write_text(
        '{"dependencies": {"rogue": "9.9.9"}}', encoding="utf-8"
    )
    root = tmp_path / "project"
    root.mkdir()

    report = DependencyAnalyzer().analyze(
        root, relative_paths=["../outside/package.json"]
    )

    assert report.dependencies == ()
    assert any("outside the workspace" in warning for warning in report.warnings)


# ------------------------------------------------------------------- redaction


def test_project_info_redacts_secrets_found_in_indexed_symbols(
    write_files,
) -> None:
    token = synthetic_stripe_key()
    root = write_files(
        {
            "config.py": (
                "from pathlib import Path\n"
                "\n"
                "def load(path):\n"
                f'    """Load a file. Token: {token}."""\n'
                "    return Path(path).read_text()\n"
            )
        }
    )
    intelligence = ProjectIntelligence(root)
    try:
        tool = ProjectInfoTool(context_for(root, intelligence))
        result = asyncio.run(
            tool.run({"action": "file_symbols", "path": "config.py"})
        )
        symbols = asyncio.run(tool.run({"action": "symbols"}))
    finally:
        intelligence.close()

    assert result.ok is True
    assert token not in result.content
    assert "[REDACTED]" in result.content
    assert token not in symbols.content


def test_non_http_url_credentials_are_redacted() -> None:
    redacted = redact_secrets(
        "DATABASE_URL=postgresql://admin:hunter2pw@db.internal:5432/app\n"
    )

    assert redacted.count == 1
    assert "hunter2pw" not in redacted.text
    assert "postgresql://admin:[REDACTED]@db.internal" in redacted.text


def test_compose_and_yaml_secrets_are_redacted() -> None:
    text = (
        "services:\n"
        "  db:\n"
        "    environment:\n"
        "      POSTGRES_PASSWORD: hunter2secret\n"
        "      - API_TOKEN=abcd1234efgh\n"
    )

    redacted = redact_secrets(text)

    assert redacted.count == 2
    assert "hunter2secret" not in redacted.text
    assert "abcd1234efgh" not in redacted.text


def test_expressions_are_not_mistaken_for_literal_secrets() -> None:
    text = "TOKEN = get_token()\nAPI_KEY = os.environ['X']\n"

    redacted = redact_secrets(text)

    assert redacted.count == 0
    assert redacted.text == text


@pytest.mark.parametrize(
    "text",
    [
        '{"password": "hunter2secret"}',
        '{"client_secret": "GOCSPX-abcdef1234567890"}',
        '{"refresh_token": "1//refresh-token-value"}',
    ],
)
def test_json_credential_assignments_are_redacted(text: str) -> None:
    redacted = redact_secrets(text)

    assert redacted.count == 1
    assert "[REDACTED]" in redacted.text
    assert "hunter2secret" not in redacted.text
    assert "GOCSPX-" not in redacted.text
    assert "1//refresh-token-value" not in redacted.text


def test_prefixed_and_exported_assignments_are_redacted() -> None:
    text = (
        'dbPassword = "hunter2secret"\n'
        'this.apiToken = "abcdef123456"\n'
        "export DB_PASSWORD=hunter2secret\n"
    )

    redacted = redact_secrets(text)

    assert redacted.count == 3
    assert "hunter2secret" not in redacted.text
    assert "abcdef123456" not in redacted.text
    assert 'dbPassword = "[REDACTED]"' in redacted.text
    assert "export DB_PASSWORD=[REDACTED]" in redacted.text


def test_php_arrow_assignments_are_redacted() -> None:
    redacted = redact_secrets("$config = ['password' => 'hunter2secret'];")

    assert redacted.count == 1
    assert "hunter2secret" not in redacted.text


def test_identifiers_that_merely_contain_a_keyword_are_not_redacted() -> None:
    text = 'tokenizer = "plain-value"\npassword_policy_url = "https://example.test"\n'

    redacted = redact_secrets(text)

    assert redacted.count == 0
    assert redacted.text == text


def test_a_truncated_private_key_is_redacted() -> None:
    text = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234567890\n"

    redacted = redact_secrets(text)

    assert redacted.count == 1
    assert "MIIEowIBAAKCAQEA" not in redacted.text
    assert "[REDACTED PRIVATE KEY]" in redacted.text


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        "prod.env",
        "appsettings.json",
        "appsettings.Production.json",
        "terraform.tfstate",
        "infra.tfvars",
        "credentials.json",
        ".docker/config.json",
        "service_account.json",
        "application_default_credentials.json",
        "prod.env.local",
        "app.env.production",
    ],
)
def test_real_world_credential_filenames_are_refused(name: str) -> None:
    assert is_sensitive_path(name) is True


def test_environment_templates_remain_readable() -> None:
    assert is_sensitive_path(".env.example") is False


# --------------------------------------------------------------- cache safety


@pytest.mark.parametrize(
    "relative",
    ["../evil.py", "/abs.py", "C:/evil.py", "a\\b.py", "a/../b.py", "a//b.py", ""],
)
def test_safe_relative_rejects_unsafe_cache_keys(relative: str) -> None:
    assert _safe_relative(relative) is False


def test_safe_relative_accepts_plain_relative_paths() -> None:
    assert _safe_relative("src/app.py") is True
    assert _safe_relative("app.py") is True


def test_tampered_cache_with_drive_letters_is_dropped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    manager = IndexManager(root)
    manager.refresh()

    payload = json.loads(manager.cache_path.read_text(encoding="utf-8"))
    payload["files"]["C:/Windows/system.ini"] = {
        "size": 1,
        "mtime_ns": 1,
        "language": None,
        "is_source": False,
        "is_binary": False,
    }
    manager.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = manager.load()

    assert loaded is not None
    assert "C:/Windows/system.ini" not in loaded.files
