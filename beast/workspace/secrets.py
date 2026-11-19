"""Secret and credential protection.

BEAST hands project text to a language model, so two rules apply everywhere in
the intelligence layer:

1. Credential-bearing files (``.env``, private keys, cloud credentials, token
   stores) are never included in automatically gathered context and are
   refused by the read tools.
2. High-confidence secret values found inside ordinary files are redacted
   before the text leaves the process.  Redaction is conservative: it targets
   recognisable token formats and ``key = "literal"`` assignments.  It cannot
   prove a file is secret-free; it removes the obvious leaks.

The file on disk is never modified by redaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Whole-file names that contain credentials.  Template files are allowed.
_SENSITIVE_NAMES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".htpasswd",
        "credentials",
        "credentials.json",
        "credentials.yaml",
        "credentials.yml",
        "credentials.toml",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.toml",
        "auth.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "keystore",
        "truststore",
        "local.settings.json",
        "appsettings.json",
        "terraform.tfstate",
        "application_default_credentials.json",
    }
)

#: Template files that are safe to read: they contain placeholders.
_SAFE_ENV_TEMPLATES = frozenset(
    {".env.example", ".env.sample", ".env.template", ".env.dist"}
)

#: Suffixes that are secret material or encrypted key stores.
_SENSITIVE_SUFFIXES = frozenset(
    {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".jks",
        ".keystore",
        ".kdbx",
        ".ppk",
        ".asc",
        ".gpg",
        ".ovpn",
        ".tfstate",
        ".tfvars",
    }
)

#: Directories whose direct children are usually credentials.
_SENSITIVE_PARENTS = frozenset({".aws", ".ssh", ".gnupg", ".docker"})


def is_sensitive_path(path: str | Path) -> bool:
    """True when ``path`` is a credential-bearing file.

    Only the path is inspected; no file content is read.
    """
    candidate = Path(path)
    name = candidate.name.lower()
    if name in _SAFE_ENV_TEMPLATES:
        return False
    # ``.env``, ``prod.env`` and dotenv variants such as ``prod.env.local`` or
    # ``app.env.production``.
    if name.startswith(".env") or name.endswith(".env") or ".env." in name:
        return True
    if name in _SENSITIVE_NAMES:
        return True
    if name.startswith("appsettings.") and name.endswith(".json"):
        return True
    if candidate.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    parts = {part.lower() for part in candidate.parts}
    if parts & _SENSITIVE_PARENTS and (
        name in {"credentials", "config", "auth"} or name.startswith("config.")
    ):
        return True
    # ``service_account.json`` and ``service-account.json`` are the same file
    # under two common spellings.
    normalized = name.replace("_", "-")
    if candidate.suffix.lower() == ".json" and "service-account" in normalized:
        return True
    if candidate.suffix.lower() in {".json", ".yaml", ".yml", ".toml"} and (
        "credential" in name
    ):
        return True
    return False


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Text after redaction and how many secrets were found."""

    text: str
    count: int = 0

    @property
    def changed(self) -> bool:
        return self.count > 0


_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

#: A private-key block cut off by an excerpt window or a byte limit.  Without
#: this, the raw base64 body would escape redaction because no END line exists.
_PARTIAL_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*",
    re.DOTALL,
)

#: Well-known token formats with low false-positive rates.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\brk_live_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
)

#: ``password = "literal"`` and similar assignments in any language, including
#: JSON (``"password": "literal"``), prefixed/camelCase keys (``dbPassword``,
#: ``this.apiToken``) and quoted keys.  A prefix is allowed because real
#: configuration rarely uses a bare key name.
_ASSIGNMENT = re.compile(
    r"(?i)([A-Za-z0-9_.-]*(?:api[_-]?key|apikey|secret(?:[_-]?key)?|"
    r"access[_-]?token|refresh[_-]?token|token|password|passwd|pwd|"
    r"client[_-]?secret|auth[_-]?key|private[_-]?key))"
    r"(['\"]?\s*(?:=>|[:=])\s*)(['\"])([^'\"\n]{6,})\3"
)

#: ``API_KEY=value`` in env files, ``export DB_PASSWORD=value`` in shell
#: scripts, ``password: value`` in YAML, and compose-style
#: ``- POSTGRES_PASSWORD=value`` list entries.  Quoted values are handled by
#: :data:`_ASSIGNMENT`.
_ENV_ASSIGNMENT = re.compile(
    r"(?im)^([ \t]*(?:export[ \t]+)?(?:-[ \t]+)?[A-Za-z0-9_.-]*"
    r"(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL)"
    r"[A-Za-z0-9_.-]*[ \t]*[:=][ \t]*)"
    # A literal secret, not an expression such as ``get_token()``.
    r"([^\s#'\"()\[\]{}]{8,})[ \t]*$"
)

#: Credentials embedded in a URL, for any scheme (``postgresql://``,
#: ``mongodb+srv://``, ``redis://``, ``amqp://`` …), not only HTTP.
_URL_CREDENTIALS = re.compile(
    r"([A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:)([^/\s@]{3,})(@)"
)

_REDACTED = "[REDACTED]"


def redact_secrets(text: str) -> RedactionResult:
    """Replace recognisable secrets in ``text`` with ``[REDACTED]``."""
    if not text:
        return RedactionResult(text, 0)

    count = 0

    def replace_private_key(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[REDACTED PRIVATE KEY]"

    text = _PRIVATE_KEY.sub(replace_private_key, text)
    text = _PARTIAL_PRIVATE_KEY.sub(replace_private_key, text)

    for pattern in _TOKEN_PATTERNS:
        def replace_token(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return _REDACTED

        text = pattern.sub(replace_token, text)

    def replace_assignment(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{_REDACTED}{match.group(3)}"

    text = _ASSIGNMENT.sub(replace_assignment, text)

    def replace_env(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}{_REDACTED}"

    text = _ENV_ASSIGNMENT.sub(replace_env, text)

    def replace_url(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}{_REDACTED}{match.group(3)}"

    text = _URL_CREDENTIALS.sub(replace_url, text)
    return RedactionResult(text, count)


def redact_text(text: str) -> str:
    """Convenience wrapper returning only the redacted text."""
    return redact_secrets(text).text
