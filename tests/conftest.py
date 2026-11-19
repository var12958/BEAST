"""Shared fixtures.

``tests/data/tags_real.json`` is a captured response from a real Ollama
``GET /api/tags`` call (Ollama 0.33.2).  Keeping the untouched payload in the
repository means the parsers are tested against reality, including quirks such
as ``"families": null`` for cloud models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beast.ollama.models import OllamaModel, OllamaModelList

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def isolated_project_cache(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep every test's project index out of the developer's real cache.

    The cache lives outside ``tmp_path`` so that project fixtures rooted at
    ``tmp_path`` never scan their own index file.
    """
    cache = tmp_path_factory.mktemp("beast-cache")
    monkeypatch.setenv("BEAST_CACHE_DIR", str(cache))
    return cache


@pytest.fixture
def write_files(tmp_path: Path):
    """Write a mapping of relative path -> text/bytes and return the root."""

    def write(mapping: dict[str, str | bytes]) -> Path:
        for relative, content in mapping.items():
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        return tmp_path

    return write


FLASK_FILES: dict[str, str] = {
    "requirements.txt": "Flask==3.0.0\npytest==8.0.0\n",
    "app.py": (
        '"""Application entry point."""\n'
        "from flask import Flask, jsonify, request\n"
        "from auth import authenticate, login_required\n"
        "\n"
        "app = Flask(__name__)\n"
        "\n"
        '@app.route("/login", methods=["POST"])\n'
        "def login():\n"
        '    body = request.get_json(silent=True) or {}\n'
        '    user = authenticate(body.get("username", ""), body.get("password", ""))\n'
        "    if user is None:\n"
        '        return jsonify({"error": "invalid credentials"}), 401\n'
        '    return jsonify({"user": user})\n'
        "\n"
        '@app.route("/private")\n'
        "@login_required\n"
        "def private():\n"
        '    return jsonify({"ok": True})\n'
    ),
    "auth.py": (
        '"""Authentication: password hashing and credential checks."""\n'
        "import hashlib\n"
        "from functools import wraps\n"
        "from flask import jsonify, request\n"
        "\n"
        'SESSION_USER = "user_id"\n'
        "\n"
        "class CredentialStore:\n"
        '    """In-memory credential store used by authenticate()."""\n'
        "\n"
        "    def __init__(self, users):\n"
        "        self._users = dict(users)\n"
        "\n"
        "    def authenticate(self, username, password):\n"
        "        digest = self._users.get(username)\n"
        "        if digest is None:\n"
        "            return None\n"
        "        if hash_password(password) == digest:\n"
        "            return username\n"
        "        return None\n"
        "\n"
        "def hash_password(password: str) -> str:\n"
        '    return hashlib.sha256(password.encode("utf-8")).hexdigest()\n'
        "\n"
        "def authenticate(username: str, password: str):\n"
        "    return _STORE.authenticate(username, password)\n"
        "\n"
        "def login_required(view):\n"
        "    @wraps(view)\n"
        "    def wrapped(*args, **kwargs):\n"
        '        if not request.headers.get("X-User"):\n'
        '            return jsonify({"error": "authentication required"}), 401\n'
        "        return view(*args, **kwargs)\n"
        "    return wrapped\n"
        "\n"
        '_STORE = CredentialStore({"admin": hash_password("secret")})\n'
    ),
    "tests/test_auth.py": (
        "from auth import hash_password\n"
        "\n"
        "def test_hash_is_stable():\n"
        '    assert hash_password("x")\n'
    ),
    "README.md": "# Sample Flask service\n\nA small service with session authentication.\n",
    ".gitignore": "venv/\n__pycache__/\n.env\n",
    ".env": "FLASK_SECRET_KEY=not-for-the-model-123456\n",
}


@pytest.fixture
def flask_project(write_files) -> Path:
    return write_files(FLASK_FILES)


REACT_FILES: dict[str, str] = {
    "package.json": (
        "{"
        '"name":"web",'
        '"dependencies":{"react":"^18.2.0","react-dom":"^18.2.0"},'
        '"devDependencies":{"typescript":"^5.0.0","vite":"^5.0.0","vitest":"^1.0.0"},'
        '"scripts":{"dev":"vite"}'
        "}"
    ),
    "tsconfig.json": '{"compilerOptions":{"strict":true}}',
    "vite.config.ts": 'import { defineConfig } from "vite";\nexport default defineConfig({});\n',
    "src/main.tsx": (
        'import React from "react";\n'
        'import App from "./App";\n'
        "export function mount() {\n"
        "  return <App />;\n"
        "}\n"
    ),
    "src/App.tsx": (
        'import { Button } from "./components/Button";\n'
        "export default function App() {\n"
        '  return <Button label="Hi" />;\n'
        "}\n"
    ),
    "src/components/Button.tsx": (
        "export interface ButtonProps { label: string }\n"
        "export function Button({ label }: ButtonProps) {\n"
        "  return <button>{label}</button>;\n"
        "}\n"
    ),
    "src/api/client.ts": (
        'export async function fetchUser(id: string): Promise<unknown> {\n'
        "  const response = await fetch(`/api/users/${id}`);\n"
        "  return response.json();\n"
        "}\n"
    ),
    "src/App.test.tsx": (
        'import App from "./App";\n'
        'test("renders", () => {});\n'
    ),
}


@pytest.fixture
def react_project(write_files) -> Path:
    return write_files(REACT_FILES)


MULTILANG_FILES: dict[str, str] = {
    "pyproject.toml": (
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        "\n"
        "[project]\n"
        'name = "service"\n'
        'dependencies = ["fastapi>=0.110"]\n'
    ),
    "service/main.py": (
        "from fastapi import FastAPI\n"
        "from service.routes import router\n"
        "\n"
        "app = FastAPI()\n"
        "app.include_router(router)\n"
    ),
    "service/routes.py": (
        "from fastapi import APIRouter\n"
        "\n"
        "router = APIRouter()\n"
        "\n"
        '@router.get("/health")\n'
        "def health():\n"
        '    return {"ok": True}\n'
    ),
    "package.json": (
        '{"name":"console","dependencies":{"vue":"^3.4.0"},'
        '"devDependencies":{"vite":"^5.0.0"}}'
    ),
    "web/src/main.ts": 'import { createApp } from "vue";\ncreateApp({}).mount("#app");\n',
    "web/src/App.vue": "<template><main>Console</main></template>\n",
    "go.mod": (
        "module example.com/service\n"
        "\n"
        "go 1.22\n"
        "\n"
        "require github.com/gin-gonic/gin v1.9.0\n"
    ),
    "cmd/server/main.go": 'package main\n\nfunc main() {}\n',
    "CMakeLists.txt": (
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(multi CXX)\n"
        "add_executable(engine native/engine.cpp)\n"
    ),
    "native/engine.cpp": (
        "class Engine {\n"
        "public:\n"
        "  void start();\n"
        "};\n"
    ),
}


@pytest.fixture
def multilang_project(write_files) -> Path:
    return write_files(MULTILANG_FILES)


@pytest.fixture
def tags_payload() -> dict:
    return json.loads((DATA_DIR / "tags_real.json").read_text(encoding="utf-8"))


@pytest.fixture
def models(tags_payload: dict) -> list[OllamaModel]:
    return OllamaModelList.model_validate(tags_payload).models


@pytest.fixture
def local_model() -> OllamaModel:
    return OllamaModel(
        name="qwen3:8b",
        size=5_232_000_000,
        details={
            "family": "qwen3",
            "parameter_size": "8.2B",
            "quantization_level": "Q4_K_M",
        },
        capabilities=["completion", "tools"],
    )
