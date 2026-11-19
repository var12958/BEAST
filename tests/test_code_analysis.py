"""Symbol and import extraction for the supported languages."""

from __future__ import annotations

from beast.intelligence.code_analyzer import (
    JavaScriptAnalyzer,
    PythonAnalyzer,
    analyze_source,
    analyzer_for,
    supported_extensions,
    supported_languages,
)
from beast.intelligence.models import SymbolKind

PYTHON_SOURCE = '''
"""Example module."""
import os
from typing import Optional

TIMEOUT = 30

class Widget:
    """A renderable widget."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def label(self) -> str:
        return self.name

    def _hidden(self) -> None:
        pass

def helper(value: int) -> Optional[int]:
    """Double a value."""
    return value * 2

async def fetch(url: str):
    return url

def _internal():
    pass
'''


def test_python_symbols_imports_and_signatures() -> None:
    analysis = analyze_source(PYTHON_SOURCE, "example.py")

    assert analysis is not None
    assert analysis.ok is True
    assert analysis.language == "Python"
    assert analysis.lines > 0

    by_name = {symbol.name: symbol for symbol in analysis.symbols}
    assert by_name["Widget"].kind is SymbolKind.CLASS
    assert by_name["Widget"].exported is True
    assert by_name["Widget"].doc == "A renderable widget."
    assert by_name["__init__"].kind is SymbolKind.METHOD
    assert by_name["__init__"].parent == "Widget"
    assert by_name["label"].kind is SymbolKind.METHOD
    assert by_name["label"].decorators == ("@property",)
    assert by_name["_hidden"].exported is False
    assert by_name["helper"].kind is SymbolKind.FUNCTION
    assert by_name["helper"].doc == "Double a value."
    assert "value: int" in by_name["helper"].signature
    assert "Optional[int]" in by_name["helper"].signature
    assert by_name["fetch"].signature.startswith("async def fetch")
    assert by_name["_internal"].exported is False
    assert by_name["TIMEOUT"].kind is SymbolKind.CONSTANT
    assert by_name["TIMEOUT"].display == "TIMEOUT = 30"
    assert by_name["Widget"].display == "Widget"

    imports = {imported.module: imported for imported in analysis.imports}
    assert imports["os"].is_from is False
    assert imports["typing"].is_from is True
    assert imports["typing"].names == ("Optional",)


def test_python_syntax_errors_are_reported_not_raised() -> None:
    analysis = analyze_source("def broken(:\n    pass\n", "broken.py")

    assert analysis is not None
    assert analysis.ok is False
    assert analysis.error.startswith("SyntaxError:")
    assert analysis.symbols == ()


def test_unsupported_languages_have_no_analyser() -> None:
    assert analyzer_for("script.rb") is None
    assert analyze_source("puts 'hi'", "script.rb") is None
    assert ".py" in supported_extensions()
    assert ".ts" in supported_extensions()
    assert {"Python", "JavaScript"} <= set(supported_languages())


JAVASCRIPT_SOURCE = '''
import React from "react";
import { useEffect } from "react";
// function commentedOut() {}
/* class NotReal {} */
export interface Props { id: string }

export class ApiClient extends Base {
  constructor(private base: string) {}
  async fetchUser(id: string): Promise<void> {
    const response = await fetch(`/users/${id}`);
  }
  _secret() {}
}

export async function load(id: string) {
  return id;
}

export const VERSION = "1.0";

const helper = (value: number) => value * 2;

const req = require("node:fs");
'''


def test_javascript_and_typescript_declarations() -> None:
    analysis = JavaScriptAnalyzer().analyze(JAVASCRIPT_SOURCE, "api.ts")

    assert analysis.ok is True
    assert analysis.language == "TypeScript"
    by_name = {symbol.name: symbol for symbol in analysis.symbols}

    assert by_name["Props"].kind is SymbolKind.INTERFACE
    assert by_name["Props"].exported is True
    assert by_name["ApiClient"].kind is SymbolKind.CLASS
    assert by_name["ApiClient"].exported is True
    assert by_name["fetchUser"].kind is SymbolKind.METHOD
    assert by_name["fetchUser"].parent == "ApiClient"
    assert by_name["_secret"].exported is False
    assert by_name["load"].kind is SymbolKind.FUNCTION
    assert by_name["load"].exported is True
    assert by_name["VERSION"].kind is SymbolKind.CONSTANT
    assert by_name["VERSION"].exported is True
    assert by_name["helper"].kind is SymbolKind.FUNCTION
    assert by_name["helper"].exported is False

    assert "commentedOut" not in by_name
    assert "NotReal" not in by_name

    modules = {imported.module for imported in analysis.imports}
    assert {"react", "node:fs"} <= modules
    react = [item for item in analysis.imports if item.module == "react"]
    assert any("useEffect" in item.names for item in react)


def test_javascript_analysis_uses_javascript_language(tmp_path) -> None:
    analysis = analyze_source('const value = require("x");\n', "index.js")
    assert analysis is not None
    assert analysis.language == "JavaScript"
    assert analysis.imports[0].module == "x"


def test_python_analyzer_accepts_plain_functions_without_parents() -> None:
    analysis = PythonAnalyzer().analyze("def ping():\n    return 1\n", "ping.py")

    assert [(symbol.name, symbol.parent) for symbol in analysis.symbols] == [
        ("ping", None)
    ]
