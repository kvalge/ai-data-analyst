# ast_check.py

"""Static deny-list for generated Python. Parse only; never run the source."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

_DENIED_MODULES = frozenset({"subprocess", "socket", "requests", "http", "urllib"})
_DENIED_CALLS = frozenset(
    {
        "eval",
        "exec",
        "__import__",
        "builtins.eval",
        "builtins.exec",
        "builtins.__import__",
        "os.system",
        "os.popen",
    }
)
_DENIED_OS_NAMES = frozenset({"system", "popen"})
_WRITE_METHODS = frozenset(
    {
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "to_excel",
        "to_json",
        "savefig",
    }
)
_PATH_KEYWORDS = frozenset(
    {"path", "file", "fname", "filename", "filepath", "path_or_buf"}
)


class PythonAstError(ValueError):
    """Generated Python did not parse or failed the deny-list."""


def check_python_ast(source: str) -> None:
    """Parse `source` and reject deny-listed imports, calls, and path writes.

    Does not exec or eval the source. Call this before spawn (4.6).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _LOG.info("python ast parse failed")
        raise PythonAstError("Generated Python could not be parsed.") from exc
    _DenyList().visit(tree)


def _is_denied_module(name: str) -> bool:
    """True when `name` is a denied module or a submodule of one."""
    root = name.split(".", 1)[0]
    return root in _DENIED_MODULES


def _is_escaped_path(value: str) -> bool:
    """True when a literal path is absolute or climbs with `..`."""
    candidate = Path(value)
    return candidate.is_absolute() or ".." in candidate.parts


def _string_constants(node: ast.AST) -> list[str]:
    """Return string literals under `node`. Does not evaluate expressions."""
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
    return values


class _DenyList(ast.NodeVisitor):
    """Walk the tree. Bound names from `import x as y` map back to `x`."""

    def __init__(self) -> None:
        self._modules: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _is_denied_module(alias.name):
                raise PythonAstError(
                    f"Import of {alias.name.split('.', 1)[0]} is not allowed."
                )
            bound = alias.asname or alias.name.split(".", 1)[0]
            self._modules[bound] = alias.name.split(".", 1)[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise PythonAstError("Relative imports are not allowed.")
        module = node.module or ""
        if _is_denied_module(module):
            raise PythonAstError(
                f"Import of {module.split('.', 1)[0]} is not allowed."
            )
        root = module.split(".", 1)[0]
        for alias in node.names:
            if alias.name == "*" and root == "os":
                raise PythonAstError("Star import from os is not allowed.")
            if root == "os" and alias.name in _DENIED_OS_NAMES:
                raise PythonAstError("os.system / os.popen is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._dotted(node.func)
        if dotted in _DENIED_CALLS:
            raise PythonAstError(f"{dotted} is not allowed.")
        method_open = self._open_is_method(node)
        if self._is_open_call(node, dotted) and _open_is_write(
            node, method=method_open
        ):
            self._reject_escaped_paths(node, skip_first_arg=method_open)
        elif dotted is not None and dotted.rsplit(".", 1)[-1] in _WRITE_METHODS:
            self._reject_escaped_paths(node)
        self.generic_visit(node)

    def _reject_escaped_paths(
        self, node: ast.Call, *, skip_first_arg: bool = False
    ) -> None:
        for value in _write_path_strings(node, skip_first_arg=skip_first_arg):
            if _is_escaped_path(value):
                raise PythonAstError(
                    "Writes outside the sandbox work dir are not allowed."
                )

    def _dotted(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self._modules.get(node.id, node.id)
        if isinstance(node, ast.Call):
            return self._dotted(node.func)
        if isinstance(node, ast.Attribute):
            base = self._dotted(node.value)
            if base is None:
                return node.attr
            return f"{base}.{node.attr}"
        return None

    def _is_open_call(self, node: ast.Call, dotted: str | None) -> bool:
        """True for builtin open(...), module.open(...), or Path.open(...)."""
        func = node.func
        if isinstance(func, ast.Name):
            return dotted in {"open", "builtins.open"}
        return isinstance(func, ast.Attribute) and func.attr == "open"

    def _open_is_method(self, node: ast.Call) -> bool:
        """True when .open is on a non-module receiver (Path(...).open).

        Builtin `open(...)` is a Name. `builtins.open(...)` is a module
        attribute and uses the same (file, mode) slots as the builtin.
        """
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "open":
            return False
        return not self._receiver_is_module(func.value)

    def _receiver_is_module(self, node: ast.expr) -> bool:
        """True when the receiver is an imported module (or builtins)."""
        if isinstance(node, ast.Call):
            return False
        dotted = self._dotted(node)
        if dotted is None:
            return False
        if dotted == "builtins" or dotted.startswith("builtins."):
            return True
        if dotted in self._modules.values():
            return True
        root = dotted.split(".", 1)[0]
        return root in self._modules.values()


def _write_path_strings(
    node: ast.Call, *, skip_first_arg: bool = False
) -> list[str]:
    """Path-like string literals on a write call. Not the written payload."""
    found: list[str] = []
    if node.args and not skip_first_arg:
        found.extend(_string_constants(node.args[0]))
    for keyword in node.keywords:
        if keyword.arg in _PATH_KEYWORDS:
            found.extend(_string_constants(keyword.value))
    if isinstance(node.func, ast.Attribute):
        found.extend(_string_constants(node.func.value))
    return found


def _open_is_write(node: ast.Call, *, method: bool) -> bool:
    """True when open() uses a write mode. Missing mode is read.

    Builtin and module.open(file, mode=...) store mode at args[1].
    A method .open(mode=...) on a non-module receiver stores mode at args[0].
    """
    mode = "r"
    mode_index = 0 if method else 1
    if len(node.args) > mode_index:
        slot = node.args[mode_index]
        if isinstance(slot, ast.Constant) and isinstance(slot.value, str):
            mode = slot.value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                mode = keyword.value.value
    return any(flag in mode for flag in "wax+")
