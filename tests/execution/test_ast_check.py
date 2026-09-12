# test_ast_check.py

"""Tests for the generated-Python AST deny-list. The checker does not run code."""

from __future__ import annotations

import pytest

from src.execution.ast_check import PythonAstError, check_python_ast

_PANDAS_SNIPPET = """
import pandas as pd

df = pd.DataFrame(
    {"date": ["2024-01-01"], "region": ["North"], "revenue": [10.0]}
)
df.to_csv("summary.csv", index=False)
print(df["revenue"].sum())
"""


def test_allowed_pandas_snippet_passes():
    """A normal pandas analysis snippet is not on the deny-list."""
    check_python_ast(_PANDAS_SNIPPET)


def test_subprocess_import_is_rejected():
    """Importing subprocess is rejected before spawn."""
    with pytest.raises(PythonAstError, match="subprocess"):
        check_python_ast("import subprocess\n")
    with pytest.raises(PythonAstError, match="subprocess"):
        check_python_ast("from subprocess import run\n")


def test_eval_is_rejected():
    """eval is rejected. The checker does not evaluate the argument."""
    with pytest.raises(PythonAstError, match="eval"):
        check_python_ast("eval('1')\n")


def test_os_system_is_rejected():
    """os.system is rejected even when os itself is imported."""
    with pytest.raises(PythonAstError, match="os.system"):
        check_python_ast("import os\nos.system('echo')\n")
    with pytest.raises(PythonAstError, match="os.system"):
        check_python_ast("import os as o\no.system('echo')\n")


def test_write_outside_work_dir_is_rejected():
    """A write whose path climbs out of the work dir is rejected."""
    with pytest.raises(PythonAstError, match="work dir"):
        check_python_ast(
            "from pathlib import Path\n"
            "Path('../out.csv').write_text('x')\n"
        )
    with pytest.raises(PythonAstError, match="work dir"):
        check_python_ast("open('../out.csv', 'w')\n")
    with pytest.raises(PythonAstError, match="work dir"):
        check_python_ast("import builtins\nbuiltins.open('../out.csv', 'w')\n")


def test_path_open_write_outside_is_rejected():
    """Path.open('w') uses the method mode slot, not builtin open(file, mode)."""
    with pytest.raises(PythonAstError, match="work dir"):
        check_python_ast(
            "from pathlib import Path\n"
            "Path('../out.csv').open('w')\n"
        )


def test_unparseable_python_is_rejected():
    """A syntax error is a reject, not a guessed tree."""
    with pytest.raises(PythonAstError, match="parsed"):
        check_python_ast("def (\n")
