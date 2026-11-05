"""
The dashboard container installs only requirements-dashboard.txt. Every
third-party package the dashboard's own code imports must be listed there, or
the deployed server fails to boot (as it did when the satellite layer brought
in Pillow). Imports guarded by `except ImportError` are optional and skipped.
"""
from __future__ import annotations

import ast
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Imported somewhere in a loaded module, but deliberately not listed.
NOT_NEEDED = {
    "flask": "installed with dash",
    "werkzeug": "installed with dash (via flask)",
    "sklearn": "only the offline training in flood_chance.main(); the dashboard reads the saved fit",
}


def _optional_imports(tree: ast.AST) -> set[int]:
    """Line numbers of imports inside `try: ... except ImportError`."""
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError")
                for h in node.handlers):
            for sub in node.body:
                for n in ast.walk(sub):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        lines.add(n.lineno)
    return lines


def _third_party_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    optional = _optional_imports(tree)
    names = set()
    for n in ast.walk(tree):
        if n.lineno in optional if hasattr(n, "lineno") else False:
            continue
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names.add(n.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {"src", "__future__"}


def test_every_package_the_dashboard_imports_is_in_the_container():
    # A fresh interpreter, so modules other tests imported are not counted:
    # exactly what the server loads when gunicorn imports the app.
    probe = ("import sys, src.dashboard.app; "
             "print(chr(10).join(m.__file__ for n, m in list(sys.modules.items())"
             " if n.startswith('src.') and getattr(m, '__file__', None)))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True,
                         text=True, check=True, timeout=600).stdout
    files = {Path(line) for line in out.splitlines() if line.endswith(".py")}
    assert any(f.name == "app.py" for f in files), out[-2000:]
    dists = md.packages_distributions()
    listed = {re.split(r"[=<>\[ ]", line.strip())[0].lower().replace("_", "-")
              for line in (ROOT / "requirements-dashboard.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith(("#", "-"))}
    missing = {}
    for f in files:
        for mod in _third_party_imports(f) - NOT_NEEDED.keys():
            wanted = [d.lower().replace("_", "-") for d in dists.get(mod, [mod])]
            if not any(d in listed for d in wanted):
                missing.setdefault(mod, []).append(f.relative_to(ROOT).as_posix())
    assert not missing, f"imported by the dashboard but not in requirements-dashboard.txt: {missing}"
