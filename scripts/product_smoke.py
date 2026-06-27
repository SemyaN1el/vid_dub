from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ".dockerignore",
    ".env.example",
    "Dockerfile.online",
    "docker-compose.yml",
    "INSTALL.md",
    "main.py",
    "config.example.py",
    "requirements.txt",
    "requirements-api.txt",
    "requirements-ci.txt",
    "requirements-online.txt",
    "pyproject.toml",
    "api/app.py",
    "api/jobs.py",
    "api/schemas.py",
    "web/index.html",
    "web/styles.css",
    "web/app.js",
    "scripts/smoke_pipeline.py",
    "tests/unit/test_api_jobs.py",
    "tests/unit/test_smoke_pipeline.py",
)

SMOKE_UNIT_TESTS = (
    "tests/unit/test_pipeline_io.py",
    "tests/unit/test_config_snapshot.py",
    "tests/unit/test_pipeline_resume.py",
    "tests/unit/test_reporting.py",
    "tests/unit/test_api_jobs.py",
    "tests/unit/test_smoke_pipeline.py",
)

COMPILE_ROOTS = (
    "main.py",
    "src",
    "utils",
    "scripts",
    "desktop",
    "api",
    "tests/unit",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for relative in COMPILE_ROOTS:
        path = PROJECT_ROOT / relative
        if path.is_file() and path.suffix == ".py":
            files.append(path)
            continue
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*.py")
                if "__pycache__" not in candidate.parts
            )
    return sorted(files)


def check_required_files() -> CheckResult:
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        return CheckResult("required files", False, ", ".join(missing))
    return CheckResult("required files", True, f"{len(REQUIRED_FILES)} files")


def check_config_file() -> CheckResult:
    path = PROJECT_ROOT / "config.py"
    if path.is_file():
        return CheckResult("local config.py", True, str(path))
    return CheckResult(
        "local config.py",
        False,
        "missing; create it with: Copy-Item config.example.py config.py",
    )


def check_compile() -> CheckResult:
    files = iter_python_files()
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            return CheckResult(
                "python compile",
                False,
                f"{path.relative_to(PROJECT_ROOT)}: {exc.msg}",
            )
    return CheckResult("python compile", True, f"{len(files)} files")


def run_command(name: str, command: list[str]) -> CheckResult:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        tail = output[-800:] if output else f"exit code {result.returncode}"
        return CheckResult(name, False, tail)
    first_line = output.splitlines()[0] if output else "ok"
    return CheckResult(name, True, first_line[:160])


def check_show_config() -> CheckResult:
    return run_command(
        "main.py --show-config",
        [sys.executable, "main.py", "--show-config"],
    )


def check_unit_tests() -> CheckResult:
    missing = [
        relative
        for relative in SMOKE_UNIT_TESTS
        if not (PROJECT_ROOT / relative).is_file()
    ]
    if missing:
        return CheckResult("smoke unit tests", False, "missing: " + ", ".join(missing))
    return run_command(
        "smoke unit tests",
        [sys.executable, "-m", "pytest", *SMOKE_UNIT_TESTS],
    )


def print_results(results: list[CheckResult]) -> None:
    width = max(len(result.name) for result in results)
    for result in results:
        status = "OK" if result.ok else "FAIL"
        detail = f"  {result.detail}" if result.detail else ""
        print(f"{result.name.ljust(width)}  {status}{detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight product-packaging smoke checks.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip Python syntax compilation.",
    )
    parser.add_argument(
        "--no-show-config",
        action="store_true",
        help="Skip main.py --show-config.",
    )
    parser.add_argument(
        "--no-unit",
        action="store_true",
        help="Skip lightweight unit tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        check_required_files(),
        check_config_file(),
    ]
    if not args.no_compile:
        results.append(check_compile())
    if not args.no_show_config:
        results.append(check_show_config())
    if not args.no_unit:
        results.append(check_unit_tests())

    print_results(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
