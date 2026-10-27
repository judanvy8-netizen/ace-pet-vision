"""Create this project's Python 3.11 environment. Run setup.bat on Windows."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def run(command):
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def check_python():
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("64-bit Python 3.11 is required. Run setup.bat or use py -3.11 setup_env.py.")


def ensure_environment(root):
    """Reuse a valid environment; never delete or replace an existing folder."""
    env = root / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if env.exists():
        if not (env / "pyvenv.cfg").is_file() or not python.is_file():
            raise RuntimeError(".venv exists but is incomplete. Rename it, then run setup.bat again.")
        try:
            output = subprocess.check_output(
                [str(python), "-c", "import sys,struct,json; print(json.dumps([list(sys.version_info[:2]),struct.calcsize('P'),sys.prefix]))"],
                text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            version, pointer_size, prefix = json.loads(output)
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            raise RuntimeError("Cannot use the existing .venv. Rename it and retry.") from exc
        if version != [3, 11] or pointer_size != 8 or Path(prefix).resolve() != env.resolve():
            raise RuntimeError("Existing .venv is not this project's 64-bit Python 3.11 environment. Rename it and retry.")
        print("[1/3] Reusing .venv", flush=True)
    else:
        print("[1/3] Creating .venv with Python 3.11...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    return python


def install_dependencies(python):
    pip = [python, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    print("[2/3] Installing CPU libraries. Internet access is required...", flush=True)
    run([*pip, "install", "--upgrade", "pip", "--timeout", "60"])
    run([*pip, "install", "torch==2.8.0+cpu", "torchvision==0.23.0+cpu",
         "--index-url", "https://download.pytorch.org/whl/cpu", "--timeout", "60"])
    run([*pip, "install", "-r", ROOT / "requirements.txt", "--timeout", "60"])


def verify_environment(python):
    print("[3/3] Checking installed libraries...", flush=True)
    run([python, "-m", "pip", "check"])
    # Imports check native DLL loading as well as package metadata.
    run([python, "-c", "import sys,torch,torchvision,cv2,numpy,pandas,PIL; from importlib.metadata import version; "
         "print('Python:',sys.version.split()[0]); print('Interpreter:',sys.executable); "
         "print('PyTorch:',torch.__version__); print('Ultralytics:',version('ultralytics')); "
         "print('OpenCV:',cv2.__version__); print('Environment ready.')"])


def main():
    parser = argparse.ArgumentParser(description="Set up ACE's Python 3.11 virtual environment")
    parser.add_argument("--venv-only", action="store_true", help="Create/reuse .venv without installing packages")
    parser.add_argument("--check", action="store_true", help="Check an existing environment without installing packages")
    args = parser.parse_args()
    if args.venv_only and args.check:
        parser.error("Use --venv-only or --check, not both.")
    check_python()
    if args.check and not (ROOT / ".venv").exists():
        raise RuntimeError("No .venv found. Run setup.bat first.")
    python = ensure_environment(ROOT)
    if args.venv_only:
        print("Virtual environment ready. Libraries are NOT installed by --venv-only.")
        return 0
    if not args.check:
        install_dependencies(python)
    verify_environment(python)
    print("\nIntelliJ: select this interpreter as your Python SDK:")
    print(python)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        print("Fix the error above, then run setup.bat again. Existing files are kept.", file=sys.stderr)
        raise SystemExit(1)
