"""Record server software and game assets without claiming a game test passed."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = Path(os.environ["SC2PATH"]).resolve()
    packages = {}
    for name in ("torch", "numpy", "PySC2", "s2clientprotocol", "protobuf", "pygame", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch
    info = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_sc2_version": os.environ.get("RPG_REQUESTED_SC2_VERSION"),
        "sc2_path": str(path),
        "binary_directories": sorted(p.name for p in (path / "Versions").glob("Base*")),
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "git_head": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": command(["git", "-C", str(root), "status", "--short"]),
        "game_launch_validated": False,
        "note": "Installation inventory only. Run scripts/check_rpg_smac.py --launch before training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "cuda_available": info["cuda_available"]}))


if __name__ == "__main__":
    main()
