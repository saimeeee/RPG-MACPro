"""Audit a Linux SMAC image without launching StarCraft or claiming GPU training.

Run with the saved image's .venv-smac/bin/python after the real-game preflight.
All persistent runtime components must resolve to the root filesystem. CUDA
software can be ready in an AutoDL CPU-mode container with no exposed GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig


TRAINING_MAPS = ("5m_vs_6m", "12m", "5m", "8m_vs_9m")
EXPECTED_PACKAGES = {
    "torch": "2.5.1", "numpy": "1.26.4", "PySC2": "4.0.0",
    "protobuf": "3.20.3", "pygame": "2.6.1", "PyYAML": "6.0.2",
    "lbforaging": "2.0.0", "gymnasium": "1.4.0", "matplotlib": "3.9.4",
}
DATA_PREFIXES = ("/root/autodl-tmp", "/root/autodl-fs", "/autodl-tmp", "/data", "/mnt")


def beneath(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def mounts():
    try:
        rows = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return []
    result = []
    for row in rows:
        fields = row.split()
        separator = fields.index("-")
        result.append({"path": unescape_mount(fields[4]), "filesystem": fields[separator + 1],
                       "source": unescape_mount(fields[separator + 2])})
    return result


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def path_record(path, mount_table):
    logical = Path(path).absolute()
    resolved = logical.resolve()
    matched = [m for m in mount_table if beneath(resolved, Path(m["path"]))]
    mount = max(matched, key=lambda m: len(m["path"])) if matched else None
    exists = resolved.exists()
    try:
        same_filesystem = exists and resolved.stat().st_dev == Path("/").stat().st_dev
    except OSError:
        same_filesystem = False
    data_prefix = next((p for p in DATA_PREFIXES if beneath(resolved, Path(p))), None)
    return {
        "logical_path": str(logical), "resolved_path": str(resolved),
        "exists": exists, "is_symlink": logical.is_symlink(), "mount": mount,
        "same_filesystem_as_root": same_filesystem,
        "data_mount_prefix": data_prefix,
        "stored_on_system_disk": bool(exists and same_filesystem and not data_prefix),
    }


def load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value, None
    except (OSError, ValueError) as exc:
        return {}, type(exc).__name__ + ": " + str(exc)


def command(arguments, env=None):
    try:
        result = subprocess.run(arguments, text=True, capture_output=True, timeout=30,
                                check=False, env=env)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__ + ": " + str(exc)}


def audit(args):
    project = args.project.resolve()
    mount_table = mounts()
    errors = []
    notes = []
    if sys.platform != "linux":
        errors.append("Image audit requires Linux; the current interpreter is not Linux")
    if sys.version_info[:2] != (3, 10):
        errors.append("Expected Python 3.10 for this saved image")
    venv = args.venv.resolve() if args.venv else project / ".venv-smac"
    if Path(sys.prefix).resolve() != venv:
        errors.append("Run this audit with the project's .venv-smac/bin/python (or specify --venv)")
    critical = {
        "project": project, "venv": venv, "python_executable": Path(sys.executable),
        "python_base_prefix": Path(sys.base_prefix),
        "python_standard_library": Path(sysconfig.get_path("stdlib")),
        "python_site_packages": Path(sysconfig.get_path("purelib")),
    }
    sc2_setting = os.environ.get("SC2PATH")
    sc2 = Path(sc2_setting).resolve() if sc2_setting else None
    if sc2 is None:
        errors.append("SC2PATH is not set; source results/server_setup/activate.sh")
    else:
        critical["starcraft_installation"] = sc2
    packages = {}
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            distribution = metadata.distribution(name)
            version = distribution.version
            correct = version.split("+", 1)[0] == expected
            packages[name] = {"version": version, "expected": expected, "matches": correct}
            critical["package/" + name] = Path(distribution.locate_file(""))
            if not correct:
                errors.append(f"{name} version {version} differs from expected {expected}")
        except metadata.PackageNotFoundError:
            packages[name] = {"version": None, "expected": expected, "matches": False}
            errors.append("Missing Python package: " + name)
    try:
        protocol = metadata.distribution("s2clientprotocol")
        packages["s2clientprotocol"] = {"version": protocol.version, "installed": True}
        critical["package/s2clientprotocol"] = Path(protocol.locate_file(""))
    except metadata.PackageNotFoundError:
        errors.append("Missing Python package: s2clientprotocol")
    cuda = {"software_expected": "12.1", "gpu_available": False,
            "gpu_validation_requested": args.validate_gpu, "gpu_validated": False}
    try:
        import torch
        critical["torch_module"] = Path(torch.__file__)
        cuda["torch_cuda_version"] = torch.version.cuda
        cuda["gpu_available"] = torch.cuda.is_available()
        if torch.version.cuda != "12.1":
            errors.append("Expected CUDA 12.1 PyTorch wheel; CPU-only torch is not image-ready")
        if cuda["gpu_available"]:
            cuda["device_name"] = torch.cuda.get_device_name(0)
            if args.validate_gpu:
                values = torch.arange(16, dtype=torch.float32, device="cuda")
                cuda["gpu_validated"] = bool(values.sum().item() == 120)
                torch.cuda.synchronize()
        else:
            notes.append("No GPU is exposed. CUDA software readiness can pass in CPU mode; GPU execution remains unvalidated.")
    except Exception as exc:
        cuda["error"] = type(exc).__name__ + ": " + str(exc)
        errors.append("PyTorch import or CUDA software inspection failed")
    binaries = sorted(sc2.glob("Versions/Base*/SC2_x64")) if sc2 else []
    if not binaries:
        errors.append("No Linux SC2_x64 executable exists under SC2PATH")
    library_checks = {}
    for binary in binaries:
        critical["starcraft_binary/" + binary.parent.name] = binary
        entry = command(["ldd", str(binary)])
        output = entry.get("stdout", "") + "\n" + entry.get("stderr", "")
        entry["missing_libraries"] = [line.strip() for line in output.splitlines() if "not found" in line]
        entry["passed"] = bool(entry.get("exit_code") == 0 and not entry["missing_libraries"])
        entry["executable"] = os.access(binary, os.X_OK)
        entry["sha256"] = digest(binary)
        if not entry["passed"] or not entry["executable"]:
            errors.append("SC2 executable or shared libraries are incomplete: " + str(binary))
        library_checks[str(binary)] = entry
    bundled = project / "src/envs/smac/smac/env/starcraft2/maps/SMAC_Maps"
    map_checks = {}
    for name in TRAINING_MAPS:
        source = bundled / (name + ".SC2Map")
        target = sc2 / "Maps/SMAC_Maps" / (name + ".SC2Map") if sc2 else None
        entry = {"bundled_exists": source.is_file(), "installed_exists": bool(target and target.is_file())}
        if source.is_file() and target and target.is_file():
            entry["bundled_sha256"] = digest(source)
            entry["installed_sha256"] = digest(target)
            entry["matches_bundled"] = entry["bundled_sha256"] == entry["installed_sha256"]
            critical["map/" + name] = target
        else:
            entry["matches_bundled"] = False
        if not entry["matches_bundled"]:
            errors.append("Missing or mismatched training map: " + name)
        map_checks[name] = entry
    paths = {name: path_record(path, mount_table) for name, path in critical.items()}
    for name, value in paths.items():
        if not value["stored_on_system_disk"]:
            errors.append("Critical component is missing or resolves outside the system disk: " + name)
    game_report_path = args.game_report or project / "results/server_setup/game-check.json"
    game_report, report_error = load_json(game_report_path)
    played = {row.get("map") for row in game_report.get("rollouts", []) if row.get("steps", 0) > 0}
    report_path_matches = bool(sc2 and game_report.get("starcraft_path")
                               and Path(game_report["starcraft_path"]).resolve() == sc2)
    observed = game_report.get("sc2_games", {})
    observed_versions = {name: observed.get(name, {}).get("game_version") for name in TRAINING_MAPS}
    version_matches = all(isinstance(version, str) and
                          (version == args.game_version or version.startswith(args.game_version + "."))
                          for version in observed_versions.values())
    real_game_ok = bool(not report_error and game_report.get("real_game_validation_passed") is True
                        and all(name in played and observed_versions[name] for name in TRAINING_MAPS)
                        and report_path_matches and version_matches)
    game_evidence = {"path": str(game_report_path), "load_error": report_error,
                     "report_flag": game_report.get("real_game_validation_passed"),
                     "maps_with_real_rollouts": sorted(x for x in played if x),
                     "observed_game_versions": observed_versions,
                     "expected_game_version": args.game_version, "game_versions_match": version_matches,
                     "sc2_path_matches": report_path_matches, "passed": real_game_ok}
    if not real_game_ok:
        errors.append("Real-game validation evidence for all four training maps is missing or does not match SC2PATH")
    smoke = {"requested": bool(args.smoke_summary), "passed": None}
    if args.smoke_summary:
        summary, smoke_error = load_json(args.smoke_summary)
        config, config_error = load_json(args.smoke_summary.parent / "config.json")
        smoke["passed"] = bool(not smoke_error and not config_error and config.get("env") == "sc2"
                               and summary.get("mode") == "training"
                               and summary.get("optimizer_updates", 0) > 0
                               and summary.get("completed_tasks"))
        smoke.update(path=str(args.smoke_summary), load_error=smoke_error,
                     environment=config.get("env"), optimizer_updates=summary.get("optimizer_updates"),
                     completed_tasks=summary.get("completed_tasks"))
        if not smoke["passed"]:
            errors.append("Requested SMAC training smoke summary did not demonstrate completed optimizer updates")
    software_ready = not errors
    gpu_ready = bool(cuda["gpu_validated"])
    return {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "software_ready": software_ready,
        "image_ready_for_future_gpu_boot": software_ready,
        "real_game_validated": real_game_ok, "gpu_validated": gpu_ready,
        "full_paper_experiment_completed": False,
        "paths": paths, "packages": packages, "cuda": cuda,
        "shared_libraries": library_checks, "training_maps": map_checks,
        "game_validation_evidence": game_evidence, "training_smoke": smoke,
        "errors": errors, "notes": notes,
        "next_step": ("Save the system-disk image, boot a GPU instance, and rerun with --validate-gpu before full training."
                      if software_ready and not gpu_ready else
                      "Run the full three-seed experiment after checking the saved reports."
                      if software_ready else "Resolve the listed errors and rerun the audit."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--game-report", type=Path)
    parser.add_argument("--game-version", default="4.10", help="Expected observed SC2 version prefix")
    parser.add_argument("--smoke-summary", type=Path, help="Optional real SMAC training summary.json")
    parser.add_argument("--validate-gpu", action="store_true", help="Also perform a small CUDA computation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args)
    destination = args.output or args.project / "results/server_setup/image-ready.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "software_ready": report["software_ready"],
                      "real_game_validated": report["real_game_validated"],
                      "gpu_validated": report["gpu_validated"], "errors": report["errors"]}, indent=2))
    return 0 if report["software_ready"] and (not args.validate_gpu or report["gpu_validated"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
