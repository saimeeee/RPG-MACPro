#!/usr/bin/env bash
# Set up the uploaded MACPro/RPG checkout on Ubuntu 22.04 x86_64.
# Download/unpack is explicit: Blizzard's archive password accepts its license.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${RPG_PYTHON:-python3.10}"
VENV_DIR="$PROJECT_ROOT/.venv-smac"
SC2_VERSION="4.10"
SC2_PATH=""
ACCEPT_LICENSE=0
SYSTEM_DEPS=0
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
CACHE_DIR="$PROJECT_ROOT/3rdparty/downloads"

usage() {
  cat <<'EOF'
Usage: bash scripts/server/setup_smac.sh [options]
  --python PATH               Python 3.10 interpreter (default: python3.10)
  --venv PATH                 Isolated environment (default: .venv-smac)
  --sc2-version VERSION       4.10 (default) or 4.6.2.69232
  --sc2-path PATH              Reuse an existing StarCraftII installation
  --accept-sc2-license         Accept Blizzard AI/ML license before unpacking
  --system-deps               Install Ubuntu system packages using apt
  --torch-index-url URL       Wheel index (default: official CUDA 12.1 index)
  --cache-dir PATH            Download cache (may live on a data disk)
  --help

Blizzard license: https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html
Existing valid game installations do not need the unpack acceptance flag.
EOF
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --python) (($# >= 2)) || fail '--python requires a value'; PYTHON_BIN="$2"; shift 2 ;;
    --venv) (($# >= 2)) || fail '--venv requires a value'; VENV_DIR="$2"; shift 2 ;;
    --sc2-version) (($# >= 2)) || fail '--sc2-version requires a value'; SC2_VERSION="$2"; shift 2 ;;
    --sc2-path) (($# >= 2)) || fail '--sc2-path requires a value'; SC2_PATH="$2"; shift 2 ;;
    --torch-index-url) (($# >= 2)) || fail '--torch-index-url requires a value'; TORCH_INDEX="$2"; shift 2 ;;
    --cache-dir) (($# >= 2)) || fail '--cache-dir requires a value'; CACHE_DIR="$2"; shift 2 ;;
    --accept-sc2-license) ACCEPT_LICENSE=1; shift ;;
    --system-deps) SYSTEM_DEPS=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This installer requires Linux x86_64.'
case "$SC2_VERSION" in 4.10|4.6.2.69232) ;; *) fail 'Supported versions: 4.10, 4.6.2.69232' ;; esac
if (( SYSTEM_DEPS )); then
  command -v apt-get >/dev/null || fail '--system-deps requires apt-get (Ubuntu/Debian).'
  APT=(apt-get)
  if (( EUID != 0 )); then
    command -v sudo >/dev/null || fail 'Run --system-deps as root or install sudo.'
    APT=(sudo apt-get)
  fi
  "${APT[@]}" update
  "${APT[@]}" install -y --no-install-recommends \
    ca-certificates curl unzip tmux git \
    libstdc++6 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libgl1
  if ! command -v "$PYTHON_BIN" >/dev/null; then
    "${APT[@]}" install -y --no-install-recommends python3.10 python3.10-venv
  fi
fi
for tool in "$PYTHON_BIN" curl unzip sha256sum; do
  command -v "$tool" >/dev/null || fail "Missing $tool. On Ubuntu use --system-deps or --python PATH."
done
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10 for the legacy SC2 stack"'

cd -- "$PROJECT_ROOT"
mkdir -p -- "$VENV_DIR" "$PROJECT_ROOT/results/server_setup"
VENV_DIR="$(cd -- "$VENV_DIR" && pwd)"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,10), "Existing venv is not Python 3.10; choose a new --venv path"'
"$PY" -m pip install --no-cache-dir --disable-pip-version-check pip==25.0.1
"$PY" -m pip install --no-cache-dir --disable-pip-version-check torch==2.5.1 --index-url "$TORCH_INDEX"
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -r "$PROJECT_ROOT/requirements-rpg-smac.txt"
"$PY" -m pip check

mkdir -p -- "$CACHE_DIR"
if [[ -z "$SC2_PATH" ]]; then
  SC2_PATH="$PROJECT_ROOT/3rdparty/sc2-$SC2_VERSION/StarCraftII"
fi
mkdir -p -- "$SC2_PATH"
SC2_PATH="$(cd -- "$SC2_PATH" && pwd)"
shopt -s nullglob
BINARIES=("$SC2_PATH"/Versions/Base*/SC2_x64)
if ((${#BINARIES[@]} == 0)); then
  (( ACCEPT_LICENSE )) || fail 'Game is absent. Rerun with --accept-sc2-license after accepting Blizzard AI/ML terms, or supply --sc2-path.'
  ARCHIVE="$CACHE_DIR/SC2.$SC2_VERSION.zip"
  # Official Blizzard origin, also used by upstream PyMARL. HTTPS is attempted
  # without silently falling back to an unencrypted mirror.
  DOWNLOAD_URL="https://blzdistsc2-a.akamaihd.net/Linux/SC2.$SC2_VERSION.zip"
  if [[ -f "$ARCHIVE" ]] && unzip -t -P iagreetotheeula "$ARCHIVE" >/dev/null 2>&1; then
    printf 'Using complete cached archive: %s\n' "$ARCHIVE"
  else
    printf 'Downloading/resuming %s\n' "$DOWNLOAD_URL"
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
      --output "$ARCHIVE" "$DOWNLOAD_URL"
  fi
  unzip -t -P iagreetotheeula "$ARCHIVE" >/dev/null || fail "Archive integrity check failed: $ARCHIVE. Keep it for diagnosis or move it aside before retrying."
  INSTALL_PARENT="$(dirname -- "$SC2_PATH")"
  [[ "$(basename -- "$SC2_PATH")" == StarCraftII ]] || fail 'New installations require --sc2-path ending in /StarCraftII.'
  # -n preserves previously extracted files on a repeated/partially completed run.
  unzip -n -P iagreetotheeula "$ARCHIVE" -d "$INSTALL_PARENT" >/dev/null
  sha256sum "$ARCHIVE" > "$PROJECT_ROOT/results/server_setup/sc2_archive.sha256"
fi
BINARIES=("$SC2_PATH"/Versions/Base*/SC2_x64)
((${#BINARIES[@]} > 0)) || fail "No SC2 binary found under $SC2_PATH after extraction."
for binary in "${BINARIES[@]}"; do chmod u+x "$binary"; done

MAP_SOURCE="$PROJECT_ROOT/src/envs/smac/smac/env/starcraft2/maps/SMAC_Maps"
MAP_DEST="$SC2_PATH/Maps/SMAC_Maps"
mkdir -p -- "$MAP_DEST"
MAPS=("$MAP_SOURCE"/*.SC2Map)
((${#MAPS[@]} > 0)) || fail "No bundled maps found in $MAP_SOURCE. Upload the complete project."
for source in "${MAPS[@]}"; do
  target="$MAP_DEST/$(basename -- "$source")"
  if [[ -f "$target" ]] && ! cmp -s -- "$source" "$target"; then
    old_hash="$(sha256sum "$target")"; old_hash="${old_hash%% *}"
    backup="$target.before-rpg-${old_hash:0:12}"
    [[ -e "$backup" ]] || cp -p -- "$target" "$backup"
  fi
  cp -p -- "$source" "$target"
done

STATE_DIR="$PROJECT_ROOT/results/server_setup"
{
  printf '# Generated environment for this checkout. Source in each new terminal.\n'
  printf 'source %q\n' "$VENV_DIR/bin/activate"
  printf 'export SC2PATH=%q\n' "$SC2_PATH"
  printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$PROJECT_ROOT/src/envs/smac"
  printf 'export SDL_VIDEODRIVER=dummy\nexport MPLBACKEND=Agg\n'
  printf 'export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n'
} > "$STATE_DIR/activate.sh"
"$PY" -m pip freeze > "$STATE_DIR/pip-freeze.txt"
sha256sum "${BINARIES[@]}" > "$STATE_DIR/sc2_binaries.sha256"
sha256sum "$MAP_DEST"/*.SC2Map > "$STATE_DIR/maps.sha256"
export SC2PATH="$SC2_PATH" RPG_REQUESTED_SC2_VERSION="$SC2_VERSION"
"$PY" "$SCRIPT_DIR/write_manifest.py" --output "$STATE_DIR/installation.json"
printf '\nInstallation files prepared. An actual game launch is still required.\n'
printf 'source %q\n' "$STATE_DIR/activate.sh"
printf 'python scripts/check_rpg_smac.py --launch\n'
