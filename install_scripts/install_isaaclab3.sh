#!/bin/bash
# Isaac Lab 3.0 (v3.0.0, kit-less) training environment for SONIC on Newton, with the package versions the
# migration was tested with (docs/source/user_guide/isaaclab3_newton.md).
#
#   bash install_scripts/install_isaaclab3.sh [<Isaac Lab checkout dir>]
#
# Clones Isaac Lab v3.0.0 next to this repo unless the directory exists, creates its .venv with `uv sync --locked`
# (Python 3.12; torch 2.11 cu130, warp 1.17, newton 1.6.1 from Isaac Lab's uv.lock), installs gear_sonic[training]
# into it under isaaclab3_constraints.txt, and fetches the G1 USD from Git LFS. Isaac Sim is not needed for Newton;
# ISAACSIM=1 also installs Isaac Sim 6.2 (Isaac Lab's isaacsim extra) for physics_backend=physx.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IL_DIR="${1:-$REPO_ROOT/../IsaacLab-v3.0.0}"

if ! command -v uv &>/dev/null; then
    # UV_INSTALL_DIR picks where uv goes (default ~/.local/bin); shell profiles are left alone
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
fi
echo "[OK] uv $(uv --version)"

if [ ! -d "$IL_DIR" ]; then
    git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$IL_DIR"
fi
IL_DIR="$(cd "$IL_DIR" && pwd)"
echo "[INFO] Creating $IL_DIR/.venv from Isaac Lab's uv.lock …"
(cd "$IL_DIR" && uv sync --locked ${ISAACSIM:+--extra isaacsim})

echo "[INFO] Installing gear_sonic[training] …"
uv pip install --python "$IL_DIR/.venv/bin/python" -e "$REPO_ROOT/gear_sonic[training]" \
    -c "$SCRIPT_DIR/isaaclab3_constraints.txt"

if git -C "$REPO_ROOT" lfs version &>/dev/null; then
    git -C "$REPO_ROOT" lfs pull --include "gear_sonic/data/assets/robot_description/usd/g1/*"
else
    echo "[WARN] git-lfs not found: fetch gear_sonic/data/assets/robot_description/usd/g1/g1_sonic_newton.usd with"
    echo "       'git lfs pull' before training"
fi

"$IL_DIR/.venv/bin/python" -c "import torch, warp, newton, isaaclab, isaaclab_newton, gear_sonic; \
print('[OK] torch', torch.__version__, '| warp', warp.config.version, '| newton', newton.__version__)"
echo "[OK] Activate with: source $IL_DIR/.venv/bin/activate"
