#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies for the Doggo project (Raspberry Pi / Debian family).
# Usage:
#   bash scripts/install_env_deps.sh
#   bash scripts/install_env_deps.sh --no-update

DO_UPDATE=1
for arg in "$@"; do
  case "$arg" in
    --no-update) DO_UPDATE=0 ;;
    *)
      echo "Unknown arg: $arg"
      echo "Usage: bash scripts/install_env_deps.sh [--no-update]"
      exit 2
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script currently supports Debian/Ubuntu/Raspberry Pi OS (apt-get)."
  exit 1
fi

PKGS=(
  python3
  python3-venv
  python3-pip
  xdg-utils
)

if [[ "${DO_UPDATE}" -eq 1 ]]; then
  echo "[deps] apt-get update"
  ${SUDO} apt-get update
fi

echo "[deps] installing: ${PKGS[*]}"
${SUDO} apt-get install -y "${PKGS[@]}"

echo
echo "[deps] done."
echo "Next steps:"
echo "  1) python3 -m venv venv"
echo "  2) source venv/bin/activate"
echo "  3) pip install -U pip"
