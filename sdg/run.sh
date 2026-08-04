#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_PY="${HOME}/isaacsim_4.5.0/bin/python"

cd "${PROJECT_ROOT}"

CONFIGS=(
    "sdg/configs/warehouse.yaml"
    "sdg/configs/hospital.yaml"
    "sdg/configs/office.yaml"
    "sdg/configs/real_room.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo
    echo "=========================================="
    echo "[SDG] Running ${config}"
    echo "=========================================="

    OMNI_KIT_ACCEPT_EULA=YES \
    "${ISAAC_PY}" \
        sdg/generate_data.py \
        --config "${config}"
done

echo
echo "[SDG] All three environments completed."