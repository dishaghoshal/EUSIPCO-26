#!/bin/bash

# 1. Configuration Array
CONFIGS=(
    "3.0:20:0.03:0.01"
    "2.0:20:0.03:0.01"
    "2.5:20:0.03:0.01"
    "3.5:20:0.03:0.01"
    "4.0:20:0.03:0.01"
    "3.0:10:0.03:0.01"
    "3.0:15:0.03:0.01"
    "3.0:25:0.03:0.01"
    "3.0:30:0.03:0.01"
    "3.0:20:0.01:0.01"
    "3.0:20:0.02:0.01"
    "3.0:20:0.04:0.01"
    "3.0:20:0.05:0.01"
    "3.0:20:0.06:0.01"
    "3.0:20:0.03:0.005"
    "3.0:20:0.03:0.015"
    "3.0:20:0.03:0.02"
    "3.0:20:0.03:0.03"
)

# 2. Setup
MAX_JOBS=4
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
mkdir -p "plots"

# Verify test.py exists
if [ ! -f "test.py" ]; then
    echo "ERROR: test.py not found in current directory!"
    exit 1
fi

echo "Starting Experiments..."

# 3. Execution Loop
for CONF in "${CONFIGS[@]}"; do
    IFS=":" read -r R L SIG_U SIG_V <<< "$CONF"
    PREFIX="R${R}_L${L}_u${SIG_U}_v${SIG_V}"
    
    echo "[LAUNCHING] $PREFIX"

    # Use python3 -u for unbuffered output to ensure logs fill up in real-time
    python -u test.py \
        --R "$R" \
        --L "$L" \
        --sig_u "$SIG_U" \
        --sig_v "$SIG_V" \
        --output_prefix "$PREFIX" > "${LOG_DIR}/${PREFIX}.log" 2>&1 &

    # Job Management
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
done

wait
echo "Done. If logs are still empty, run: 'python3 test.py --R 3.0 --L 20' manually to see errors."