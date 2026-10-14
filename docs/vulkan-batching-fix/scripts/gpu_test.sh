#!/usr/bin/env bash
# usage: gpu_test.sh <label> <env assignments...> -- runs general model on AMD, checks GPU health after
set -u
LABEL="$1"; shift
M=${MODEL:-/tmp/opencode/gguf/BiRefNet-F16-Acly.gguf}
IN=/home/danqing/projects/RMBG/research-assets/vision-cpp-rmbg/input/9.png
OUT=/tmp/opencode/out/${OUTDIR:-vulkan}/$LABEL
CLI=${VISION_CLI:-/opt/vision.cpp/build/bin/vision-cli}
mkdir -p "$(dirname "$OUT")"
start=$SECONDS
env "$@" timeout -s KILL "${TMO:-300}" "$CLI" birefnet -b gpu -m "$M" -i "$IN" \
  -o "${OUT}-mask.png" --composite "${OUT}.png" > /tmp/opencode/log/$LABEL.log 2>&1
rc=$?
echo "[$LABEL] rc=$rc wall=$((SECONDS-start))s env: $*"
grep -E "complete|error|Error|assert" /tmp/opencode/log/$LABEL.log | head -3 | sed 's/^/    /'
timeout -s KILL 15 vulkaninfo --summary >/dev/null 2>&1
vrc=$?
if [ $vrc -ne 0 ]; then echo "    !!! GPU DEAD (vulkaninfo rc=$vrc) — STOP"; exit 99; fi
echo "    gpu ok"
exit $rc
