#!/usr/bin/env bash
# 第5轮：在"小批量(n=32, 完全损坏)"基础上只增加批边界（不会产生巨型 cmdbuf，避开挂 GPU 的配置）
set -u
export VISION_CLI=${VISION_CLI:-/tmp/opencode/vision.cpp-src/build/bin/vision-cli}
GT=/tmp/opencode/gpu_test.sh

$GT n32ctrl VISP_NODES_PER_SUBMIT=32 || exit 99          # 期望 IoU 0.161（对照）
$GT n4ctrl  VISP_NODES_PER_SUBMIT=4  || exit 99          # 期望 IoU 1.000（对照）

for op in CONV_2D_DEFORM IM2COL CONV_2D MUL_MAT FLASH_ATTN_EXT UPSCALE MEAN NORM REPEAT GET_ROWS; do
  $GT "n32b_$op" VISP_NODES_PER_SUBMIT=32 VISP_SUBMIT_BEFORE_OP="$op" || exit 99
done
for op in CONV_2D_DEFORM IM2COL MUL_MAT FLASH_ATTN_EXT UPSCALE; do
  $GT "n32a_$op" VISP_NODES_PER_SUBMIT=32 VISP_SUBMIT_AFTER_OP="$op" || exit 99
done
# n=8 是半损坏(IoU 0.40)，单点边界可能就修好
for op in CONV_2D_DEFORM IM2COL CONV_2D MUL_MAT; do
  $GT "n8b_$op" VISP_NODES_PER_SUBMIT=8 VISP_SUBMIT_BEFORE_OP="$op" || exit 99
done
echo "all done"
