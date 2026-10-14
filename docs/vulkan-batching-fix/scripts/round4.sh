#!/usr/bin/env bash
# 重启后第5轮：算子级二分定位"批边界落在哪种算子上会出错"
# 用法：VISION_CLI=/tmp/opencode/vision.cpp-src/build/bin/vision-cli ./round4.sh
set -u
export VISION_CLI=${VISION_CLI:-/tmp/opencode/vision.cpp-src/build/bin/vision-cli}
GT=/tmp/opencode/gpu_test.sh
OFF="VISP_NODES_PER_SUBMIT=999999 VISP_MUL_MAT_BYTES_PER_SUBMIT=0"

# 1) baseline：默认批量，确认 bug 仍在（期望 IoU≈0.16）
$GT baseline || exit 99

# 2) 只在某类算子"之后"提交一次（其余全图一个 cmdbuf）；IoU 显著升高即命中
for op in CONV_2D CONV_2D_DEFORM MUL_MAT IM2COL FLASH_ATTN_EXT UPSCALE ADD MEAN NORM REPEAT CONCAT CPY; do
  $GT "after_${op}" VISP_NODES_PER_SUBMIT=999999 VISP_MUL_MAT_BYTES_PER_SUBMIT=0 VISP_SUBMIT_AFTER_OP="$op" || exit 99
done

# 3) 反向：只在某类算子"之前"提交
for op in CONV_2D CONV_2D_DEFORM MUL_MAT IM2COL FLASH_ATTN_EXT; do
  $GT "before_${op}" VISP_NODES_PER_SUBMIT=999999 VISP_MUL_MAT_BYTES_PER_SUBMIT=0 VISP_SUBMIT_BEFORE_OP="$op" || exit 99
done

# 4) 命中后的复核：nodes=4 必然正确，作为对照
$GT n4 VISP_NODES_PER_SUBMIT=4 || exit 99
echo "all done"
