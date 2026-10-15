# 把 depend/llama 换成最新 llama.cpp 的可行性研究（2026-10-14）

目标：把本仓库 `depend/llama` 从 Acly 固定的 hash（`5fde0fae4`，基线为上游 2026-03-31 的
`624733d63` + 2 个 `[VISION]` 算子提交）换成上游最新（`upstream/master` = `3fb47677e`，
2026-10-14），看"批量提交导致输出损坏"的问题在上游是否已经修好。

## 结论（当前）

**没能确证。** 卡点：vision.cpp 需要的 Vulkan 算子补丁（Acly 的 `[VISION] vulkan` 提交）
无法直接 cherry-pick 到最新 llama.cpp —— ggml-vulkan 在这半年里被大重构（拆成 6 个文件，
`ggml-vulkan.cpp` 1.2 万行改动），16 处冲突里有 3 处是整段错位。CPU 侧补丁可以移植（见下），
Vulkan 侧还没做。

倾向性判断（**未经实测验证**）：上游默认仍然是"每 100 个节点一次提交"
（`device->max_nodes_per_submit = 100`，`ggml-vulkan.cpp:4263`），
依赖同步那套 `overlaps_unsynced` / `unsynced_nodes_*` 结构和旧版基本一致，
上游改动针对的是 DeviceLost/超时（PR #26371、#24872），不是"批量提交算错数"。
所以**大概率还得靠改 nodes_per_submit 绕过**。

## 已经做了什么

1. 拷贝：`/tmp/opencode/vision.cpp-latest/`（来自 `/opt/vision.cpp`，不含 build/）。
2. `depend/llama` 加了 remote `upstream`（https://github.com/ggml-org/llama.cpp.git），
   fetch 了最新 master（shallow）。
3. 分支 `upstream-test`：`upstream/master` + cherry-pick **CPU 侧** vision 提交
   （`ggml-cpu : add conv2d_deform, conv_2d cwhn, concat_n`，落地为 `e50e845aa`）。
   冲突 2 处，解法：
   - `ggml/src/ggml-cpu/ops.cpp`：Acly 用泛型的 `ggml_compute_forward_concat`（支持 N 元 concat）
     替换了上游的 `concat_any/i8/f16/f32` 特化版本 → 采纳 Acly 的做法（删特化版本）。
   - `ggml/src/ggml.c` `ggml_conv_2d`：保留 Acly 的 `ggml_is_contiguous_channels`（cwhn 布局）分支，
     同时把 im2col 的目标类型改成上游写法（BF16→F32，否则 F16）。
4. **Vulkan 侧提交未移植**（见"剩余工作"）。缺它就跑不了非 lite BiRefNet：上游 ggml 没有
   `GGML_OP_CONV_2D_DEFORM`（`ggml.h` 里无此项），vision.cpp 用的是单后端（`ggml_backend_init_by_type`，
   无 `ggml_backend_sched` 回退），缺算子直接失败。

## 上游这半年改了什么（与本 bug 相关）

- **PR #26371**（已合入）"fix submission batching size, add debug tools"：
  批次改成按 flops 预算在"越过阈值前"切断，`mul_mat_bytes_per_submit` 那套被移除；
  新增 `GGML_VK_SERIALIZE_SUBMISSIONS`、`VK_EXT_device_fault` 诊断。
- **PR #24872**（已合入）"allow reducing graph submission batches to avoid device timeouts"：
  `nodes_per_submit` 改名为 `device->max_nodes_per_submit`（默认**仍是 100**），
  环境变量 `GGML_VK_MAX_NODES_PER_SUBMIT` 可覆盖 —— 修的是 #21724 这类 **DeviceLost/超时**，
  不是输出算错。
- ggml-vulkan 拆分：`ggml-vulkan-buffers.cpp` / `-common.h` / `-debug.cpp` /
  `-push-constants.h` / `-types.h`；提交判定在 `ggml-vulkan.cpp:14648`
  （`bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) || ...`）。
- 依赖同步逻辑（`overlaps_unsynced`，`ggml-vulkan.cpp:12287-12358`）结构与旧版一致。

## 合成图探针（失败，记录避免重走）

`/tmp/opencode/vkbatch-test/`：独立 CMake 工程，`add_subdirectory(<llama>/ggml)`，
构一个 conv 链图（im2col+mul_mat 路径，可选 f16 权重），CPU vs Vulkan 对比。

- 构建A（旧）：`build-old/` ← `/tmp/opencode/vision.cpp-src/depend/llama`（`5fde0fae4` + 探针）
- 构建B（新）：`build-new/` ← 最新 upstream master
- **对照失败**：旧代码上把 `VISP_NODES_PER_SUBMIT` 设成 1（已知正确的逐节点提交）和默认 100，
  输出**逐字节相同** —— 说明合成图上的 CPU/GPU 差异是 f16 数值噪声（放大后 rel 0.18），
  与批量提交无关。合成图不复现 BiRefNet 的 bug，因此它**不能**用来判断上游修没修。
- 教训：图里必须有真模型那种节点组合（CONV_2D 直接算子/cwhn 布局、deform、f16 等）才可能复现。

## 剩余工作（要做确证必须完成）

1. 把 Acly 的 `[VISION] vulkan` 提交（`5fde0fae4`，改动：`ggml-vulkan.cpp` 93 行 +
   `conv2d_mm.comp` 59 行 + `im2col.comp` 61 行 + `vulkan-shaders-gen.cpp` 15 行 +
   `ggml-vulkan.h` 2 行 + `ggml.c` 4 行）手工移植到拆分后的上游文件结构。
   16 处冲突中需人工判断的核心是：
   - `conv2d_mm.comp`：deform 的 `bilinear_interpolate` + 边界检查被上游重写（`aligned`/`src_offset`），
     要把 deform 分支接到新代码上；
   - `im2col.comp`：`CWHN` 分支要接到上游改过的 `main()` 循环上；
   - `vulkan-shaders-gen.cpp` 2 处、`ggml-vulkan.cpp` 的 CREATE_CONV 宏 / repeat_f16 /
     im2col elements 计算 / dispatch 各 1 处。
2. `vision.cpp` 本体可能还要跟着 ggml API 漂移改（半年跨度）。
3. 移植后在**同一张卡**上跑三个模型（RMBG-2.0 / BiRefNet_toonout / BiRefNet-F16-Acly），
   默认不加任何环境变量：IoU≈1.0 → 上游已修；IoU≈0.16 → 仍需 nodes_per_submit 绕过。
   对照：同样的模型在旧代码默认值下 IoU 0.161（已验证）。

## 实测结果（2026-10-14，已完成）

移植完成后的分支 `depend/llama` @ `upstream-test`：
- `3fb47677e` upstream master（2026-10-14）
- `+ e50e845aa` CPU vision 补丁（cherry-pick，2 处冲突）
- `+ 5b26abe50` Vulkan vision 补丁（逐处手工移植 16 处冲突 + 补漏）

构建：`cmake -B build -DVISP_VULKAN=ON`（注意开关是 `VISP_VULKAN`，不是 `GGML_VULKAN`），
一次编译通过（含全部 shader）。测试脚本 `/tmp/opencode/newtest.sh`、`/tmp/opencode/newtest-batch.sh`，
产物在 `/tmp/opencode/out/newtest/`。

### 三模型 GPU vs CPU（默认设置，不加任何环境变量）
| 模型 | IoU | 差异像素 |
|---|---|---|
| RMBG-2.0 | 0.999882 | 18 |
| BiRefNet_toonout | 0.999965 | 5 |
| BiRefNet general | 0.999887 | 17 |

RMBG 最终二进制对照 session CPU 参考：IoU 0.999888。每次跑后 vulkaninfo 自检通过。

### 批大小开关（RMBG，新代码）
`GGML_VK_MAX_NODES_PER_SUBMIT = 1 / 8 / 32 / 100` → 输出完全相同（IoU 均 0.999882）。
旧代码同值：8 → 0.43，32/100 → 0.161。

### 插桩证明"不是碰巧批次小"
临时加了 `GGML_VK_LOG_SUBMIT` 日志（测完已 `git checkout` 还原，树是干净的）：
RMBG 一次图 3316 节点被切成 ~65 批，**最大一批 150 个节点**（另有 152/150/149/143…），
远超旧代码 8 节点即坏的阈值，输出依然干净。

### 结论
**上游最新代码上该 bug 不复现，用最新 llama.cpp 不需要再改 nodes_per_submit 绕过。**
fork（/opt）里钉的旧 llama.cpp 上的 `nodes_per_submit=2` 修复继续有效。
上游没有单独的"修结果错误"提交；#26371（按 flops 阈值切批次）+ #24872（上限开关）之后
批次切分与同步逻辑都重写了，属于重构顺带消掉的，具体根因仍未定论。

### 移植时的关键改动清单（供复用）
1. `ggml.c` 两处 `static_assert(GGML_OP_COUNT)` → 102（upstream 101 + CONV_2D_DEFORM）。
2. `ggml-vulkan.cpp` 三大块冲突取 upstream（旧 include/结构体已拆到 -common/-types/-push-constants 头文件），
   Acly 增量另行补进新家：`ggml-vulkan-types.h` 加 `pipeline_im2col_cwhn_f32`、
   `pipeline_conv2d_deform_f32/f16_f32`；`ggml-vulkan.h` 的 `ggml_backend_dev_supports_f16_t` 自动生效。
3. REPEAT 不再需要 repeat_f32/f16：upstream 用 i32/i16 管线按位宽通吃（f32→repeat_i32，f16→repeat_i16），
   所以**没带** Acly 的 repeat_f16 管线（shaders-gen 里也不注册）。
4. `CREATE_CONV(..., n_sources)` 合并 upstream 的 subgroup 参数；deform 用 5 个 source。
5. `conv2d_mm.comp`：deform 分支保留 + `bilinear_interpolate`；非 deform 分支用 upstream 新采样
   （aligned/src_offset）；deform 的 `src_base` 补上 `p.src_offset`（upstream 新增的错位补偿）；
   删掉 Acly 的旧 `fastdiv`（upstream 已有）。
6. `im2col.comp`：CWHN 入口接进 upstream 重构后的 `main()`（ow 循环）；cwhn 里的 `p.pelements`
   改成从 push constants 现算 `OW*KW*KH*IC`（upstream 结构体里没这个字段了）。
7. `supports_op` 放宽：IM2COL 与新拆出的 `GGML_OP_CONV_2D_DEFORM` 分支允许
   `ggml_is_contiguous_channels`（src0/src1/dst 都可能 cwhn）；CONV_2D/TRANSPOSE 保持 upstream 的严格连续要求。
8. `GGML_VK_MAX_NODES_PER_SUBMIT` 是上游环境变量（默认 100），与 fork 里的 `GGML_VK_NODES_PER_SUBMIT` 不同名。
