# Vulkan 批量提交导致输出损坏（BiRefNet 系列模型）— 调查记录与修复

日期：2026-10-13 ~ 10-14
涉及：本仓库 + `depend/llama`（Acly/llama.cpp 子模块）里的 ggml Vulkan 后端
测试机：AMD RX 550 (RADV POLARIS12, 2GB) + Intel UHD 620 (Mesa 26.1.6)，Linux

## 一句话

Vulkan 后端把图里的节点**打包提交**给 GPU 时（默认一次最多 100 个节点），算出来的结果是坏的；
把每批节点数降到 **2** 之后，输出与 CPU 参考**逐像素一致**。根因还没找到，所以先用这个绕过。

## 现象

- 症状：GPU 出图与 CPU 出图几乎不相关（IoU≈0.16），画面上是横向条带状的坏遮罩。
- **CPU 后端一直是对的**，同模型、同图片、同权重。
- 三个模型全部中招，改完全部修好：

| 模型 | 默认（坏） | 修复后（nodes_per_submit=2） | CPU vs Python 参考 |
|---|---|---|---|
| RMBG-2.0-F16 | IoU 0.161 | **0.999954** | 0.992 |
| BiRefNet-toonout-F16 | IoU 0.161 | **0.999972** | 0.957 |
| BiRefNet-F16（Acly 官方 non-lite） | IoU 0.161 | **1.000000** | 0.972 |
| BiRefNet-**lite**（embed96） | **一直是好的** | 不需要改 | — |

（IoU/MAE 都是「GPU 出图 vs 同模型 CPU 出图」；最后一列是「CPU 出图 vs Python 项目出图」，属于模型实现差异，与本 bug 无关。）

耗时（同一张 1024×1024 输入）：默认 11.0s → 修复后 12.4s（+13%），CPU 后端 30~34s。

## 影响范围

- **不是权重问题**：三个来源的 GGUF（自转、Acly 官方、gobeldan）逐张量比对内容一致。
- **不是算子算错**：开 `GGML_VULKAN_CHECK_RESULTS` 编译选项后，1899 个节点逐个和 CPU 对拍全部通过（最大误差 4e-5），最终输出也等于 CPU —— 因为该模式本质上就是"每个节点单独提交"。
- **不是精度/开关问题**：flash attention、fusion、F16、coopmat、MMVQ 等开关的输出 md5 完全不变。
- **不是某一张卡的问题**：AMD RX 550 和 Intel UHD 620 输出完全相同地坏。NVIDIA 等未测。
- 批大小阶梯（三模型趋势一致，BiRefNet non-lite 上测得最细）：

| 每批节点数 | 1 | 2 | 4 | 8 | 16 | 32/64/100 |
|---|---|---|---|---|---|---|
| IoU vs CPU | 1.000 | 1.000 | 1.000 | 0.43 | 0.71 | 0.161 |

- **只需一个边界**：图共 3316 节点，在节点 3310（`ADD decoder.ipt_blk1`）前插一次提交，
  其余保持默认 32/批，输出即完全正确；插在 3311/3312/3313 分别只有 0.84/0.17/0.16。
  说明问题集中在打包提交路径的某一处衔接，而不是普遍性的算错。

## 修复

`depend/llama/ggml/src/ggml-vulkan/ggml-vulkan.cpp`，`ggml_backend_vk_graph_compute` 内：

```cpp
int nodes_per_submit = 2;                     // 原值 100
if (const char * env = getenv("GGML_VK_NODES_PER_SUBMIT")) {
    nodes_per_submit = atoi(env);             // 调试用：可临时改回坏值复现
}
```

已验证：`GGML_VK_NODES_PER_SUBMIT=100` 能稳定复现 IoU 0.161 的坏结果，说明默认值确实在生效。

## 怎么用 / 怎么验证

```bash
git clone https://github.com/danqingximeng/vision.cpp.git && cd vision.cpp
git submodule update --init
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVISP_VULKAN=ON
cmake --build build -j6 --target vision-cli

build/bin/vision-cli birefnet -b gpu -m RMBG-2.0-F16.gguf -i input.png \
    -o mask.png --composite out.png
```

出图不对时的对照：同一张图加 `-b cpu` 跑一遍，两者的 mask 应当几乎一致（IoU > 0.999）。

## ⚠️ 调试时的两个坑

1. **`GGML_VULKAN_CHECK_RESULTS=ON` 会让结果"假装修好"**：它把每个节点强制单独提交并同步，
   等价于把批大小变成 1，所以输出正确 —— 别据此以为问题不存在。
2. **不要把整图塞进一个命令缓冲**（比如把 `nodes_per_submit` 调到 999999）：
   在 RX 550 上跑完后 GPU 会挂死，`vulkaninfo` 永久卡死，**只能重启，杀进程无效**。
   本次调查挂死 3 次全部是这类配置造成的；正常的 1/2/4/8/.../100 批大小跑了几十次都没事。

## 还没查清的（留给后续）

- 病根在"多节点打包提交"这条路径上，但具体是少了一道同步、还是缓冲区被提前复用，没有定论。
  源码里 `overlaps_unsynced` 的依赖检测、descriptor set 的单次使用、命令缓冲的复用逻辑看起来都是对的；
  而提交边界本身既不加 barrier、也不清 `unsynced_nodes_*` 列表 —— 这解释不通，所以还没收敛。
- 可疑点（`ggml-vulkan.cpp`，行号基于 commit `5fde0fae`）：
  - `nodes_per_submit` / 提交判定：约 14400、14605
  - 依赖同步判定：约 12825–12910
  - 提交路径 `submit || last_node`：约 13272；`ggml_vk_compute_forward` 的 memcpy 提交顺序：约 13304
  - `ggml_vk_preallocate_buffers` 先提交后 flush memcpy 的顺序问题：约 12740
- 建议上游（ggml-org/llama.cpp）复现方式：跑一次 BiRefNet 图，观察 `nodes_per_submit` 阶梯，
  以及"只在节点 3310 前提交一次"这个最小复现。

## 这个目录里的文件

```
README.md              本文
session-notes.md       调查全过程的原始笔记（每一轮的命令、数字、踩坑）
patches/
  vk-submit-probe.patch      调查用探针（10 个环境变量：强制某处提交、dump 节点信息等），打在 depend/llama 上
  ggml-vulkan-checker.patch  让 CHECK_RESULTS 支持 CONV_2D_DEFORM + 放大其内存池
scripts/
  gpu_test.sh  round4.sh  round5.sh   自动化测试脚本（每个用例后做 GPU 健康检查）
outputs/
  <model>-cpu.png / <model>-gpu-broken.png / <model>-gpu-fixed.png   三模型对照出图
```
