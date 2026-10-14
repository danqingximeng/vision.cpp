# vision.cpp 抠图研究 · 2026-10-13 第二轮（GPU 挂掉前的记录）

本目录是重启前从 `/tmp/opencode` 抢救出来的资料。`VISION_CPP_RMBG_RESEARCH.md`（项目根）仍是第一轮交接文档，本文是第二轮增量，两者一起看。

## 一、本轮结论（按证据强度）

### 1. 权重不是问题

下载并逐张量对比了三个来源的 GGUF：

| 文件 | 来源 | 与对比对象差异 |
|---|---|---|
| `/archive/AI/models/vision.cpp/RMBG-2.0-F16.gguf` | gobeldan/RMBG-2.0-GGUF（重下后 md5 相同） | 与本地旧文件**字节级一致** |
| 自转 `RMBG-2.0-self-F16.gguf` | `convert.py` 从 `models/RMBG/RMBG-2.0/model.safetensors` | 585 张量 **0 差异** |
| `/archive/AI/models/vision.cpp/BiRefNet-F16-Acly.gguf` | 作者 Acly/BiRefNet-GGUF（**非 lite**） | 与自转 RMBG 全不同（本来就是不同底模） |
| `/archive/AI/models/vision.cpp/BiRefNet-ToonOut-F16-Acly.gguf` | 作者 Acly/BiRefNet-toonout-GGUF | 与自转 ToonOut 仅 17 张量差 ≤0.00049（F16 舍入） |
| 自转 `toonout-self-F16.gguf` | safetensors | 与 `/archive/.../BiRefNet_toonout-F16.gguf` **字节级一致**（md5 4afad945…） |

结论：三个来源（自转、作者官方、第三方）权重等价 → **排除“权重文件有问题”**。

### 2. CPU 出图全部正常（问题模型也正常）

`out/cpu/` 与 Python 基准（`research-assets/.../results/python-current/`）对比：

| 模型 | MAE | IoU@0.5 |
|---|---|---|
| RMBG-2.0（自转 / gobeldan，二者完全一致） | 0.00167 | 0.992 |
| ToonOut（自转 / 作者） | 0.00949 | 0.957 |
| 作者 BiRefNet-F16（general，无 python 同模参考，与 python general 比） | 0.00587 | 0.972 |

CPU 推理约 30–34s/张（load ~1s），无横条。

### 3. Vulkan 出错与权重、驱动、多数开关无关

embed_dim=192（585 张量）模型在 Vulkan 上输出 IoU≈0.16 的垃圾：

- 自转 / gobeldan / 作者官方，F32 / F16 —— 全错
- AMD RX 550 与 Intel UHD 620 —— 全错
- `VISP_FLASH_ATTENTION=0`、`GGML_VK_DISABLE_FUSION=1`、`GGML_VK_DISABLE_F16=1`、`GGML_VK_DISABLE_COOPMAT=1` —— 输出 **md5 完全相同**（即这些开关不改变结果）
- `GGML_VK_DISABLE_ASYNC=1` —— 仍错（IoU 0.27，垃圾形态还变了）
- BiRefNet-**lite**（embed_dim=96、429 张量）：同机 AMD Vulkan **完全正确**（IoU=1.0）

### 4. 决定性实验：`GGML_VULKAN_CHECK_RESULTS`

重新克隆 vision.cpp 并打开该编译选项后：

- **1899 个节点逐一对拍，全部与 CPU 参考一致**（最大 avg_err 4e-5，输出层 4.8e-10），deform conv 也全对
- 同一次运行的输出遮罩 **与 CPU 完全一致（MAE=0、IoU=1.0）**
- 同一模型同一权重用普通二进制跑 → IoU 0.16

→ **根因方向：不是权重、不是算子算错，而是时序/同步类 bug**（checker 每节点强制 `ggml_vk_buffer_read` 同步后结果就正确）。`GGML_VK_DISABLE_ASYNC=1` 不能修复，说明不是简单 async 开关，**更像缓冲复用 / 同步点缺失 / 竞态**。

> 注：CHECK_RESULTS 模式慢（本例 334s vs 普通 11s），且它会改变时序本身，所以“开了就对”既可能是同步效应，也可能是掩盖了竞态。下一步要做的正是二分确认。

### 5. GPU 状态

本轮末尾（17:46 起）`vulkaninfo --summary` 卡死、20s 只能 SIGKILL、无 deviceName 输出 → **GPU 已挂，需重启**。当时最后一个 GPU 操作是 `GGML_VK_DISABLE_ASYNC=1` 的普通推理（正常结束、rc=0），之后还有一次被工具超时杀掉的 python（与 GPU 无关）。未对挂死进程发过信号。

## 二、可复用文件

```
archive/AI/models/vision.cpp/
  BiRefNet-F16-Acly.gguf            作者官方 non-lite（md5 5ffc9a20…）
  BiRefNet-ToonOut-F16-Acly.gguf    作者官方 ToonOut（md5 d76c9a9c…）
  RMBG-2.0-F16.gguf                 gobeldan 版（md5 05dbea2e…，=旧文件）
  RMBG-2.0-F32.gguf                 旧 F32（本轮未用）
  BiRefNet_toonout-F16.gguf         旧自转版（md5 4afad945…，=本轮自转）

research-assets/vision-cpp-rmbg/session-2026-10-13/
  ggml-vulkan-checker.patch   ★ 两处调试改动：checker 补 CONV_2D_DEFORM + 内存池 2GB→6GB
  run_vision.sh               测试驱动脚本（$VISION_CLI 可换二进制，$TMO 超时）
  log/check-general.log       首次 check：缺 CONV_2D_DEFORM 分支而 abort
  log/check2.log              补完 deform 后：1893 节点全对，checker 池 2GB OOM abort
  log/check3.log              6GB 池：1899 节点全对、rc=0、输出=CPU（关键证据）
  out/cpu/                    各权重 CPU 出图（对 python 基准）
  out/vulkan/                 各开关 Vulkan 出图（含 check3 正确样本）
  gaptree/  vk_switch_ops.txt ggml op 清单
  vkinfo.txt                  GPU 挂掉时的 vulkaninfo 残留输出
```

自转 GGUF 没保留（840MB，3 秒可重生）：

```sh
cd /home/danqing/projects/RMBG
PYTHONPATH=/opt/vision.cpp/depend/llama/gguf-py .venv/bin/python \
  /opt/vision.cpp/scripts/convert.py birefnet models/RMBG/RMBG-2.0/model.safetensors \
  --output /tmp/opencode/conv/RMBG-2.0-self-F16.gguf --quantize f16
# ToonOut 同理：models/RMBG/BiRefNet/BiRefNet_toonout.safetensors
```

## 三、复现 check 版构建（重启后照抄）

```sh
cd /tmp/opencode
git clone https://github.com/Acly/vision.cpp.git vision.cpp-src
cd vision.cpp-src && git submodule update --init
patch -p1 -i /home/danqing/projects/RMBG/research-assets/vision-cpp-rmbg/session-2026-10-13/ggml-vulkan-checker.patch   # 打在 depend/llama/ 下
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVISP_VULKAN=ON -DGGML_VULKAN_CHECK_RESULTS=ON -DVISP_TESTS=OFF
cmake --build build -j6 --target vision-cli        # 约 4.5 分钟
VISION_CLI=/tmp/opencode/vision.cpp-src/build/bin/vision-cli TMO=3000 \
  /tmp/opencode/run_vision.sh gpu <model.gguf> <input.png> out/vulkan/xxx
```

（patch 路径注意：改动在子模块 `depend/llama/ggml/src/ggml-vulkan/ggml-vulkan.cpp`，`git -C depend/llama apply` 或在子模块内 `patch -p1`。）

## 四、重启后建议顺序

1. `timeout -s KILL 20 vulkaninfo --summary` 确认 GPU 恢复；恢复前不要跑任何 GPU 命令。
2. 沿 4 号证据做二分定位同步 bug：
   - 用 check 版 + `GGML_VULKAN_SKIP_CHECKS=<n>` 跳过前 n 个节点，找到“从第几个节点开始需要同步”的边界；
   - 读 `ggml-vulkan.cpp` 中 `ggml_vk_build_graph` 的 `!node->buffer` 静默跳过分支、buffer 复用/`is_owner`、`ggml_backend_tensor_copy` 与 fence 路径；
   - 对比 lite（正常）与 general（错）在 graph 分割、fused op 数量上的差异。
3. 每个 GPU 实验后立刻 `timeout -s KILL 20 vulkaninfo` 自检，挂了立即停并通知用户。
4. 不占 GPU 的欠账：九张图 Python vs vision.cpp CPU 完整基准、`rmbg` CLI 冷/热计时、Q8_0/Q4_0（`convert.py` 只支持 f16，需走 `depend/llama/tools/quantize` 或自己写量化，注意 vision.cpp 载入时会把量化权重重转回 float，收益要实测）。

## 五、第三轮增量（2026-10-13 20:26 重启后，跑到 20:40 GPU 再次挂死）

重启后 GPU 正常（vulkaninfo 即时返回）。check 版已重新 clone+patch+构建成功。

**新事实：`GGML_VK_DISABLE_MULTI_ADD=1` 显著改变了错误输出**（`out/vulkan-round2/`、`log/round2/`）：

| 实验（AMD，BiRefNet-F16-Acly，9.png） | IoU vs CPU | 备注 |
|---|---|---|
| baseline（普通二进制） | 0.1612 | 与重启前一致 |
| `GGML_VK_DISABLE_MMVQ=1` | 0.1612 | 无变化 |
| `GGML_VK_FORCE_MMVQ=1` | 0.1612 | 无变化 |
| `GGML_VK_DISABLE_MULTI_ADD=1` | **0.7826** | 错误形态改变，仍不正确；**跑完后 vulkaninfo 卡死 → GPU 挂** |

→ multi-add 融合参与了数值错误，但关掉它不能修好；说明底层还有别的问题（与 check 版"全对"对照，仍指向同步/批处理）。

**代码侧已确认（源码审计）**：`GGML_VULKAN_CHECK_RESULTS` 不是被动 checker，它把每个节点强制 `last_node=true`（`ggml-vulkan.cpp:13266-13270`）、每节点提交一次命令缓冲并等 fence（`13321-13324`），即等价于 `nodes_per_submit=1`。普通模式按 ≤100 节点或 `mul_mat_bytes` 分批提交（`14400-14405`、`14605-14608`）。

**下一轮最决定性实验（按顺序）**：
1. 在普通（无 CHECK_RESULTS）构建里把 `ggml-vulkan.cpp:14400` 的 `nodes_per_submit` 强制为 1 → 若输出变正确，即锁定"批内排序/同步"bug。
2. 对照 `14405` 强制 `mul_mat_bytes_per_submit=0`。
3. 审计候选：`12740-12749` mid-graph `ggml_vk_preallocate_buffers` 先 submit 后 flush memcpy 的顺序问题（源码确认存在，check 模式会掩盖）；`8154-8176` needs_split 大张量路径；`7011-7023` split_k。
4. ⚠️ `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` 依然禁用（曾导致挂死）；`GGML_VK_DISABLE_MULTI_ADD=1` 这次虽正常退出，但之后 GPU 挂了，慎用。
5. 每次 GPU 实验后必须 `timeout -s KILL 15 vulkaninfo` 自检（脚本 `gpu_test.sh` 已内置，挂在了脚本里）。

## 六、第四轮增量（2026-10-13 21:32 重启后，跑到 21:55 GPU 第三次挂死）

### 已锁死根因（决定性实验）

在**普通（无 CHECK_RESULTS）构建**里把 `nodes_per_submit` 改成可用 env 控制（`VISP_NODES_PER_SUBMIT`、`VISP_MUL_MAT_BYTES_PER_SUBMIT`），对同一个模型同一张图测出阶梯（`out/vulkan-round3/`、`log/round3/`）：

| 配置 | IoU vs CPU |
|---|---|
| nodes_per_submit=1 / 2 / 4 | **1.000000**（MAE=2e-6，与 CPU 逐像素一致） |
| nodes_per_submit=8 | 0.401 |
| nodes_per_submit=16 | 0.681 |
| nodes_per_submit=32 / 64 / 100（=默认） | 0.161195（完全损坏） |
| nodes_per_submit=100 + mul_mat_bytes=0 | 0.161195 |
| 全图只提交一次（nodes=999999 + mul=0） | 0.745 |

→ **权重、算子、驱动全部排除；bug 在"多节点批量提交"路径**：批越小越正确，4 节点一批完全正确；但非单调（16 比 8 好、整图一次提交 0.74），说明不是"同步密度"问题，而是**批边界落在某些特定算子之间**时触发。

性能代价：N=1 时 12.5s vs 默认 11.0s（约 +13%），完全可接受——**这就是一个可用的绕过方案**。

### 新增代码探针（已打进 `/tmp/opencode/vision.cpp-src`，源码已随 tmp 丢失，需重打）

1. `VISP_NODES_PER_SUBMIT` / `VISP_MUL_MAT_BYTES_PER_SUBMIT` env 覆盖（在 `nodes_per_submit=100` 与 `mul_mat_bytes_per_submit=...` 两行后各加 env 读取）。
2. `VISP_SUBMIT_BEFORE_OP` / `VISP_SUBMIT_AFTER_OP`：节点 op 名匹配时强制 `submit=true`；首次建图打印 op 直方图（`[vk-hist]`）。

### 该模型的 op 直方图（1899+ 节点，`nosub.log`）

ADD 553、MUL 135、MEAN 5、REPEAT 48、CONCAT 50、NORM 112、MUL_MAT 200、SCALE 20、CPY 144、CONT 211、RESHAPE 842、VIEW 216、PERMUTE 359、GET_ROWS 48、IM2COL 2、CONV_2D 80、**CONV_2D_DEFORM 20**、UPSCALE 17、PAD 48、ROLL 48、FLASH_ATTN_EXT 48、UNARY 110

### GPU 死机模式

三轮共死 2 次（multiadd-off 后、nosub 后），都是"跑完正常退出 rc=0"之后 `vulkaninfo` 卡死 → **某些配置会让 GPU 悬挂**，与是否 KILL 无关。每次实验后必须自检。

### 重启后下一步（脚本：`round4.sh`，跑完自动在首次 GPU 死机时停）

1. baseline 复测（默认）确认 bug 仍在。
2. 用 `VISP_SUBMIT_AFTER_OP=<op>` 逐个算子类型试（`VISP_NODES_PER_SUBMIT=999999`、`VISP_MUL_MAT_BYTES_PER_SUBMIT=0`，只留 op 驱动的提交），找出"在它之前/之后提交就能修好"的那个算子 → 直接定位缺同步的具体算子。
3. 若 2 找到，则去 `ggml-vulkan.cpp` 对应算子的 cmdbuf/barrier 逻辑里找缺失的 barrier，并做最小修复（不用 env 绕过）。
4. 同时把绕过方案写进上游 issue（Acly/vision.cpp 或 ggml-org/llama.cpp）。

## 七、第五/六轮（2026-10-14 上午，GPU 未再挂）

### 算子级探针（round5，全部小批量，未挂 GPU）

`VISP_SUBMIT_BEFORE_OP/AFTER_OP` 在 `n=32`（完全损坏）基础上加边界，IoU：

| 配置 | IoU |
|---|---|
| n=32 对照 | 0.161 |
| n=4 对照 | 1.000 |
| n=32 + **BEFORE CONV_2D** | **1.000** |
| n=32 + BEFORE/after 其它（DEFORM/IM2COL/MUL_MAT/FLASH/MEAN/NORM/REPEAT/GET_ROWS/UPSCALE…） | 0.161–0.206 |
| n=8 + BEFORE CONV_2D | 1.000 |

### 单边界定位（关键：只需**一个**边界）

图共 **3316 节点**（conv 全在 2821–3312，decoder 段）。逐个测：

- `n=32 + VISP_SUBMIT_ALL_FROM=K`：K≤3310 → **IoU=1.000**；K≥3320（超过图长，等于无边界）→ 0.161
- `n=32 + VISP_SUBMIT_AT_IDX=K`（**只在 K 处加一个边界**）：
  - K=**3310** → **IoU=1.000000**
  - K=3311 → 0.836，K=3312 → 0.175，K=3313/3315 → 0.161

→ 整张图**只有节点 3310（`ADD decoder.ipt_blk1`）前缺一个批边界**；其它 3315 个位置的边界可有可无。
尾部序列：3304 UPSCALE → 3305 CONV_2D → 3306 RESHAPE(bias) → 3307 ADD → 3308 CONV_2D → 3309 RESHAPE → **3310 ADD** → 3311 CONCAT → 3312 CONV_2D → 3313 RESHAPE → 3314 ADD → 3315 UNARY(output)。

注意：`nodes_per_submit=4` 时 3310 并不落在边界上却也正确 → "conv 必须是 cmdbuf 首节点"的假设已被否定，机制仍未定论（见下）。

### 两个目标模型验证（本轮最重要的实用结果）

同一二进制、同一张图（9.png），对照 CPU 参考：

| 模型 | 默认 | `VISP_NODES_PER_SUBMIT=4` | `VISP_SUBMIT_BEFORE_OP=CONV_2D` |
|---|---|---|---|
| **RMBG-2.0-F16** | IoU 0.161 | **0.999954** | **0.999954**（与 n=4 md5 相同） |
| **BiRefNet_toonout-F16** | Io IoU 0.161 | **0.999972** | **0.999972**（md5 相同） |
| BiRefNet-F16-Acly(general) | 0.161 | **1.000000** | **1.000000** |

耗时：默认 11.0s，n=4 / before-conv **12.4s**（+13%），CPU 为 30–34s → **Vulkan + 该绕过 ≈ 2.6× CPU**。

> 教训：round6 第一次跑忘了 `export VISION_CLI=`，用成了 `/opt` 原版二进制，得出"两个模型三配置结果完全相同"的假结论；日志里没有 `[vk-debug]`/`[vk-hist]` 行就是没跑探针二进制的铁证。

### 仍未解决（根因机制）

一个批边界为何能修好，尚未定论。已排除/存疑：
- `overlaps_unsynced` 依赖检测、descriptor set 单次使用、cmdbuf 复用（源码审计认为正确）；
- 提交边界本身**不**清除 `unsynced_nodes_*`（13319 只设 `submit_pending`），也不加 barrier → 所以"边界补 barrier"这个解释需要验证 `ggml_vk_submit`/新 context 是否真的带来依赖（`ggml_vk_get_compute_ctx` 新建 context，无 wait semaphore）。
- 待查：`ggml_vk_preallocate_buffers`(12740) 先 submit 后 flush memcpy 的顺序；prealloc scratch 跨节点复用与 `prealloc_*_need_sync`；以及长 cmdbuf 内 push constant / descriptor 池行为。

### 探针补丁

`vk-submit-probe.patch`（本轮已更新并验证可反向 apply）：`VISP_NODES_PER_SUBMIT` / `VISP_MUL_MAT_BYTES_PER_SUBMIT` / `VISP_SUBMIT_BEFORE_OP` / `VISP_SUBMIT_AFTER_OP` / `VISP_SUBMIT_OP_IDX_LO/HI` / `VISP_SUBMIT_ALL_FROM` / `VISP_SUBMIT_ALL_UNTIL` / `VISP_SUBMIT_AT_IDX` / `VISP_DUMP_OP_IDX` / `VISP_DUMP_TAIL`。

## 八、待办

1. **把绕过固化**：给 vision.cpp 打 1 行补丁（`nodes_per_submit=4` 或 CONV_2D 前强制 submit）并构建一个可用的 `vision-cli`（当前 /opt 原版未动）。
2. 九张图全量基准：CPU vs Vulkan+绕过（IoU/耗时）。
3. 继续根因：验证"边界是否真的补了 barrier"、prealloc 顺序问题。
4. 上游 issue：给出最小复现（nodes_per_submit 阶梯 + 单边界 3310 + 两模型 0.161→1.0）。
