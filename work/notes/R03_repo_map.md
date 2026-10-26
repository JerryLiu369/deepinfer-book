---
# R03 仓库地图：目录、代码量、依赖与 vLLM 派生清单

## 0. 元信息
- 代码版本：`deepseek-ai/DeepInfer`，本地 `work/sources/DeepInfer`，branch `main`，HEAD `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21，commit message: `Bump DeepEP to f6aa38fe and drop the SSH->HTTPS rewrite (#157)`）
- 包版本：`0.1.0`（`pyproject.toml:11`），`requires-python = ">=3.12"`（`pyproject.toml:16`），Apache-2.0
- git 历史：只有 4 个 commit（`d585e65 initial commit` 2026-10-20 → `9bd9cf2` 2026-10-21），无 tag；tracked 文件 348 个
- 统计口径：`os.walk` 遍历 + 物理行数（含空行/注释/docstring），不是 SLOC。vendor 内 `3rdparty/`、`third-party/` 已剔除（否则 CUTLASS 副本会把数字放大一个量级，见 §3 注）
- 注释语言：Python 用 `tokenize` 区分 COMMENT / STRING token，逐行判断是否含 CJK（`[一-龥]`）；C++ 用行首 `//` 判定
- 查阅来源：`README.md`、`pyproject.toml`、`setup.py`、`Makefile`、`build.sh`、`scripts/{build,build_vendor,format_csrc,gen_stub}.sh`、`.gitmodules`、`doc/inference_deployment_qa.md`、`deepinfer/__init__.py`、`deepinfer/entrypoints/llm.py`、`deepinfer/engine/llm_engine.py`、`deepinfer/engine/arg_utils.py`、`deepinfer/model_executor/layers/tilelang_ops/ops.py`、`deepinfer/npu_ops/{__init__,patch_imports}.py`、`csrc/**`、`vendor/*/README.md`

## 1. 三个最重要的发现
1. **调度器是 C++，不是 Python。** `Scheduler`/`BlockManager`/`Sequence` 住在 `csrc/engine/`，经 `deepinfer._C.engine`（pybind11）暴露；`LLMEngine` 只做装配、请求入队和 step 驱动。这是与 vLLM（Python 侧调度）最结构性的差异。
2. **代码重心在算子层，不在"引擎"。** `tilelang_ops/` 三个目录合计 23,331 行，占 `deepinfer/` 47,172 行的 49.5%；同一组 kernel 用 `cuda/` 与 `asc/` 两套各 30 个文件镜像维护。
3. **vLLM 血统是"骨架级"的。** 带 `Derived from vLLM` 头的只有 23 个文件、6,817 行，占一阶 Python（55,748 行）的 12.2%，全部落在 engine / worker / sequence / sampling 这些胶水层；`csrc/`、`tilelang_ops/`、`v4_modules/` 一行 vLLM 头都没有。

## 2. 事实条目

### #1 仓库分三层：Python 前端 + C++ 引擎核 + 算子 submodule
- 证据：`deepinfer/engine/llm_engine.py:20-33`
  > from deepinfer._C.engine import (  # pyright: ignore[reportMissingModuleSource]
  >     BlockManager, FlopsEstimator, RequestOutputPool, Scheduler,
  >     SchedulerOutputs, SequenceMetadata, SequenceMetadataPool, SequenceStatus, ...
- 等级：A
- 意味着什么：读这本书时要先建立"Python 编排 / C++ 调度 / 设备 kernel"三层心智模型，否则会在 `LLMEngine.step()` 里找不到真正的调度逻辑。
- 疑点：`FlopsEstimator`、`prepare_*` 等函数在 C++ 侧算，Python 侧只传 numpy 数组，跨语言边界是每 step 一次还是多次，需在 R05 细查。

### #2 `deepinfer/__init__.py` 明令禁止 import torch
- 证据：`deepinfer/__init__.py:1`
  > # 严禁在本文件内进行任何形式的 import torch 或初始化 device 的行为，会导致子进程的故障
- 等级：A
- 意味着什么：包初始化必须保持"零设备副作用"，因为 worker 进程要靠 fork/spawn 后自己设 `CUDA_VISIBLE_DEVICES`；这条约束解释了文件里为什么有一堆 `os.environ` 和延迟 import。
- 疑点：无。

### #3 编译产物按 `<accelerator>-<arch>` 分目录，两平台共存于同一棵树
- 证据：`deepinfer/__init__.py:28-32`
  > # 构建产物按平台分目录存放（deepinfer/_so/<tag>/），加入包搜索路径后，import 仍按 canonical
  > # soabi 名解析到本平台的那一份；同 arch 的 nvidia / ascend 因此可以共存于同一棵代码树。
  > _so_dir = os.path.join(os.path.dirname(__file__), "_so", _so_platform_tag())
  > if os.path.isdir(_so_dir):
  >     __path__.append(_so_dir)
- 等级：A
- 意味着什么：同一份 checkout 可以先后编出 NVIDIA 和 Ascend 两套 `.so`，靠 `__path__` 追加做运行时分流；这也解释了 `scripts/build.sh` 里的 `SO_DST="deepinfer/_so/$PLATFORM_TAG"` 软链逻辑。
- 疑点：无。

### #4 pybind 只暴露三个子模块，`csrc/` 里没有一行 CUDA
- 证据：`csrc/pybind.cpp:6-13`
  > pybind11::module cuda_utils = m.def_submodule("cuda_utils", "DeepInfer cuda utils");
  > pybind11::module engine = m.def_submodule("engine");
  > pybind11::module graph_timer = m.def_submodule("graph_timer", "CUDA graph timer");
- 等级：A
- 意味着什么：`csrc/` 是纯 host 侧 C++（调度、元数据准备、pybind 胶水），设备代码全在 `vendor/` 与 TileLang kernel 里；全书讲"kernel 实现"时不该去 `csrc/` 找。
- 疑点：`csrc/asc_utils.cpp` / `csrc/cuda_utils.cpp` 各 100 余行，是设备 API 的薄封装（显存/流查询），不含计算。

### #5 TileLang 算子按后端镜像，运行时分发只在 `ops.py` 一个文件里
- 证据：`deepinfer/model_executor/layers/tilelang_ops/ops.py:1-5, 15-22`
  > """Kernel dispatcher: interface modules do
  > ``from deepinfer.model_executor.layers.tilelang_ops import ops`` and call
  > ``ops.xxx_tl``; backend selection happens only in this file.
  > _pkg = importlib.import_module(f"{__package__}.{PLATFORM.tilelang_ops_backend}")
  > ... if _name.endswith("_tl"): globals()[_name] = _obj
- 等级：A
- 意味着什么：`xxx_tl` 后缀是"后端 kernel 工厂"的约定；接口层（同名无后缀函数）永远调 `ops.xxx_tl`，因此新增后端只需加一个目录 + 改 `platform.py` 的 `tilelang_ops_backend`（`platform.py:57` = `"cuda"`，`platform.py:88` = `"asc"`）。
- 疑点：符号冲突用 `assert` 而非异常，导入期就会炸；这是有意的严格性还是临时手段，未确认。

### #6 kernel 自带 CPU 参考实现和内联测试，`tests/` 不是唯一测试入口
- 证据：`pyproject.toml:27-30`
  > # Use importlib import mode so that collecting test files inside the
  > # deepinfer package (e.g. deepinfer/model_executor/layers/tilelang_ops/*.py) does NOT
  > # trigger the package __init__.py chain (which imports heavy deps like deep_gemm).
- 等级：A
- 意味着什么：`tilelang_ops/` 93 个文件里有 94 个 `test_*` 函数、46 个 `*_ref` 参考实现、29 个文件直接 `import pytest`；这是"kernel 与它的正确性证据放在一起"的组织方式，也是 pytest 必须用 importlib 模式的原因。
- 疑点：`pytest` 被 kernel 模块顶层 import，意味着生产环境运行这些模块会依赖 dev 依赖；实际部署是否裁剪，未确认。

### #7 模型注册表里只有一个架构
- 证据：`deepinfer/model_executor/models/__init__.py:15-17`
  > _MODELS = {
  >     "DeepseekV41ForCausalLM": ("deepseek_vl", "DeepseekVLForCausalLM"),
  > }
- 等级：A
- 意味着什么：这个开源版本是"单模型引擎"——V4.1-Flash 一套代码路径，没有通用模型适配层；`DeepseekVLForCausalLM` 同时承担纯文本与多模态，说明 VL 是主路径而非插件。
- 疑点：`deepseek.py` 里的 `DeepseekForCausalLM` 未注册，是遗留还是内部使用，未确认。

### #8 vendor 六个 submodule 成对（NVIDIA / Ascend），TileKernels 却是 pip 包
- 证据：`.gitmodules` + `scripts/build_vendor.sh:9-18`
  > nvidia)  PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM)      INSTALL=(deep_select flash_mla deep_ep deep_gemm)
  > ascend)  PACKAGES=(DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend)  INSTALL=(deep_select flash_mla deep_ep deep_gemm)
- 等级：A
- 意味着什么：Ascend 版复用同一组 import 名（`deep_ep`/`deep_gemm`），所以上层代码零改动；但 `README.md:23` 把 TileKernels 与这四个并列，实际它是 `requirements.txt` 里的 `tile-kernels==2.1.0`，不在 `vendor/`。
- 疑点：DeepSelect 两边共用一个仓库（`DEEP_SELECT_BUILD_TARGET_PLATFORM` 切平台），没有 Ascend 分支仓库。

### #9 构建链是三段式：`build.sh` → `build_vendor.sh` → `setup.py`
- 证据：`build.sh:14-25`
  > pip install -r "requirements-${DEEPINFER_ACCELERATOR}.txt"
  > bash ./scripts/build_vendor.sh
  > bash ./scripts/build.sh build
- 等级：A
- 意味着什么：vendor 算子先各自 `python setup.py bdist_wheel` 出 wheel 再 `pip install --no-deps --no-index`，与主扩展解耦；主扩展的源文件清单写死在 `setup.py:22-39`（11 个 .cpp），新增 C++ 文件必须手动登记。
- 疑点：无。

### #10 依赖全部精确 pin，torch 走平台专属 index
- 证据：`requirements-nvidia.txt:2-6` / `requirements-ascend.txt:2-5`
  > --find-links https://download.pytorch.org/whl/cu132/torch/
  > torch==2.13.0+cu132
  > flash-attn-4==4.0.0b34
  > （ascend）torch==2.13.0+cpu / torch-npu==2.13.0rc1
- 等级：A
- 意味着什么：没有 setup.py 里的 `install_requires` 浮动区间，全部 `==`；`torch 2.13` 与 `flash-attn-4 4.0.0b34` 都是很新的版本，读者复现时要按这两个文件装，不能只 `pip install -e .`。
- 疑点：`flash-attn-4` 在 Ascend 侧不装，但 `npu_ops` 会 patch `flash_attn.cute.interface._flash_attn_fwd`，说明 NPU 侧另有 flash-attn 来源，未在 requirements 中体现。

### #11 23 个文件带 `Derived from vLLM` 头，共 6,817 行
- 证据：`grep -rn "Derived from vLLM"`（排除 vendor）→ 23 个命中，全部在第 2 行
  > # SPDX-License-Identifier: Apache-2.0
  > # Derived from vLLM; modified by DeepSeek.
- 等级：A
- 意味着什么：vLLM 的影响集中在 `LLM`/`LLMEngine`/`Worker`/`ModelRunner`/`CacheEngine`/`Sequence`/`SamplingParams` 这条"引擎骨架"，即书的第 2–5 章要讲的主干；清单见 §3 Q5。
- 疑点：另有 2 个文件（`logger.py:4`、`rotary_embedding.py:3-6`）同时带 "Adapted from vLLM / Copyright 2023 The vLLM team"，属于双重署名，但已包含在上述 23 个之内。

### #12 注释语言：Python 22% 中文，C++ 69% 中文，且分布极不均匀
- 证据：统计脚本（`tokenize` 逐 COMMENT token 判定 CJK）→ `deepinfer/` 2,620 条 Python 注释中 578 条含中文（22.1%）；`csrc/` 349 条 `//` 注释中 240 条含中文（68.8%）
- 等级：A（本文自测）
- 意味着什么：越靠近调度/kernel 内部（`csrc/engine/scheduler.hpp` 128 条中文注释、`worker/cache_engine.py` 49 条），中文越密；越靠近 vLLM 派生骨架（`model_executor/models/v4_modules/` 210 条注释 0 条中文、`hugging_face/` 41 条 0 条中文），越是纯英文。这本身是"哪部分自研、哪部分继承"的一个旁证。
- 疑点：`asc/` 与 `cuda/` 差异明显（13% vs 4%），可能是 Ascend 团队与 CUDA 团队写作习惯不同，也可能只是文件抽样偏差。

### #13 README 自陈三块能力不在本版本内
- 证据：`README.md:38-42`
  > Required for a complete inference service (not included in this release):
  > - A KV cache system
  > - VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)
  > - Load balancing across data-parallel replicas for both prefill and decode
- 等级：B（官方 README）
- 意味着什么：书里讲 PD 分离、分布式 KV cache、DP 负载均衡时，必须写成"引擎已留接口/部分实现，但开源版不含"，否则就是把文档能力说成代码能力。
- 疑点：`init_pd_split`（`deepinfer/engine/llm_engine.py` 内有该方法）与 `EngineType` 枚举说明 PD 分离的骨架在代码里，需在 R06 核实到什么程度。

### #14 只有离线批量入口，没有 AsyncLLM / OpenAI 兼容服务
- 证据：`deepinfer/entrypoints/__init__.py` 为空文件；全仓 `grep -rn "AsyncLLM"` 无命中；唯一的 HTTP 服务是 `deepinfer/engine/profile_server.py`（profiling，不是推理）
- 等级：A
- 意味着什么：这个版本的定位是"离线推理 + 基准测试"，不是在线 serving；书里若出现 `LLMEngine` 的在线用法，需要标注为读者自行扩展。
- 疑点：`examples/benchmark.py` 自建了并发/吞吐压测逻辑，可作为"怎么在没有 serving 层时测性能"的参考。

### #15 git 历史被压缩成 4 个 commit，无法做溯源考古
- 证据：`git log --oneline` → `d585e65 initial commit`、`8b16149`、`774ce24`、`9bd9cf2`，全部在 2026-10-20/21
- 等级：A
- 意味着什么：想判断某个模块"从哪个版本开始有"只能靠 `doc/` 与 README 的功能列表，不能靠 git blame；这也解释了为什么 vLLM 派生关系要用文件头声明而不是 commit 记录。
- 疑点：无。

## 3. 必答问题逐条回答

### Q1 每个顶层目录的职责

| 顶层项 | 规模 | 职责 |
| :--- | ---: | :--- |
| `deepinfer/` | 222 文件 / 47,172 行 Python | 引擎本体（Python 侧）：编排、模型、算子、worker、多模态、结构化输出。详见 Q2 |
| `csrc/` | 33 文件 / 2,425 行 C++ + 3,484 行头文件 | C++ 引擎核：`Scheduler`、`BlockManager`、`Sequence`、`RequestOutput`、`FlopsEstimator`、输入元数据准备（`prepare_tensor_input_*`）、`graph_timer`（CUDA graph 内探针）、`cuda_utils`/`asc_utils`。经 pybind11 暴露为 `deepinfer._C` |
| `deep_cluster/` | 8 文件 / 656 行 Python | 硬件探测：`accelerator_kind()`（cpu/nvidia/ascend）、GPU/NPU 型号与显存、NUMA/拓扑绑定（`topology/bind.py`）。被 `setup.py` 与 `worker` 依赖，是唯一被 `find_packages` 一起打包的非 `deepinfer` 包 |
| `vendor/` | 6 个 submodule / 30,043 行 Py + 33,753 行 CUDA（自有部分） | 设备算子库：DeepEP、DeepEP-Ascend、DeepGEMM、DeepGEMM-Ascend、DeepSelect、FlashMLA。见 Q4 |
| `examples/` | 12 文件 / 3,018 行 Python + 103 行 shell | 离线推理样例（`offline_inference.py`）、压测（`benchmark.py`）、Pareto sweep（`sweep/`）、YoCo 等价性验证（`yoco/`）、结构化输出样例、PPL 基线刷新 |
| `eval/` | 12 文件 / 1,694 行 Python | AIME 2026 / HLE / GPQA-Diamond 的离线评测：`run_eval.py`（生成）+ `score.py`（打分，Pass@1）+ 各 benchmark 的 `bench.py`/`run.sh` + `common/judge.py`（LLM judge） |
| `tests/` | 19 文件 / 3,117 行 Python | 端到端与一致性测试：离线推理、batch-invariant、KL、sampler、EPLB、PPL 长上下文、shm broadcast、CUDA 同步护栏。**不含** kernel 单测（那些在 `tilelang_ops/` 内联） |
| `scripts/` | 4 文件 / 194 行 shell | `build.sh`（编主扩展 + 软链 .so）、`build_vendor.sh`（编 submodule wheel）、`format_csrc.sh`（clang-format）、`gen_stub.sh`（pybind11-stubgen 生成 `.pyi`） |
| `doc/` | 1 篇 / 312 行 Markdown + `images/` | 唯一官方文档：`inference_deployment_qa.md`（部署 Q&A，覆盖系统概览、集群存储、EPLB、KV cache、低精度、kernel、投机解码、异步调度、结构化输出） |
| 根文件 | `pyproject.toml`(2.2K)、`setup.py`(91 行)、`Makefile`(983B)、`build.sh`(27 行)、`requirements*.txt`(4 个)、`.clang-format`、`.gitmodules` | 打包与构建入口；`Makefile` 只是 `scripts/*.sh` 的薄封装（`build-nvidia`/`build-ascend`/`vendor-*`/`clean`/`stubgen`/`format`） |

### Q2 `deepinfer/` 子目录与超过 150 行的文件

#### 2.1 子目录职责（22 个子目录 + 顶层，共 222 个 `.py`/`.pyi`，47,172 行）

| 目录 | 文件数 | 行数 | 核心职责 |
| :--- | ---: | ---: | :--- |
| `deepinfer/` | 15 | 2,465 | 顶层设施：配置、环境变量、日志、平台抽象、Sequence 封装、共享内存、计时器、杂项工具 |
| `deepinfer/_C/` | 4 | 819 | 编译扩展 `deepinfer._C` 的 Python 类型桩（pybind11-stubgen 生成） |
| `deepinfer/distributed/` | 2 | 461 | 跨进程共享内存广播（engine ↔ worker） |
| `deepinfer/engine/` | 6 | 1,957 | 引擎编排：`LLMEngine`、参数解析、step 统计、profiling HTTP |
| `deepinfer/entrypoints/` | 2 | 198 | 对外入口：离线批量 `LLM` |
| `deepinfer/hugging_face/` | 8 | 1,375 | HF 资产适配：config 解析、参数名映射、权重加载/校验、tokenizer |
| `deepinfer/model_executor/` | 7 | 611 | 模型执行层根：模型加载、metadata 派生、采样输入容器 |
| `deepinfer/model_executor/eplb/` | 3 | 792 | 专家并行负载均衡（EPLB）：策略、expert 权重 shuffle |
| `deepinfer/model_executor/input_metadata/` | 5 | 1,151 | 每 step 输入元数据构建（含 ZOS / YOCO / DFlash 变体） |
| `deepinfer/model_executor/layers/` | 9 | 1,665 | 通用层与算子封装：linear、RoPE、norm、sampler、attention backend |
| `deepinfer/model_executor/layers/tilelang_ops/` | 33 | 10,945 | TileLang 算子层：接口 + CPU/参考实现 + 内联测试；`ops.py` 按后端分发 |
| `deepinfer/model_executor/layers/tilelang_ops/asc/` | 30 | 7,418 | 上述算子的 Ascend（NPU）后端实现（`*_tl` 工厂） |
| `deepinfer/model_executor/layers/tilelang_ops/cuda/` | 30 | 4,968 | 上述算子的 CUDA 后端实现（`*_tl` 工厂） |
| `deepinfer/model_executor/models/` | 4 | 1,068 | 模型定义：decoder layer、V4.1 主干、VL、MTP/DFlash、视觉塔 |
| `deepinfer/model_executor/models/mtp/` | 2 | 392 | 投机解码（MTP/DFlash）draft 模型 |
| `deepinfer/model_executor/models/v4_modules/` | 24 | 5,197 | V4.1 的模块化组件：attention、MLA、indexer、MoE、mHC、Engram、跨层共享 |
| `deepinfer/model_executor/models/vision_encoder/` | 4 | 394 | ViT 视觉塔与 VL aligner |
| `deepinfer/model_executor/parallel_utils/` | 6 | 706 | 进程组初始化、CP/EP 通信、DeepEP user buffer、NCCL 显存 |
| `deepinfer/npu_ops/` | 4 | 211 | Ascend 适配补丁：import hook 替换 topk / flash-attn 实现 |
| `deepinfer/structure_output/` | 5 | 595 | 结构化输出（xgrammar）：matcher、编译、mask |
| `deepinfer/ve/` | 4 | 566 | 多模态（VE）预处理入口与 VL 输入组装 |
| `deepinfer/ve/processor/` | 5 | 413 | 图像/文本 processor 实现 |
| `deepinfer/worker/` | 10 | 2,805 | 执行侧：Worker/WorkerBase、ModelRunner、CacheEngine、CUDA graph |
| **合计** | **222** | **47,172** | |

几点读法：
- `model_executor/` 一层文件很少（7 个 611 行），重量都在二级目录；`layers/tilelang_ops/` 一个目录就占 49.5%。
- `cuda/` 与 `asc/` 文件数相同（各 30），但 Ascend 侧多 2,450 行（7,418 vs 4,968），说明 NPU 需要更多手工 tile/对齐处理。
- `model_executor/models/mtp/` 只有 `dflash.py` 一个实现（392 行），没有独立的 `__init__` 导出。

#### 2.2 超过 150 行的文件（102 个，按目录分组，行数降序）

**顶层 `deepinfer/*.py`** — 8 个文件，2,071 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `shared_memory.py` | 363 | 多模态数据共享内存池（`SharedTorchMemoryPool`、`BlockHandle`） |
| `timer.py` | 295 | CPU / CUDA / CUDA-graph 三种计时器与 `cpu_timer.record` 装饰器 |
| `utils.py` | 282 | `Counter`、`get_ip`/`get_open_port`、`CudaIPCWrapper`、`forbid_cuda_sync` 同步护栏 |
| `sequence.py` | 281 | Python 侧 Sequence 封装：`MTPPredictInfo`/`SampledInfo`/`SamplerOutput`，桥接 C++ Sequence |
| `config.py` | 228 | `ModelConfig`/`CacheConfig`/`ParallelConfig`/`SchedulerConfig` 四个配置 dataclass 与派生上限（`_get_runtime_max_model_len`） |
| `platform.py` | 227 | `PlatformGPU`/`PlatformNPU` 抽象，决定 `tilelang_ops_backend`（cuda/asc）与通信后端 |
| `environ.py` | 221 | 环境变量注册表：`EnvVar` + `envs`，统一解析 bool/int/float/str |
| `logger.py` | 174 | 日志格式、崩溃头 `print_crash_header`、子进程崩溃时打印父进程栈 |

**`_C/`** — 1 个文件，735 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `_C/engine.pyi` | 735 | pybind 扩展 `deepinfer._C.engine` 的自动生成类型桩（22 个类 + 12 个函数） |

**`distributed/`** — 1 个文件，461 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `distributed/shm_broadcast.py` | 461 | 共享内存环形缓冲 + `MessageQueue`，engine→worker 跨进程广播 tensor dict |

**`engine/`** — 3 个文件，1,826 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `engine/llm_engine.py` | 1029 | `LLMEngine`：装配 config/scheduler/worker，`add_request`/`step`/`post_process`；driver 在本进程、其余 CP rank 为子进程 |
| `engine/stats_logger.py` | 477 | 每 step 的 `StepStats` 采集、滑动窗口聚合与落盘 |
| `engine/profile_server.py` | 320 | 每个 engine 一个 FastAPI profiling 端点 `POST /profile`（torch profiler 采样落盘） |

**`entrypoints/`** — 1 个文件，198 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `entrypoints/llm.py` | 198 | 离线批量入口 `LLM` 类：同步 `generate()`，无 streaming |

**`hugging_face/`** — 3 个文件，948 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `hugging_face/weights.py` | 436 | safetensors 权重覆盖度检查、expert/engram 表扫描、ue8m0 反量化 |
| `hugging_face/config.py` | 328 | 解析 HF `config.json` 为 V4.1 内部结构（attn 模块表、RoPE-YaRN、win/compress/dflash spec） |
| `hugging_face/names.py` | 184 | HF 参数名 ↔ 内部模块名双向映射 |

**`model_executor/`** — 2 个文件，341 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/model_metadata_utils.py` | 188 | `ModelMetadata` 派生：compress ratio、每通道 KV cache 元素数 |
| `model_executor/model_loader.py` | 153 | 架构选择、meta device 建模、HF checkpoint 加载、MoE 冗余配置 |

**`model_executor/eplb/`** — 2 个文件，652 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/eplb/__init__.py` | 434 | `ExpertLoadBalancer`：物理/逻辑 expert 映射、冗余 expert、层状态 |
| `model_executor/eplb/shuffle.py` | 218 | expert 权重的 intra/inter-node shuffle 计划与原地交换 |

**`model_executor/input_metadata/`** — 3 个文件，1,054 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/input_metadata/input_metadata.py` | 490 | 每 step 的 `InputMetadata` 及 V4/VL 变体 |
| `model_executor/input_metadata/prepare_tensor_input_zos.py` | 382 | ZOS 异步调度下的 decode 输入准备（缓冲复用、`finalize_decode_outputs_zos`） |
| `model_executor/input_metadata/yoco.py` | 182 | YoCo prefill 截断的 metadata mixin |

**`model_executor/layers/`** — 4 个文件，1,348 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/layers/sampler.py` | 683 | Python 侧 `Sampler`：logits 处理、prompt logprobs、调用 tilelang 采样 kernel |
| `model_executor/layers/rotary_embedding.py` | 284 | RoPE 与 YaRN 缩放（vLLM 派生） |
| `model_executor/layers/sampler_utils.py` | 196 | `get_logits`/`process_sampling`/prompt logprobs 分块计算 |
| `model_executor/layers/linear.py` | 185 | `ReplicatedLinear`、FP8 cast 粒度与参数 |

**`model_executor/layers/tilelang_ops/`**（接口 + 参考实现） — 22 个文件，10,040 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/layers/tilelang_ops/fused_transform.py` | 1330 | norm+RoPE+Hadamard+cast+写 KV cache 的单趟融合变换 |
| `model_executor/layers/tilelang_ops/sampler.py` | 1080 | 投机验证 metadata、采样参数展开、draft top-k 重映射、accept 结果 |
| `model_executor/layers/tilelang_ops/fused_sample.py` | 832 | 融合采样 kernel（top-p/top-k/拒绝采样）与参考实现 |
| `model_executor/layers/tilelang_ops/compress.py` | 663 | KV 压缩与 state 更新（prefill/decode 两路） |
| `model_executor/layers/tilelang_ops/image_embed_ops.py` | 630 | 把图像 embedding 写入 input embeds（patch 化、下采样） |
| `model_executor/layers/tilelang_ops/norm.py` | 587 | RMSNorm 与 FP8 量化融合（含 scale 分配） |
| `model_executor/layers/tilelang_ops/prepare_v4_decode_metadata.py` | 545 | V4 decode metadata 与 compact block table 构建 |
| `model_executor/layers/tilelang_ops/get_kv_index.py` | 518 | CA / SWA 的 KV index 生成（prefill 与 decode） |
| `model_executor/layers/tilelang_ops/cache_common.py` | 461 | KV cache 格式（`V41_FORMAT`/`V41_FP4_FORMAT`）、tensor 视图与分配 |
| `model_executor/layers/tilelang_ops/prepare_cp_mirror.py` | 394 | CP mirror 元数据准备 + YoCo 边界融合 |
| `model_executor/layers/tilelang_ops/load_paged_kv_cache.py` | 382 | 按 block load plan 加载 paged KV cache |
| `model_executor/layers/tilelang_ops/store_kv_cache.py` | 380 | KV 写入 paged cache（含 slot scatter） |
| `model_executor/layers/tilelang_ops/random_sampling.py` | 332 | 基于 hash 的确定性随机采样（可复现） |
| `model_executor/layers/tilelang_ops/candidate_blocks.py` | 328 | 稀疏注意力候选块的 top-k 压缩、索引还原及参考实现 |
| `model_executor/layers/tilelang_ops/gather_engram_tokens_zos.py` | 248 | ZOS 下按索引 gather Engram token |
| `model_executor/layers/tilelang_ops/store_mtp_info.py` | 226 | MTP 预测信息写入池 |
| `model_executor/layers/tilelang_ops/dflash.py` | 224 | DFlash confidence head 融合 + dflash 输入落库 |
| `model_executor/layers/tilelang_ops/state_cache.py` | 220 | state cache（压缩状态）切片选择与加载 |
| `model_executor/layers/tilelang_ops/gather_input_ids_zos.py` | 193 | ZOS 下 gather 下一步 input_ids |
| `model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` | 166 | 按 confidence rate 计算每请求验证长度 `seqlens_q` |
| `model_executor/layers/tilelang_ops/dflash_copy.py` | 151 | DFlash decode 输入的拷贝/打包 |
| `model_executor/layers/tilelang_ops/structure_output.py` | 150 | 按 step 原地应用 token bitmask（结构化输出） |

**`model_executor/layers/tilelang_ops/asc/`**（Ascend 后端） — 15 个文件，6,122 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/layers/tilelang_ops/asc/fused_transform.py` | 899 | `fused_transform` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/fused_sample.py` | 673 | `fused_sample` 的 Ascend 实现（分块归约顺序对齐参考实现） |
| `model_executor/layers/tilelang_ops/asc/sampler.py` | 663 | `sampler` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/prepare_cp_mirror.py` | 518 | `prepare_cp_mirror` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/norm.py` | 431 | `norm` 的 Ascend 实现（bf16 打包、分行 tile） |
| `model_executor/layers/tilelang_ops/asc/compress.py` | 407 | `compress` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/image_embed_ops.py` | 402 | `image_embed_ops` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/candidate_blocks.py` | 350 | `candidate_blocks` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py` | 345 | `load_paged_kv_cache` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/store_kv_cache.py` | 310 | `store_kv_cache` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/get_kv_index.py` | 263 | `get_kv_index` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/random_sampling.py` | 258 | `random_sampling` 的 Ascend 实现（inclusive scan 版） |
| `model_executor/layers/tilelang_ops/asc/prepare_v4_decode_metadata.py` | 257 | `prepare_v4_decode_metadata` 的 Ascend 实现 |
| `model_executor/layers/tilelang_ops/asc/kv_quant.py` | 196 | KV 量化/反量化查表（倒数表、scale 构造） |
| `model_executor/layers/tilelang_ops/asc/dflash.py` | 150 | `dflash` 的 Ascend 实现 |

**`model_executor/layers/tilelang_ops/cuda/`**（CUDA 后端） — 12 个文件，3,799 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/layers/tilelang_ops/cuda/fused_transform.py` | 619 | `fused_transform` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/fused_sample.py` | 524 | `fused_sample` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/sampler.py` | 411 | `sampler` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/norm.py` | 326 | `norm` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/prepare_cp_mirror.py` | 313 | `prepare_cp_mirror` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/compress.py` | 292 | `compress` 的 CUDA 实现（按 warp 粒度切块） |
| `model_executor/layers/tilelang_ops/cuda/image_embed_ops.py` | 275 | `image_embed_ops` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/random_sampling.py` | 258 | `random_sampling` 的 CUDA 实现（两级布局 + 定点累积） |
| `model_executor/layers/tilelang_ops/cuda/candidate_blocks.py` | 225 | `candidate_blocks` 的 CUDA 实现 |
| `model_executor/layers/tilelang_ops/cuda/prepare_decode_metadata.py` | 196 | `prepare_v4_decode_metadata` 的 CUDA 实现（文件名少了 v4 前缀） |
| `model_executor/layers/tilelang_ops/cuda/store_kv_cache.py` | 195 | `store_kv_cache` 的 CUDA 实现（amax→ue8m0 scale） |
| `model_executor/layers/tilelang_ops/cuda/load_paged_kv_cache.py` | 165 | `load_paged_kv_cache` 的 CUDA 实现 |

**`model_executor/models/（仅本级文件）`** — 3 个文件，1,031 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/models/deepseek.py` | 525 | `DeepseekModel`/`DeepseekForCausalLM`：V4.1 主干组装与 forward |
| `model_executor/models/deepseek_vl.py` | 267 | `DeepseekVLForCausalLM`：注册表唯一 arch，接视觉塔与多模态 token |
| `model_executor/models/decoder_layer.py` | 239 | `DeepseekDecoderLayer` 与 `DFlashDecoderLayer` 两种层结构 |

**`model_executor/models/mtp/`** — 1 个文件，392 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/models/mtp/dflash.py` | 392 | `DFlash` + `ConfidenceHead`：投机解码 draft 模型 |

**`model_executor/models/v4_modules/`** — 8 个文件，3,664 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/models/v4_modules/attn.py` | 1008 | `V4Attention`/`DFlashAttention`：MLA 注意力 + CP gather 取值 |
| `model_executor/models/v4_modules/mlp.py` | 616 | `DeepseekMoE`：grouped GEMM、Mega MoE 权重转换、swiglu 融合 |
| `model_executor/models/v4_modules/index_attn.py` | 575 | DSA lightning indexer 的 MQA 打分 |
| `model_executor/models/v4_modules/hc.py` | 397 | mHC 残差主干 mixing（`HCLayerMixin`/`HCHead`） |
| `model_executor/models/v4_modules/cross_layer.py` | 383 | 跨层共享（share-KV/index/index-K）拓扑与每 forward 状态 |
| `model_executor/models/v4_modules/mla_attn.py` | 299 | MLA 的 LoRA 压缩、MQA indexer、runtime attn sink |
| `model_executor/models/v4_modules/engram.py` | 219 | Engram：n-gram 哈希、跨 rank 分片表 gather、融合投影与门控 |
| `model_executor/models/v4_modules/compress.py` | 167 | `KProject`/`Compressor`：KV 压缩的模型侧参数与调用 |

**`model_executor/models/vision_encoder/`** — 1 个文件，230 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/models/vision_encoder/vit.py` | 230 | ViT 视觉塔（patch embed、RoPE、attention、SwiGLU block） |

**`model_executor/parallel_utils/`** — 2 个文件，470 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `model_executor/parallel_utils/userbuffer.py` | 240 | DeepEP buffer 管理（`EPBuffer`/`EngramBuffer`）、SM 数限制与生命周期 |
| `model_executor/parallel_utils/parallel_state.py` | 230 | `PG` 初始化与 DP/CP/EP/Engram 进程组切分 |

**`structure_output/`** — 1 个文件，347 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `structure_output/matcher.py` | 347 | xgrammar matcher 封装、vocab mask 分配、trigger 边界校验 |

**`ve/（仅本级文件）`** — 2 个文件，489 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `ve/multimodal_processor.py` | 326 | 多模态预处理总入口：图像/文本分流、token 展开 |
| `ve/prepare_tensor_input_vl.py` | 163 | VL 输入的 chunk 切分与 metadata 组装 |

**`ve/processor/`** — 1 个文件，318 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `ve/processor/image_processor.py` | 318 | 图像解码、缩放、patch 打包与 token 长度计算 |

**`worker/`** — 6 个文件，2,596 行

| 文件 | 行数 | 核心职责 |
| :--- | ---: | :--- |
| `worker/cache_engine.py` | 978 | `CacheEngine`：KV cache 布局/字节估算/分配（含 SWA、state cache、FP4） |
| `worker/model_runner.py` | 644 | `ModelRunner`：单 step forward 编排（input metadata → model → sampler） |
| `worker/base.py` | 357 | `WorkerBase`：分布式环境初始化、NUMA 绑定、模型加载、KV cache 建立 |
| `worker/worker.py` | 283 | `Worker`：单卡 step 循环、engine 回调、seqlens 修正 |
| `worker/cuda_graph_runner.py` | 168 | CUDA graph 捕获与 batch size 分档 |
| `worker/decode_perf_cache.py` | 166 | decode 端按 workload 缓存 step 耗时估计 |

### Q3 代码规模（物理行数，按目录 × 语言）

| 目录 | Python | C++(.cpp) | CUDA(.cu/.cuh) | 头文件(.h/.hpp) |
| :--- | ---: | ---: | ---: | ---: |
| `deepinfer/` | 47,172 | 0 | 0 | 0 |
| `csrc/` | 0 | 2,425 | 0 | 3,484 |
| `deep_cluster/` | 656 | 0 | 0 | 0 |
| `tests/` | 3,117 | 0 | 0 | 0 |
| `examples/` | 3,018 | 0 | 0 | 0 |
| `eval/` | 1,694 | 0 | 0 | 0 |
| 根（`setup.py`） | 91 | 0 | 0 | 0 |
| **一阶小计** | **55,748**（274 文件） | **2,425**（15 文件） | **0** | **3,484**（18 文件） |
| `vendor/DeepEP` | 5,942 | 408 | 7,388 | 4,172 |
| `vendor/DeepEP-Ascend` | 4,718 | 90 | 0 | 6,335 |
| `vendor/DeepGEMM` | 7,340 | 49 | 16,327 | 10,111 |
| `vendor/DeepGEMM-Ascend` | 6,120 | 93 | 0 | 13,696 |
| `vendor/DeepSelect` | 1,896 | 257 | 3,500 | 479 |
| `vendor/FlashMLA` | 4,027 | 1,307 | 6,538 | 14,360 |
| **vendor 小计** | **30,043** | **2,204** | **33,753** | **49,153** |
| **合计** | **85,791** | **4,629** | **33,753** | **52,637** |

非 Python 语言还有：Shell 约 410 行（`scripts/` 194 + 根 `build.sh` 27 + `examples/*/*.sh` 103 + `eval/*/run.sh` 86）；Markdown 约 814 行（README 107 + `doc/` 312 + `examples/README.md` 124 + `eval/README.md` 221 + CONTRIBUTING 50）。

三条口径警告：
1. **一阶代码里没有 CUDA。** 0 个 `.cu` 文件（`find . -name "*.cu" -not -path "./vendor/*"` 为空）。设备代码要么在 `vendor/`，要么是 TileLang 生成的 kernel。书里若出现"`csrc/xxx.cu`"一定是错的。
2. **`vendor/*/3rdparty/` 与 `third-party/` 已剔除。** 若把它们（主要是 CUTLASS 与 CuTe 副本）算进来，全仓物理行数会变成 Python 655,437 / C++ 53,338 / CUDA 1,408,047 / 头文件 2,563,484（约 4.68M 行），完全不代表 DeepSeek 的代码量。用 `grep -r` 或 IDE 统计时必须排除，否则会得出"仓库有 140 万行 CUDA"的错误结论。
3. **Python 与 C++ 的比例约 23:1，但两者职责不对等。** C++ 只有 5,909 行，却握着 `Scheduler`（`scheduler.hpp` 单文件 1,329 行，是全仓最大的手写 C++ 文件）；Python 的 47K 行里近一半是 kernel 包装与参考实现。

### Q4 依赖分析

#### 4.1 外部 Python 依赖（全部 `==` 精确 pin）

| 依赖 | 版本 | 用途（从 import 位置反推） |
| :--- | :--- | :--- |
| `torch` | 2.13.0+cu132 / 2.13.0+cpu | 张量、分布式、CUDA graph、`torch.utils.cpp_extension` 编译 |
| `torch-npu` | 2.13.0rc1（仅 Ascend） | NPU 后端与 `NpuExtension` |
| `flash-attn-4` | 4.0.0b34（仅 NVIDIA） | `flash_attn.cute` 接口，被 `npu_ops` 在 Ascend 上 patch 掉 |
| `tilelang` | 0.1.15 | `tilelang_ops/**` 的 kernel DSL（`import tilelang` 出现在每个 `*_tl` 工厂） |
| `tile-kernels` | 2.1.0 | 更上层的融合算子库：`quant`/`cast`/`transform.apply_rotary`/`moe.swiglu_forward` 等，49 个文件 import |
| `xgrammar` | 0.2.2 | `structure_output/`（regex / JSON schema / EBNF 约束解码） |
| `transformers` | 4.50.0 | HF config / tokenizer 结构 |
| `tokenizers` | 0.21.4 | 分词实现 |
| `huggingface_hub` | 0.36.2 | 模型下载与缓存（`snapshot_download`） |
| `safetensors` | 0.8.0 | 权重加载 |
| `numpy` 2.3.5 / `scipy` 1.17.0 / `sympy` 1.14.0 / `einops` 0.8.2 | — | 数组、统计、符号形状推导、张量重排 |
| `pybind11` 3.0.1 / `ninja` 1.13.0 | — | C++ 扩展编译 |
| `posix_ipc` 1.3.2 / `pyzmq` 27.1.0 | — | `distributed/shm_broadcast.py` 的共享内存与消息队列 |
| `orjson` 3.11.7 / `tabulate` 0.9.0 / `tqdm` 4.67.3 / `psutil` 7.2.2 / `requests` 2.32.5 / `pillow` 11.3.0 | — | 序列化、统计表、进度条、进程/资源、HTTP、图像解码 |
| dev: `pytest` 8.4.2 / `pytest-xdist` 3.8.0 / `ruff` 0.15.0 / `clang-format` 22.1.3 / `pybind11-stubgen` 2.5.5 | — | 测试、lint/格式化、`.pyi` 生成 |
| dev: `fastapi` 0.128.0 / `uvicorn` 0.40.0 | — | `engine/profile_server.py`（注意：在 `requirements-dev.txt` 里，不是运行时依赖） |
| dev: `matplotlib` 3.11.2 / `tree-sitter` 0.26.0 / `tree-sitter-cpp` 0.23.4 | — | `examples/sweep/plot_pareto.py`、vendor `.pyi` 生成 |

#### 4.2 `vendor/` 六个 submodule

| submodule | 上游 | pin（`git submodule status`） | 用途 | 谁 import |
| :--- | :--- | :--- | :--- | :--- |
| `DeepEP` | `github.com/deepseek-ai/DeepEP` | `f6aa38fe`（`v1.2.1-43-gf6aa38f`） | MoE 专家并行 all-to-all（dispatch/combine，含 FP8）、NVLink 冗余 expert 交换、PP/CP/DP 与 Engram RMA 原语；kernel 运行时经 DeepJIT 编译 | `deep_ep`：`model_executor/parallel_utils/{userbuffer,parallel_state}.py`、`models/v4_modules/{mlp,engram}.py` |
| `DeepEP-Ascend` | `deepseek-ai/DeepEP-Ascend` | `3b25377`（main） | 同上，Ascend 移植；HCCL/HCOMM + UBMEM + URMA；公开 buffer API 与 NVIDIA 版对齐 | 同上（同一个 `deep_ep` 包名） |
| `DeepGEMM` | `deepseek-ai/DeepGEMM` | `1fc9c66`（`v2.1.1.post3-25`） | FP8/FP4/BF16 GEMM、Mega MoE（通信重叠）、lightning indexer 的 MQA 打分、HyperConnection(HC) | `deep_gemm`：`v4_modules/{mlp,attn,hc,index_attn}.py`、`vision_encoder/vit.py` |
| `DeepGEMM-Ascend` | `deepseek-ai/DeepGEMM-Ascend` | `3a35326`（main） | 同上，Ascend 移植，API 兼容（BF16/FP8/FP4/MQA logits/MegaMoE） | 同上 |
| `DeepSelect` | `deepseek-ai/DeepSelect` | `bfa4507`（main） | DSA（稀疏注意力）与 sampler 用的高性能 TopK，比 `torch.topk` 快 2–20×；同时支持 CUDA 与 Ascend | `deep_select.interface.topk`：`npu_ops/deep_select.py`（经 `patch_imports` 在 Ascend 上替换实现） |
| `FlashMLA` | `deepseek-ai/FlashMLA` | `2e5429f` | MLA attention（dense / sparse）、`fused_norm_rope_attn_rope_cast`；2026-09-30 起移除 Hopper 与 V3/V3.2/V4.0 支持并改了 FP8/FP4 KV cache 格式 | `flash_mla`：`model_executor/layers/attention_backend.py` |

补充：`vendor/DeepSelect` 与 `vendor/FlashMLA` 各自带一份 CUTLASS 副本（`csrc/3rdparty/cutlass`），这是 Q3 口径警告 2 的来源。

#### 4.3 构建期依赖与工具链
- GCC ≥ 13、CUDA 13（NVIDIA）或 CANN 9.2（Ascend）、Python 3.12（`README.md:47-53`）
- `setup.py:12` 用 `deep_cluster.hardware.accelerator_kind()` 决定编 `NpuExtension` 还是 `CUDAExtension`，源文件清单在 `setup.py:22-39` 显式列出（11 个 `.cpp`）
- `build.sh:16-19` 在 NVIDIA 侧强制把 NCCL 覆写成 `nvidia-nccl-cu13==2.32.3`

### Q5 带 `Derived from vLLM` 头的文件完整清单（23 个，6,817 行）

命中方式：`grep -rn "Derived from vLLM"`（全仓，排除 `vendor/`），全部位于文件第 2 行，固定形式为 `# Derived from vLLM; modified by DeepSeek.`（第 1 行是 `# SPDX-License-Identifier: Apache-2.0`）。

| # | 文件 | 行数 |
| ---: | :--- | ---: |
| 1 | `deepinfer/config.py` | 228 |
| 2 | `deepinfer/distributed/shm_broadcast.py` | 461 |
| 3 | `deepinfer/engine/arg_utils.py` | 114 |
| 4 | `deepinfer/engine/llm_engine.py` | 1029 |
| 5 | `deepinfer/entrypoints/llm.py` | 198 |
| 6 | `deepinfer/logger.py` | 174 |
| 7 | `deepinfer/model_executor/layers/layernorm.py` | 63 |
| 8 | `deepinfer/model_executor/layers/rotary_embedding.py` | 284 |
| 9 | `deepinfer/model_executor/layers/sampler.py` | 683 |
| 10 | `deepinfer/model_executor/layers/vocab_parallel_embedding.py` | 30 |
| 11 | `deepinfer/model_executor/model_loader.py` | 153 |
| 12 | `deepinfer/model_executor/models/__init__.py` | 37 |
| 13 | `deepinfer/model_executor/parallel_utils/communication_op.py` | 82 |
| 14 | `deepinfer/model_executor/utils.py` | 101 |
| 15 | `deepinfer/sampling_params.py` | 119 |
| 16 | `deepinfer/sequence.py` | 281 |
| 17 | `deepinfer/utils.py` | 282 |
| 18 | `deepinfer/worker/base.py` | 357 |
| 19 | `deepinfer/worker/cache_engine.py` | 978 |
| 20 | `deepinfer/worker/model_runner.py` | 644 |
| 21 | `deepinfer/worker/worker.py` | 283 |
| 22 | `setup.py` | 91 |
| 23 | `tests/distributed/test_shm_broadcast.py` | 145 |

合计 6,817 行，占一阶 Python（55,748 行）的 **12.2%**，占 `deepinfer/`（47,172 行）的 14.4%。

分布特征：
- 全部集中在引擎骨架：`engine/`(2 个)、`worker/`(4 个)、`model_executor/layers/`(4 个)、顶层基础设施（`config`/`logger`/`utils`/`sequence`/`sampling_params`）。
- `csrc/`、`model_executor/layers/tilelang_ops/`、`model_executor/models/v4_modules/`、`ve/`、`structure_output/`、`hugging_face/`、`deep_cluster/` **零命中**——这些是 DeepSeek 自研部分。
- `tests/distributed/test_shm_broadcast.py` 也带 vLLM 头（测试是从 vLLM 用例改的），这是唯一一个非 `deepinfer/` 的命中。
- 另有 2 个文件带双重署名：`deepinfer/logger.py:4` 与 `deepinfer/model_executor/layers/rotary_embedding.py:3-6` 同时写 `# Adapted from ... Copyright 2023 The vLLM team.`（后者还带 `Copyright 2022 EleutherAI and the HuggingFace Inc. team`）。
- `README.md:95` 的官方表述与此一致："originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM"。但代码里 `grep -rn "SGLang\|TensorRT"` 零命中，SGLang/TensorRT-LLM 的影响没有落到文件级署名。

### Q6 注释语言分布

**Python（`deepinfer/`，2,620 条 `#` 注释）：**

| 子目录 | 注释数 | 含中文 | 中文占比 |
| :--- | ---: | ---: | ---: |
| `model_executor/layers/tilelang_ops/asc` | 530 | 69 | 13% |
| `model_executor/layers/tilelang_ops`（顶层） | 470 | 55 | 12% |
| `model_executor/layers/tilelang_ops/cuda` | 455 | 20 | 4% |
| `model_executor/models/v4_modules` | 210 | 0 | 0% |
| `worker/` | 164 | 80 | 49% |
| `engine/` | 134 | 67 | 50% |
| 顶层（`deepinfer/*.py`） | 129 | 42 | 33% |
| `model_executor/layers/` | 81 | 12 | 15% |
| `model_executor/models/` | 73 | 59 | 81% |
| `distributed/` | 67 | 0 | 0% |
| `model_executor/eplb/` | 55 | 37 | 67% |
| `ve/` | 46 | 42 | 91% |
| `model_executor/input_metadata/` | 44 | 26 | 59% |
| `hugging_face/` | 41 | 0 | 0% |
| `ve/processor/` | 31 | 27 | 87% |
| `model_executor/parallel_utils/` | 23 | 1 | 4% |
| `models/vision_encoder/` | 19 | 17 | 89% |
| `model_executor/`（本级） | 13 | 6 | 46% |
| `entrypoints/` | 11 | 2 | 18% |
| `models/mtp/` | 9 | 6 | 67% |
| `npu_ops/` | 8 | 4 | 50% |
| `structure_output/` | 7 | 6 | 86% |
| **`deepinfer/` 合计** | **2,620** | **578** | **22.1%** |

**其它语言/目录：**
- `csrc/`：349 条 `//` 行注释，240 条含中文（**68.8%**）；其中 `csrc/engine/scheduler.hpp` 一个文件 128 条中文注释，`mtp_varlen.hpp` 34 条，`prepare_tensor_input_v4.cpp` 60 条。
- `tests/`：201 条注释，0 条中文（纯英文）。
- `examples/` + `eval/` + `deep_cluster/`：277 条，132 条中文（47.7%）。
- `vendor/`：1,584 条注释，0 条中文（上游开源项目，纯英文）。
- **一阶 Python 合计**：3,098 条注释，710 条含中文（**22.9%**）。

**分布规律（可用于书里的一句话结论）：** 中文注释集中在"DeepSeek 自己写、且逻辑容易踩坑"的地方——调度器（`scheduler.hpp`）、KV cache（`cache_engine.py` 49/83）、多模态预处理（`ve/` 91%、`image_processor.py` 24/28）、EPLB 策略（67%）、结构化输出（86%）；英文注释集中在"从 vLLM 继承、改动不大"的地方——`hugging_face/`(0%)、`distributed/`(0%)、`v4_modules/`(0%)、`tilelang_ops/cuda`(4%)。**中文密度可当作"自研程度"的粗略代理指标**（等级 E，旁证性质）。

### Q7 公开 API

#### 7.1 `deepinfer/__init__.py` 导出什么

| 名称 | 证据 | 说明 |
| :--- | :--- | :--- |
| `AttrDict` | `deepinfer/__init__.py:5`（`from deepinfer.attrdict import AttrDict as AttrDict`） | 支持属性访问的 dict，HF config 的通用容器 |
| `envs` | `deepinfer/__init__.py:6`（`from deepinfer.environ import envs as envs`） | 环境变量注册表（`DEEPINFER_*`） |
| `__version__ = "0.1.0"` | `deepinfer/__init__.py:34` | 与 `pyproject.toml:11` 一致 |
| `PLATFORM` | `deepinfer/__init__.py:53-64`（模块级 `__getattr__` 惰性构造） | 平台单例，首次访问才 import `deepinfer.platform`，避免 torch 副作用 |
| `deepinfer._so/<tag>` | `deepinfer/__init__.py:28-32` | 把平台专属 `.so` 目录追加进 `__path__` |

**关键结论：`LLM` 不在 `deepinfer/__init__.py` 里，也没有 `__all__` 声明对外 API。** 用户必须写 `from deepinfer.entrypoints.llm import LLM`（`examples/_config.py:110`、`examples/benchmark.py:20`、`examples/_structured_output.py:8` 都是这么写的）。`deepinfer/entrypoints/__init__.py` 是 0 字节空文件。

#### 7.2 `LLM` 类的接口形态

```python
# deepinfer/entrypoints/llm.py:23, 54-66
class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        trust_remote_code: bool = False,
        expert_parallel_size: int = 1,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        decode_sequence: bool = True,
        **kwargs: Any,
    ) -> None: ...
```

方法清单（`grep -n "    def "`，共 6 个）：

| 方法 | 位置 | 形态 |
| :--- | :--- | :--- |
| `__init__` | `llm.py:54-85` | 显式 9 个参数 + `**kwargs` 全部透传给 `EngineArgs`；强制 `offline_inference=True`，并 `setdefault(disable_log_stats=True, stop_preempted_request=False)` |
| `get_tokenizer()` | `llm.py:87-88` | 直接返回 `self.llm_engine.tokenizer` |
| `generate(prompts, sampling_params, prompt_token_ids, use_tqdm=True, multi_modal_data=None)` | `llm.py:90-142` | 同步阻塞；内部 `_add_request` 逐条入队，再 `_run_engine` 跑到全部完成 |
| `_add_request(...)` | `llm.py:144-166` | 私有：构造 `Sequence` 并入队，`request_id` 用自增 `Counter` |
| `_run_engine(use_tqdm)` | `llm.py:168-195` | 私有：`while unfinished: step()`，最后按 `request_id` 数值排序；`decode_sequence=True` 时把 token 序列 decode 成字符串 |
| `close()` | `llm.py:197-198` | `self.llm_engine.close()` |

行为要点（都属 A 级）：
- **纯同步、无 streaming、无 async**：`generate` 返回 `list[list[RequestOutput] | str]`，`decode_sequence=True`（默认）时是字符串列表，否则是每请求的 `RequestOutput` 增量列表（`llm.py:188-190`）。
- **批量自动拼装**：`generate` 文档字符串写明 "This class automatically batches the given prompts"（`llm.py:98-100`），没有 continuous batching 的外部开关，全部由 `Scheduler` 决定。
- **`kwargs` 是真正的并行度入口**：`expert_parallel_size` 是唯一显式暴露的并行参数；`data_parallel_size` / `context_parallel_size` / `engram_parallel_size` / `max_num_batched_tokens` / `chunked_prefill` 等只能通过 `**kwargs` 传给 `EngineArgs`（`engine/arg_utils.py:21-46`），例如 `examples/_config.py:113-124` 用 `context_parallel_size=cp_size` 走的就是这条路。
- **`EngineArgs` 的并行度字段**：`data_parallel_size=0`、`context_parallel_size=1`、`expert_parallel_size=1`、`engram_parallel_size=0`、`engine_rank`、`engine_group_rank`（`engine/arg_utils.py:28-33`）。
- **没有 serving 层**：全仓无 `AsyncLLM`、无 OpenAI 兼容 HTTP server；唯一的 HTTP 是 profiling（`engine/profile_server.py:1-10` 明确写"每个 engine 进程一个，默认关"）。

## 4. 新发现的问题

1. **输入准备逻辑分散在四处，是读懂"一步 step 的输入怎么来"的最大障碍。** `csrc/engine/input_metadata/`（`prepare_tensor_input_base.cpp`/`_v4.cpp`/`block_load_plan.cpp`）、`deepinfer/model_executor/input_metadata/`（`input_metadata.py`/`prepare_tensor_input_zos.py`/`yoco.py`/`dflash.py`）、`deepinfer/ve/prepare_tensor_input_vl.py`、`deepinfer/model_executor/model_metadata_utils.py`（调 `prepare_prompt_v4`/`prepare_decode_v4`）各承担一段。建议 R05/R06 单独画一张"输入准备时序图"，否则书里会出现同一件事讲三遍。
2. **`FlopsEstimator` 是 C++ 调度器的一部分，vLLM 没有对应物。** `csrc/engine/flops_estimator.hpp`（330 行）+ `deep_cluster.hardware.get_bf16_tflops_limit` + `engine/llm_engine.py:17` 的 import 说明调度决策里掺了"预估 FLOPs"。它是用来定 step 预算还是定 chunk 大小，值得单独一问（R05）。
3. **CI 配置未开源。** `pyproject.toml:30-36` 定义了 `ci` / `ci_npu` / `benchmark` / `long_context` / `gpu` 五个 marker，但仓库没有 `.github/` 目录；无法确认哪些测试真的跑在 CI 上。
4. **`pytest` 是 kernel 模块的顶层依赖。** 29 个 `tilelang_ops/` 文件在模块级 `import pytest`（用于 `pytest.mark.parametrize` 生成 kernel 变体），而 `pytest` 只在 `requirements-dev.txt` 里。生产部署是否需要剥离这些 import，未确认。
5. **命名不一致暗示迁移未收尾。** `tilelang_ops/cuda/prepare_decode_metadata.py` 对应顶层与 `asc/` 的 `prepare_v4_decode_metadata.py`，少了 `v4`；`deepinfer/model_executor/models/deepseek.py` 里的 `DeepseekForCausalLM` 没有进 `_MODELS` 注册表。
6. **`vendor/` 的 pin 策略不统一。** 只有 DeepEP 停在带版本号的 `v1.2.1-43-gf6aa38f`，其余五个都在 `main`/裸 commit（`3b25377`、`1fc9c66`、`3a35326`、`bfa4507`、`2e5429f`），且 DeepEP-Ascend / DeepGEMM-Ascend / DeepSelect 没有 tag。复现构建时无法用版本号表达，只能钉 commit。
7. **`README.md:38-42` 说 "most required engine functionality is already implemented"，但代码里 `EngineType` + `init_pd_split`（`engine/llm_engine.py`）只到"骨架"级别。** 这句话的边界需要 R06 用代码证实，否则书里容易把 PD 分离写成"已支持"。

## 5. 可用于图表的材料

### 5.1 分层架构图（节点 + 边，可直接画）
```
examples/offline_inference.py · eval/run_eval.py · tests/
        │  只依赖 LLM.generate()
deepinfer/entrypoints/llm.py  (LLM)
        │
deepinfer/engine/llm_engine.py  (LLMEngine: add_request → step → post_process)
        ├── deepinfer/_C/engine  ←── csrc/engine/*.hpp|cpp  (Scheduler, BlockManager, Sequence, RequestOutput, FlopsEstimator)
        ├── deepinfer/worker/worker.py (Worker, 每卡一个；driver 在 engine 进程内，CP rank 1..n 是子进程)
        │        └── model_runner.py → model_executor/models/deepseek_vl.py → v4_modules/* → tilelang_ops/ops.py → cuda/ | asc/
        ├── deepinfer/worker/cache_engine.py  (KV cache 布局/分配)
        └── deepinfer/engine/stats_logger.py + profile_server.py
                     ↓ 设备侧
vendor/FlashMLA · vendor/DeepGEMM · vendor/DeepEP · vendor/DeepSelect · tile-kernels(pip) · tilelang(pip)
```

### 5.2 代码量条形图数据（一阶，物理行）
`deepinfer/` 47,172 ｜ `tests/` 3,117 ｜ `examples/` 3,018 ｜ `eval/` 1,694 ｜ `deep_cluster/` 656 ｜ `csrc/` 5,909（C++/头文件）｜ `vendor/` 115,153（自有部分，含 33,753 CUDA）

### 5.3 "钱花在哪"饼图数据（`deepinfer/` 内部，47,172 行）
`layers/tilelang_ops/**` 23,331（49.5%）｜ `models/**` 7,051（14.9%）｜ `worker/` 2,805（5.9%）｜ `engine/` 1,957（4.1%）｜ `hugging_face/` 1,375（2.9%）｜ 其余 10,653（22.6%）

### 5.4 双后端镜像表（可直接做"同一算子两套实现"的对照表）
顶层接口 22 个 >150 行文件 ↔ `cuda/` 12 个 ↔ `asc/` 15 个；`asc/` 独有 `kv_quant.py`，`cuda/` 独有 `prepare_decode_metadata.py`（其余同名）。

### 5.5 vLLM 派生 vs 自研的分布图
23 个文件 / 6,817 行（12.2%）集中在 engine+worker+layers+顶层；`csrc`、`tilelang_ops`、`v4_modules`、`ve`、`structure_output`、`hugging_face`、`deep_cluster` 零命中。

### 5.6 构建流程三步图
`export DEEPINFER_ACCELERATOR=nvidia|ascend` → `build.sh`（装 requirements + 覆写 NCCL）→ `scripts/build_vendor.sh`（4 个 submodule 各出 wheel 再 `pip install --no-deps`）→ `scripts/build.sh build`（`setup.py build` 编 `deepinfer._C`，`.so` 软链进 `deepinfer/_so/<tag>/`）。

### 5.7 注释语言热力表
直接用 Q6 的 22 行表格；结论一句话："C++ 69% 中文、Python 22% 中文，且中文密度与自研程度正相关"。
