# R18 上手条件与配置速查（构建 / 环境变量 / 参数 / 硬件门槛 / 无卡学习路径）

## 0. 元信息

- 代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21 12:55 +0800，`Bump DeepEP to f6aa38fe ...`）。本文所有 `路径:行号` 均指该 commit。本地路径 `research/sources/DeepInfer`（`work` 是指向 `deepinfer-book-spec/work` 的软链）。
- 查阅的主要来源：
  - 构建链：`build.sh`、`Makefile`、`scripts/build.sh`、`scripts/build_vendor.sh`、`scripts/gen_stub.sh`、`setup.py`、`pyproject.toml`、`requirements{,-nvidia,-ascend,-dev}.txt`、`.gitmodules`、`deepinfer/__init__.py`
  - 配置面：`deepinfer/environ.py`（全文 221 行）、`deepinfer/config.py`、`deepinfer/engine/arg_utils.py`、`deepinfer/engine/engine_type.py`、`deepinfer/entrypoints/llm.py`、`deepinfer/sampling_params.py`、`deepinfer/worker/base.py`、`deepinfer/worker/runtime_config.py`、`deepinfer/platform.py`、`deep_cluster/hardware/{__init__,device}.py`
  - 例子与 profiling：`examples/README.md`、`examples/benchmark.py`、`examples/_config.py`、`examples/_launch.py`、`examples/_trace.py`、`examples/offline_inference.py`、`examples/yoco/{README.md,yoco_equiv.py,run_yoco_equiv_l2.sh}`、`examples/sweep/run_sweep.sh`、`deepinfer/engine/profile_server.py`、`deepinfer/timer.py`
  - 官方口径：`README.md`、`doc/inference_deployment_qa.md`（下文简称 `doc:NNN`）
  - 测试：`tests/`（19 个 `test_*.py`）、`deepinfer/model_executor/layers/tilelang_ops/conftest.py`
- 环境说明：本机**无 GPU / 无 NPU / 未安装 torch**，未运行任何 benchmark、未编译任何 `.so`。唯一实际执行的是纯 Python 导入（见 #21，命令与输出均为实测）。凡"应当能跑"的判断都标 E。
- 与已有笔记的分工：`R10` 已详述 vendor 五库的编译目标与 TileLang 双后端；本文只补"上手视角"的步骤、失败点与配置面。重复处标注"见 R10"。
- 环境变量总数：`deepinfer/environ.py` 内 **59** 个 `DEEPINFER_*`（58 个 `EnvVar` 赋值 + 1 个 `@EnvVar` 装饰函数），另有 **5** 个不在 `environ.py` 里的 `DEEPINFER_*`（C++ 日志级别、例子脚本、评测脚本、算子自测）。全表见 §5.1。

## 1. 三个最重要的发现

1. **构建是一条命令串起来的三段链，产物按 `<accelerator>-<arch>` 分目录，能不能 `import deepinfer._C` 取决于这个目录名在构建期和运行期是否一致。** `build.sh` 依次做：装 requirements（含 dev 工具与 torch cu132 wheel）→ 打 NCCL 补丁 → 把 6 个 submodule 编成 4 个 wheel 并本地安装 → 编 C++/pybind 扩展；`scripts/build.sh` 把 `.so` 软链进 `deepinfer/_so/nvidia-x86_64/`，`deepinfer/__init__.py:9-32` 在 import 时用同一套标签算目录并 `__path__.append`。所以 `DEEPINFER_ACCELERATOR` 不是"只给构建用"的开关，它同时决定运行时找哪份 `.so`。
2. **参数与环境变量的优先级是"环境变量赢"，而且 `envs` 是首次访问即缓存的描述符。** `EngineArgs.__post_init__`（`arg_utils.py:48-60`）用 `os.environ.setdefault` 把 `engine_type` 派生成 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` / `DEEPINFER_USE_MULTIMODAL_POOL`，再 `environ.rearm()` 清缓存；反过来，运行中改 `os.environ` 对已读过的变量无效（`environ.py:24-44` 的 `__get__` 把值写回类字典）。这是"命令行参数传了却不生效"和"设了环境变量却没反应"两类问题的共同根因。
3. **官方文档给的是生产口径，仓库代码给的是"能跑"口径，两者相差一个数量级。** 文档的最小高效部署单元是 **64 卡 prefill + 64 卡 decode**（`doc:65-67`，decode 推荐 EP64–128），而 `examples/offline_inference.py:13` 的默认就是单机 **8 卡 EP8**；硬件假设只以**算力常量**形式出现（`deep_cluster/hardware/device.py:151-164`：SM100 = 2250/2500 TFLOPS BF16，昇腾 = 432），而这个常量被 `LLMEngine.__init__` 用来算 MFU，于是非 SM100 的卡在建引擎时就抛 `RuntimeError: unknown GPU type`——不是显式检查，报错也不解释原因（#15）。

## 2. 事实条目

### #1 `build.sh` 只有 27 行，把构建分成"依赖 / vendor / 本体"三段，且第一件事是校验加速器

- 证据：`build.sh:5-27`
  > `case "${DEEPINFER_ACCELERATOR:-}" in nvidia|ascend) ;; *) echo "Usage: DEEPINFER_ACCELERATOR=nvidia|ascend $0" >&2; exit 1 ;; esac`
  > `pip install -r "requirements-${DEEPINFER_ACCELERATOR}.txt"`
  > `bash ./scripts/build_vendor.sh`
  > `bash ./scripts/build.sh build`
- 等级：A
- 意味着什么：第 16 章讲"怎么装"时，可以照这三段讲，不需要读者理解 Makefile；`make build-nvidia` 只是同一件事的包装（`Makefile:3-6`），且**不装 requirements**。
- 疑点：无。

### #2 依赖安装会把 dev 工具一起装进来（因为 `requirements-nvidia.txt` 第一行是 `-r requirements-dev.txt`）

- 证据：`requirements-nvidia.txt:1-5`、`requirements-dev.txt:1-18`、`requirements.txt:1-20`
  > `-r requirements-dev.txt`
  > `--find-links https://download.pytorch.org/whl/cu132/torch/`
  > `torch==2.13.0+cu132`
  > `flash-attn-4==4.0.0b34`
  > （dev 侧）`clang-format==22.1.3`、`pytest==8.4.2`、`ruff==0.15.0`、`matplotlib==3.11.2`、`fastapi==0.128.0`、`uvicorn==0.40.0`
- 等级：A
- 意味着什么：一次 `./build.sh` 之后 `pytest`、`matplotlib`（Pareto 画图）、`fastapi/uvicorn`（profile HTTP 服务）都已就绪，第 16 章的实验不必再单独 `pip install`。反面是镜像会变大，且生产部署没有"只装 runtime"的 requirements。
- 疑点：`requirements.txt` 里 `torch` 不在列（由 nvidia/ascend 两个文件各自钉版本），直接用 `pip install -r requirements.txt` 装不出可运行环境。

### #3 NVIDIA 路径会强制把 NCCL 换回 2.32.3

- 证据：`build.sh:16-19`
  > `if [ "${DEEPINFER_ACCELERATOR}" = nvidia ]; then`
  > `    pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3`
- 等级：A
- 意味着什么：torch cu132 wheel 自带的 NCCL 版本与 DeepInfer 预期不一致，官方用"覆盖安装、不带依赖"的方式钉死。读者若自己升级过 NCCL，行为可能与他们不同——复现性能数字时要先确认这一条。
- 疑点：仓库没写为什么必须是 2.32.3（E：大概率与多机 EP 的 GIN/PXN 行为有关，见 `worker/runtime_config.py:34-40` 关掉 `NCCL_NVLS_ENABLE`、`NCCL_PXN_DISABLE=1`）。

### #4 vendor 段：6 个 submodule → 4 个 wheel → `--no-index` 本地安装，任何一包失败整段失败

- 证据：`scripts/build_vendor.sh:7-22,27-28,38-53`
  > `nvidia) BUILD_TARGET=CUDA; PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM); INSTALL=(deep_select flash_mla deep_ep deep_gemm) ;;`
  > `git -C "${REPO_ROOT}" submodule update --init`
  > `git submodule update --init --recursive`
  > `env "${envs[@]}" python setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"`
  > `python -m pip install --no-deps --no-index --find-links "${WHEEL_DIR}" "${INSTALL[@]}"`
- 等级：A
- 意味着什么：vendor 库**不会**从 PyPI 装预编译包，必须本地现场编；Ascend 走 `DeepEP-Ascend`/`DeepGEMM-Ascend`，但装出来的 Python 包名与 NVIDIA 完全相同（`deep_ep`/`deep_gemm`），因此同一环境不能同时存在两套。变量传递：`DeepSelect`/`FlashMLA` 收 `<PKG>_BUILD_TARGET_PLATFORM=CUDA|ASCEND`，`DeepGEMM` 额外收 `DG_FORCE_BUILD=1`（`:30-44`）。
- 疑点：无。

### #5 本体扩展：CUDA 与 NPU 共用 11 个源文件，只换 2 个后端文件；编译标准是 C++20 + CXX11 ABI

- 证据：`setup.py:12-34,38-72`
  > `COMMON_COMPILE_FLAGS = ["-O2", "-std=c++20", "-Wno-deprecated-declarations"]`
  > `ABI_FLAG = "-D_GLIBCXX_USE_CXX11_ABI=1"`
  > `assert torch._C._GLIBCXX_USE_CXX11_ABI, "must use CXX11 ABI"`
  > （CUDA 侧）`sources=[*COMMON_SOURCES, "csrc/cuda_utils.cpp", "csrc/graph_timer/graph_timer_cuda.cpp"]`，`libraries=[..., "cuda"]`，`library_dirs=[os.path.join(CUDA_HOME, "lib64", "stubs")]  # for libcuda.so`
- 等级：A
- 意味着什么：`csrc/engine/*`（调度器、ZOS、input metadata、采样）两平台共用，所以"调度逻辑只有一份"；链接用的是 CUDA toolkit 的 stub 库，**编译和链接不需要真卡、不需要驱动**（只需 nvcc/CUDA 13），这对无卡读者很关键（见 Q6）。
- 疑点：`torch_npu` 路径用 `NpuExtension`，`extra_compile_args` 里多了 `-fopenmp -fvisibility=hidden -DWITH_NPU`（`setup.py:48-50`），CANN 的编译选项仓库里看不到。

### #6 CUDA 路径硬性要求处于 conda/venv 中，否则 `setup.py` 直接抛错

- 证据：`setup.py:55-57`
  > `conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")`
  > `if not conda_prefix: raise RuntimeError("Neither CONDA_PREFIX nor VIRTUAL_ENV is set; a conda env or virtualenv is required to build.")`
- 等级：A
- 意味着什么：README 说"先建一个空的 Python 3.12 环境"不是建议而是硬约束；用系统 Python 直接 `./build.sh` 会在链接阶段前就失败。注意 NPU 分支没有这个检查。
- 疑点：`conda_prefix` 实际未被使用（只用于判空），E：历史遗留，可能是想设 `-L$CONDA_PREFIX/lib`。

### #7 C++ 产物落到 `build/<accel>-<arch>/`，`.so` 软链到 `deepinfer/_so/<accel>-<arch>/`，链接失败有兜底断言

- 证据：`scripts/build.sh:11-29`
  > `PY_BUILDTAG=$(python -c "import sys, sysconfig; print(f'{sysconfig.get_platform()}-{sys.implementation.cache_tag}')")`
  > `PLATFORM_TAG="${DEEPINFER_ACCELERATOR:-nvidia}-$(uname -m)"`；`SO_DST="deepinfer/_so/$PLATFORM_TAG"`
  > `for so in "${BUILD_BASE}/lib.${PY_BUILDTAG}/deepinfer/"*.so; do ln -fs "$(realpath --relative-to="$SO_DST" "$so")" "$SO_DST/"; done`
  > `[ -n "$(ls -A "$SO_DST")" ] || { echo "No .so was linked into $SO_DST" >&2; exit 1; }`
  > `python -m compileall deepinfer -j "$MAX_JOBS" -q`
- 等级：A
- 意味着什么：`MAX_JOBS` 默认 16（`:6-9`），编不动时第一个该调的就是它；`.so` 是软链不是拷贝，`make clean` 会同时删 `build/<tag>` 和 `deepinfer/_so/<tag>`（`Makefile:21-25`）。`failglob` 使"一个 `.so` 都没生成"变成硬错误而不是静默成功。
- 疑点：无。

### #8 运行期靠 `deepinfer/__init__.py` 重算同一个标签来挂 `__path__`，这是双平台共存于同一棵代码树的机制

- 证据：`deepinfer/__init__.py:9-32`
  > `accelerator = envs.DEEPINFER_ACCELERATOR`，探测失败则 `accelerator = "nvidia"`
  > `_so_dir = os.path.join(os.path.dirname(__file__), "_so", _so_platform_tag())`
  > `if os.path.isdir(_so_dir): __path__.append(_so_dir)`
- 等级：A
- 意味着什么：`import deepinfer._C` 找的是 `deepinfer/_so/<tag>/_C.<soabi>.so`；如果构建时 `DEEPINFER_ACCELERATOR=ascend`、运行时没设而机器探测成 nvidia（或反之），就会 `ModuleNotFoundError: deepinfer._C`。README 因此强调"Keep it exported in the environment where you run DeepInfer"（`README.md:59-61`）。
- 疑点：`_so` 目录不存在时 `__path__` 不变，报错信息是普通的 `ModuleNotFoundError`，不提示"你是不是没 build 或标签不一致"。

### #9 `environ.py` 不是 `os.getenv` 包装，而是"首次访问即缓存"的非数据描述符表

- 证据：`deepinfer/environ.py:24-44,66-96,212-219`
  > `class EnvVar[T]: """Non-data descriptor, resolved on first access"""`
  > `value = self.init(os.getenv(self.name)); setattr(owner, self.name, value)`
  > `def rearm(name): """Rearm specific descriptor, will resolve again on next access.""" setattr(envs, name, _descriptors[name])`
- 等级：A
- 意味着什么：①测试或脚本"先 import 再设环境变量"是无效的，必须 `environ.rearm()`（引擎自己在 `EngineArgs.__post_init__` 里就这么做）；②类型与默认值集中在声明处，附录 B 可以逐行机械核对，不会漏。
- 疑点：`EnvVar.__get__` 是多线程安全的（`with self.lock`），但 `rearm` 不加锁——只在启动期调用，可接受。

### #10 `_bool` 只认 4 个假值，`off`/`no`/`False ` 全被当成 True

- 证据：`deepinfer/environ.py:12-13`
  > `return (v if v is not None else default) not in ("0", "", "false", "False")`
- 等级：A（实测：`_bool("False")` → `False`，`_bool("off")` → `True`）
- 意味着什么：写 `DEEPINFER_BATCH_INVARIANT=off` 会**打开**批不变，进而（见 #13）把 deep_gemm / tile_kernels 切到确定性算法、改变性能。附录 B 应明确写"布尔变量只接受 0/1"。
- 疑点：无。

### #11 三个变量的默认值是"派生"的，不能只看等号右边

- 证据：`environ.py:129-132,147`
  > `DEEPINFER_MAX_BACKTRACKING_TOKENS = EnvVar(lambda v: max(_int(v), envs.DEEPINFER_USE_NEXTN_DECODE + 1))`
  > `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE = EnvVar(lambda v: _bool(v) or envs.DEEPINFER_ACCELERATOR == "ascend")`
  > （另有 `DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 支持逗号分隔、且个数必须等于 `nextn`，`:135-144`）
- 等级：A（实测：默认 `nextn=5` → `MAX_BACKTRACKING_TOKENS=6`；`nextn=3` → `4`；`DEEPINFER_ACCELERATOR=ascend` 时 `ENGRAM_FORCE_GPU_STORAGE=True`）
- 意味着什么：附录 B 的"默认值"列对这三项必须写表达式而不是常量；也说明"回退长度 ≥ 投机长度+1"是硬约束，YoCo 的 140（窗口 128 + 回退）正是据此推出来的（`README.md:25`）。
- 疑点：无。

### #12 `EngineArgs` 26 个字段；`__post_init__` 会把 `engine_type` 反写成两个环境变量

- 证据：`deepinfer/engine/arg_utils.py:21-60`；`deepinfer/engine/engine_type.py:11-17`
  > `self.engine_type: EngineType = EngineType.NORMAL`（默认）
  > `os.environ.setdefault("DEEPINFER_ZERO_OVERHEAD_SCHEDULING", "1" if self.engine_type.default_zero_overhead_scheduling else "0")`；`environ.rearm(...)`
  > `default_zero_overhead_scheduling → return self in (EngineType.NORMAL, EngineType.COMPLETION)`
- 等级：A
- 意味着什么：`PROMPT` 引擎（纯 prefill 服务）默认**关**零开销调度，`NORMAL`/`COMPLETION` 默认**开**；由于用的是 `setdefault`，外部显式设 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING=0` 可以覆盖引擎类型——这就是 #2 说的"环境变量赢"。
- 疑点：`EngineType` 的 C++ 侧定义在 `csrc/engine/scheduler.hpp`，Python 注释要求两边保持一致（`engine_type.py:8`），但仓库没有自动校验。

### #13 `BATCH_INVARIANT` 不只是采样开关，还会改两个 kernel 库的算法

- 证据：`deepinfer/worker/runtime_config.py:54-68`
  > `if envs.DEEPINFER_BATCH_INVARIANT or PLATFORM.is_npu: deep_gemm.use_deterministic_algorithms(True); tile_kernels.use_deterministic_algorithms(True)`
  > `if PLATFORM.is_npu: assert not envs.DEEPINFER_BATCH_INVARIANT  # TODO(NPU): 待完整支持`
- 等级：A
- 意味着什么：批不变/确定性解码的代价落在 GEMM 与 tile kernel 上，不只是 sampler；NPU 上直接断言禁止（所以 README 的 "NVIDIA only" 有代码依据，`README.md:27-28`）。`examples/offline_inference.py:42-43` 在 NVIDIA 上默认就设 `BATCH_INVARIANT=1`，而 `benchmark.py` 不设——**两个例子的数值/性能不可直接对比**。
- 疑点：无。

### #14 `gpu_memory_utilization` 的语义是"总显存的一个比例上限"，KV block 数由 profile run 反推，并在 world 组内取平均

- 证据：`deepinfer/worker/base.py:225-253`
  > `peak_allocated_memory = peak_torch_allocated_memory + max(non_torch_allocated_memory, 0)`
  > `available_kv_cache_memory = total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_allocated_memory`
  > `state_size *= 2  # state store & copy buffer`
  > `num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)`
  > `num_gpu_blocks = torch.tensor(...); dist.all_reduce(SUM); num_gpu_blocks = ... // size // 2 * 2`
- 等级：A
- 意味着什么：**显存不是"配置多少"，而是"跑完一个 profile step 后剩下多少"**，所以第 16 章讲显存需求时唯一可给的公式是这条；log 里会打出 `total_gpu_memory / peak_torch_memory / state_size / kv_cache_size` 五项（`:255-262`），是读者自查显存瓶颈的第一手数据。
- 疑点：block 数取的是**全局平均**（`sum // world_size`）而不是最小值，理论上显存更少的 rank 会分到多于自己可用量的 block。E：实际靠对称配置保证，但没有断言。

### #15 硬件门槛以"算力常量"的形式出现，并顺带构成一道启动期的隐式门禁

- 证据：`README.md:33-52`（`NVIDIA SM100 GPUs` / `Huawei Ascend 950 NPUs`；Linux、GCC ≥ 13、Python 3.12、CUDA 13、CANN 9.2）；`deep_cluster/hardware/device.py:151-164`；`deepinfer/engine/llm_engine.py:300-305`
  > `def get_bf16_tflops_limit(): if accelerator_kind()=="ascend": return 432; if is_sm_major_10(): return 2500 if platform.machine() in {"aarch64","arm64"} else 2250; raise RuntimeError("unknown GPU type")`
  > `self.stats_logger = StatsLogger(..., tflops_limit=get_bf16_tflops_limit() * 2,  # FP8-equivalent peak: 2x BF16.`
- 等级：A（型号与前置条件是 B：官方 README）
- 意味着什么：`is_sm_major_10()`（compute capability major == 10）只用在两处——算力上限、ViT 的融合线性层（`models/vision_encoder/vit.py:42`）；但 `get_bf16_tflops_limit()` 在 `LLMEngine.__init__` 里被调用来给 `StatsLogger` 传 MFU 分母，所以**任何 NVIDIA 非 SM100 的卡（以及纯 CPU 机器）在建引擎时就会 `RuntimeError: unknown GPU type`**。这是事实上的硬门禁，比"跑到 kernel 才炸"更早，只是错误信息不解释原因。
- 疑点：门禁是通过算力表间接实现的，不是显式的 capability 断言——若将来给旧卡补了常量，这道门就自动消失。另外 2250/2500 是 BF16 **dense** 峰值，书中引用时必须注明 dense、并说明 FP8 口径是 ×2（`llm_engine.py:304` 的注释）。

### #16 单机 8 卡可以跑：`offline_inference.py` 的默认就是 EP8、DP8

- 证据：`examples/offline_inference.py:12-29,45-57`；`examples/_config.py:86-125` 是同一套构造
  > `--ep` 默认 `8`；`dp_size = max(ep_size // cp_size, 1)`；`world_size = dp_size * cp_size`；`assert world_size % ep_size == 0`
  > `create_llm(..., data_parallel_size=dp_size, max_num_batched_tokens=32768, max_model_len=32768, max_num_seqs=4, gpu_memory_utilization=0.85, lookahead_tokens=min(1, nextn))`
- 等级：A
- 意味着什么："最小可运行配置"就是 8 卡单机（一个 DP 组、EP8、CP1），与文档的 prefill 区间 EP2–8/DP2–8 对齐（`doc:45-47`）；decode 侧 8 卡只是文档区间的下限（EP8 vs EP64 的对比图正是文档用来论证 Wide-EP 收益的，`doc:55-59`）。
- 疑点：显存是否够装 V4.1-Flash 的 1/8 权重 + KV，仓库没有数字；无法从代码判断，只能实测（见 Q4）。

### #17 生产拓扑（官方 Q&A）：P 与 D 的并行策略不对称，最小高效单元 64+64

- 证据：`doc:45-67`
  > `Prefill: EP2–8 / DP2–8; use the smallest EP/DP that HBM allows.`
  > `Decode: EP64–128 / DP64–128 (Wide-EP); use the largest EP/DP that the available GPUs allow.`
  > `VE: a single GPU per instance is generally enough, and cheaper GPUs are recommended.`
  > `TP and CP are not used for now.`
  > `1P + 1D is deployable. ... the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.`
- 等级：B
- 意味着什么：第 16 章必须先讲清"能跑（8 卡）"与"高效（128 卡起）"的差别；也解释了为什么 README 的性能图是"prefill 4 卡、decode 64 卡"（`README.md:83`）——两条曲线本来就在不同规模的域里测的。
- 疑点：文档说"TP 和 CP 暂不使用"，但代码有完整的 `context_parallel_size` 与 `--cp`（`arg_utils.py:29`、`config.py:154,171-173`），README 也把 "balanced CP" 列为特性。这是"支持 ≠ 生产使用"，书中不能写成"DeepInfer 不支持 TP/CP"。

### #18 `benchmark.py` 的默认就是"开 profile"，`run_sweep.sh` 才强制关

- 证据：`examples/benchmark.py:68`、`examples/sweep/run_sweep.sh:71-72`
  > `profiling_enabled = bool(int(os.environ.get("PROFILE", "1")))`
  > `# PROFILE=0：profiler 会扰动稳态读数，扫点一律关掉`
  > `MASTER_PORT=$((BASE_PORT + IDX)) PROFILE=0 python -u "${REPO_ROOT}/examples/benchmark.py" ...`
- 等级：A
- 意味着什么：`examples/README.md:47` 写的是 "With `PROFILE=1`"，读者会以为默认关；实际默认开，直接 `python examples/benchmark.py` 就会采 trace（也因此更慢、更吃显存）。要纯性能数字必须显式 `PROFILE=0`。
- 疑点：文档与代码的这处不一致值得在第 16 章点出来，避免读者照文档操作后怀疑自己的机器。

### #19 三个性能指标的口径写死在文档里，且 decode 吞吐是"全接受"假设

- 证据：`examples/README.md:45,53-55`
  > 默认 `decode_num_requests=64, context_length=32768, non_cached_tokens=4096, prefill_batch_size=32768`，`nextn=5`
  > `prefill throughput (tokens/s/GPU)`：跳过 5 个 warmup chunk，取 10 个稳态区间，再除以设备数
  > `decode avg_SPS (steps/s)`：跳过 50 步 warmup + 60 步带日志，取 60 个稳态区间
  > `decode throughput = avg_SPS × decode_num_requests × (1 + nextn) / --cp`，全接受口径
- 等级：B
- 意味着什么：书的性能数字必须带这三条口径，否则与官方图不可比；"全接受"意味着 decode 吞吐是**上界**，真实值要乘接受率（`offline_inference.py` 会打印 `accept_rate`，可作为修正因子）。
- 疑点：`--cp` 出现在吞吐公式的分母里，说明 CP 是把一个 DP 组切成 cp 份、每份独立算 token——与 `config.py:171-173` 的 `model_parallel_size = cp` 一致。

### #20 trace 的结构：GPU 事件的 `pid` 是 device index，CPU 与 GPU 靠 `correlation`/`External id` 关联，step 边界是 `deepinfer:schedule`

- 证据：`examples/_trace.py:12-15,95-97`；`deepinfer/timer.py:60`；`deepinfer/engine/llm_engine.py:881-907`
  > `# torch chrome trace 中 CPU 事件的 pid 为进程号，GPU 事件的 pid 为 device index（driver rank 恒为 0）`
  > `_TRACE_ID_KEYS = ("correlation", "External id")`
  > `return sorted(event["ts"] for event in events if event.get("name") == "deepinfer:schedule")`
  > `with torch.profiler.record_function(f"deepinfer:{record_name or name}"):`
- 等级：A
- 意味着什么：打开 trace 后按 `deepinfer:` 前缀就能把 CPU 侧阶段（`schedule`、`schedule:cpp`、`run_workers`、`prepare_input`、`launch_forward_N?_B?`、`launch_sample`、`bubble`）与 GPU kernel 对上；`deepinfer:schedule` 是每 step 一个，用来切分 step、算稳态区间。
- 疑点：`launch_forward` 的 `record_name` 带 `N{num_reqs}_B{total_q}`（`worker/model_runner.py:322`），trace 里能直接读出 batch 形状，但这是 CPU 事件名，同名 kernel 仍要按时间窗归属。

### #21 无卡可做的第一件事：`import deepinfer` 与 `envs` 表在裸 Python 上就能跑（实测）

- 证据：本机实测（无 torch、Python 3.14.7，仓库根目录执行）
  ```
  $ python3 -c "import deepinfer; from deepinfer import envs; print(envs.DEEPINFER_USE_NEXTN_DECODE, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS)"
  5 6
  $ DEEPINFER_ACCELERATOR=ascend DEEPINFER_USE_NEXTN_DECODE=3 python3 -c "from deepinfer import envs; print(envs.DEEPINFER_ENGRAM_FORCE_GPU_STORAGE, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS)"
  True 4
  ```
- 等级：C（可复现，命令与输出如上；源码依据 `deepinfer/__init__.py:1` 明令"严禁 import torch"）
- 意味着什么：附录 B 的每一行默认值都可以用这种方式在**任何**机器上核对，包括派生默认与 `_bool` 的坑；这也是第 16 章"没有 64 张 SM100 也能动手"的第一个练习，零依赖、10 秒完成。
- 疑点：Python 版本不受 `requires-python >= 3.12` 之外的约束，实测 3.14 可用；但真正跑引擎时不能高于官方验证过的 3.12。

### #22 仓库里没有发布 TileLang 算子的测试文件，但测试脚手架还在

- 证据：`deepinfer/model_executor/layers/tilelang_ops/conftest.py:1-14,52-92`
  > `# Loads the tile_kernels pytest plugins (CLI options, markers, fixtures) for tests in .../tilelang_ops/`
  > `pytest_plugins = ["tile_kernels.testing.pytest.benchmark", "...gpu_mem", "...random", "...tilelang_warning", "...xdist_failfast"]`
  > `def pytest_collect_file(...)  # 收集目录下任何含 def test_ 的顶层 *.py`
  > `TK_BENCHMARK_BASELINES_DIR = .../benchmark_baselines`（该目录不存在）
- 等级：A（`find ... -name "test_*.py"` 在 `tilelang_ops/` 下返回 0；`benchmark_baselines/`、`gpu_mem_profiles/` 目录均不存在）
- 意味着什么：想"读测试学算子"的读者在这里会扑空——开源版只给了测试**框架**（基准回归阈值 0.15、显存 profile、xdist 绑卡），没给每个 kernel 的用例。第 16 章应把练习改为"读 `cuda/` 与 `asc/` 同名文件对照"，而不是"跑 kernel 测试"。
- 疑点：`conftest.py:26-29` 提到"conftest.py turns it into a test skip on non-cuda backends"，说明内部版有测试；开源时被裁掉了。

### #23 一个可读、可跑（有 CUDA toolkit 时）的 CPU 侧测试：`test_sequence.py` / `test_flops_estimator.py`

- 证据：`tests/engine/test_sequence.py:1-10`；`tests/engine/test_flops_estimator.py:1-21`
  > `import torch  # noqa: F401  Load torch libraries before the C++ extension.`
  > `from deepinfer._C.engine import Sequence, encode_compressed`
  > `pytestmark = [pytest.mark.ci, pytest.mark.ci_npu]`
  > （flops 测试）`MODEL_NAMES = [DEFAULT_MODEL_NAME]`，fixture 里 `ModelConfig(...)` + `ModelMetadata(config.model_cfg.gpt)`
- 等级：A（代码事实）；"能跑"是 E（本机无 torch/无 `.so`，未执行）
- 意味着什么：这两个测试只碰 C++ 的 `Sequence`/`FlopsEstimator` 与模型元数据，不 launch kernel，是最适合无卡读者的"真实单测"；但都需要先编出 `deepinfer._C`（要 CUDA 13 toolkit），且 flops 测试会 `snapshot_download` 模型 config（只需 config.json，实际会拉整个 repo 的元数据，E）。
- 疑点：`test_flops_estimator.py` 依赖 `examples._config.resolve_model_path`，说明测试与示例共享同一套模型解析逻辑；离线环境必须预置 `DEEPINFER_HF_MODEL_DIR` 或本地目录。

### #24 例子不需要 `pip install -e .`：`examples/_config.py` 自己把仓库根塞进 `sys.path`

- 证据：`examples/_config.py:17-24`
  > `# python examples/xxx.py puts examples/ (the script's own directory) on sys.path rather than the repository root`
  > `REPO_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO_ROOT))`
- 等级：A
- 意味着什么：构建完直接 `python examples/offline_inference.py` 即可，不必装包；也意味着**改了 Python 源码不需要重装**，但改了 `csrc/` 必须重新 `./scripts/build.sh build`（见 #25）。
- 疑点：`_config.py` 顶层**不能** import deepinfer（`:5-7` 注释），因为要先设 `CUDA_VISIBLE_DEVICES` 再 import；这条约束对写自己的实验脚本同样适用。

### #25 改了 C++ 不重编不会报错，只会行为不一致——`yoco_equiv.py` 专门写了一个探针

- 证据：`examples/yoco/yoco_equiv.py`（README 亦有）
  > `The .so compiled from the current repository must be used; when PrepareArgs.yoco_prefill_truncation_mode is missing, the script asks you to rebuild.`
- 等级：B（`examples/yoco/README.md:37`）
- 意味着什么：`.so` 是软链、`setup.py build` 是增量编译，最容易被忽略的失败点是"跑的是旧二进制"。书中给读者的第一条纪律应是"改 `csrc/` → `bash scripts/build.sh build`"。
- 疑点：没有版本戳/哈希校验机制，只能靠字段存在性这种间接探针。

### #26 `profile` 有一条不依赖 benchmark 的 HTTP 通道：`POST /profile`，端口 = 6070 + driver_gpu_id

- 证据：`deepinfer/engine/profile_server.py:1-12,37-50,193-230`
  > `POST /profile 采 num_steps 个 step 并落盘，阻塞到文件写完才返回路径，所以没有停止接口。监听 DEEPINFER_PROFILE_HTTP_PORT_BASE + driver_gpu_id，默认关，设 DEEPINFER_PROFILE_HTTP_ENABLE=1 开启。`
  > `_DEFAULT_NUM_STEPS = 20`；`_MAX_NUM_STEPS = 50`；`# trace 实测约 4.9 MB/step`
  > `只采 driver worker 这一张卡：torch profiler 是进程内的，CP > 1 时 slave worker 在别的进程里，采不到。`
- 等级：A
- 意味着什么：长时间跑服务时可以在不中断的情况下抓 trace（第二个并发请求拿 409），比 `benchmark.py` 更贴近真实负载；但 CP>1 只能看到 driver 卡。
- 疑点：`num_steps` 上限 50、trace 约 245 MB，导出要几十秒（`_COMPLETION_TIMEOUT_S = 600`），磁盘要留够。

## 3. 必答问题逐条回答

### Q1 构建流程每一步做什么、要从网络拉什么、常见失败点

**逐步动作**（全部 A 级，见 #1–#8）：

| 步 | 动作 | 关键代码 | 产物/副作用 |
|---|---|---|---|
| 0 | 校验 `DEEPINFER_ACCELERATOR ∈ {nvidia, ascend}` | `build.sh:5-11` | 未设/拼错 → 打印 usage 退出 |
| 1 | `pip install -r requirements-<accel>.txt` | `build.sh:14` | 装 torch 2.13.0+cu132（或 +cpu & torch-npu）、flash-attn-4、tilelang 0.1.15、tile-kernels、xgrammar，以及 dev 工具（#2） |
| 2 | NVIDIA：`pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3` | `build.sh:16-19` | 覆盖 torch 自带 NCCL |
| 3 | `git submodule update --init`（6 个） | `scripts/build_vendor.sh:27-28` | 拉 DeepSelect / FlashMLA / DeepEP / DeepGEMM（+Ascend 两库）及其嵌套 third-party（cutlass 等） |
| 4 | 逐包 `python setup.py bdist_wheel` | `:38-50` | `build/<tag>/vendor-wheels/*.whl`；`DeepSelect`/`FlashMLA` 需 `*_BUILD_TARGET_PLATFORM`，`DeepGEMM` 需 `DG_FORCE_BUILD=1` |
| 5 | `pip install --no-deps --no-index --find-links ... deep_select flash_mla deep_ep deep_gemm` | `:52-53` | 四个包名与 Ascend 版相同（互斥） |
| 6 | `python setup.py build --build-base build/<tag>`（`MAX_JOBS=16`） | `scripts/build.sh:6-18` | 编 11 个公共源 + 2 个后端源 → `deepinfer._C` |
| 7 | `.so` 软链到 `deepinfer/_so/<tag>/`，`compileall` | `:19-29` | 运行期 `__path__` 挂载点（#8） |

**需要从网络拉取**（A）：
1. PyPI：`requirements.txt` 20 个包 + dev 6 个（含 `matplotlib`、`fastapi`、`uvicorn`）；
2. `https://download.pytorch.org/whl/cu132/torch/`：`torch==2.13.0+cu132`（体积最大的一项）；Ascend 走 `download.pytorch.org/whl/cpu/torch/` + `torch-npu==2.13.0rc1`；
3. `flash-attn-4==4.0.0b34`、`nvidia-nccl-cu13==2.32.3`；
4. git：6 个 submodule（`https://github.com/deepseek-ai/{DeepSelect,FlashMLA,DeepEP,DeepGEMM,DeepEP-Ascend,DeepGEMM-Ascend}.git`）；
5. 首次运行：HF checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017`（`huggingface_hub.snapshot_download`，`examples/_config.py:46-49`）；可用 `DEEPINFER_HF_MODEL_DIR` 指到本地副本避免下载（`README.md:68`，注意它只覆盖默认模型名，`_config.py:42`）。
   **不需要网络**：本体 C++ 编译（源码全在仓库内，只需本地 CUDA 13 toolkit）。

**常见失败点**（A = 代码可证；E = 推断）：

| # | 现象 | 根因 | 等级 |
|---|---|---|---|
| 1 | `Usage: DEEPINFER_ACCELERATOR=nvidia\|ascend ./build.sh` | 环境变量没 export 或拼错 | A `build.sh:7-10` |
| 2 | `RuntimeError: Neither CONDA_PREFIX nor VIRTUAL_ENV is set` | 用系统 Python 构建 | A `setup.py:55-57` |
| 3 | `AssertionError: must use CXX11 ABI` | 装的是 pre-CXX11 ABI 的 torch | A `setup.py:19` |
| 4 | `TypeError: expected str ... not NoneType`（在 setup.py） | 没有 CUDA toolkit → `CUDA_HOME is None`，`os.path.join(None, ...)` | A `setup.py:70`（推断报错形态为 E） |
| 5 | `ModuleNotFoundError: deepinfer._C` | 没 build / `DEEPINFER_ACCELERATOR` 与构建时不符 / `deepinfer/_so/<tag>` 不在 | A #8 |
| 6 | `ImportError: libc10.so: cannot open shared object file` | 直接 import `deepinfer._C` 前没 import torch（测试文件都写了 `import torch  # noqa: F401`） | A `tests/engine/test_sequence.py:5` 注释 |
| 7 | vendor 某包编译失败 → 整个 pip 安装步失败 | `--no-index` 只用本地 wheel，四包是一组 | A `build_vendor.sh:52-53` |
| 8 | 编到一半 OOM / 卡死 | `MAX_JOBS=16` 对内存小的机器太大；nvcc 每进程吃内存 | E |
| 9 | 跑起来行为与源码不符 | 增量编译留了旧 `.so` | B #25 |
| 10 | 换 torch 版本后 `.so` 加载失败/未定义符号 | `.so` 与运行时 torch 的 ABI/版本不一致（requirements 钉死 2.13.0+cu132） | E |

### Q2 全部 `DEEPINFER_*` 环境变量

见 **§5.1 表 B-1**（59 个 `environ.py` 内 + 5 个仓内其它位置，逐行给类型/默认值/语义/影响模块/出处）。要点：

- **59 个**在 `deepinfer/environ.py:101-210`，按代码里的分组：Platform 1、Runtime 2、Distributed 4、MoE&EP 8、Attention 2、MTP 4、Model Loading 2、Sampling 2、CUDA Graph 6、KV Cache Store 3、Structured Output 5（含 `SO_BUILDER_CPUS`）、Scheduling 3、Logging 4、YoCo 2、Profiling&Debug 7、Multi-Modal 4。
- **另外 5 个**：`DEEPINFER_CPP_LOG_LEVEL`（C++ 日志级别，`csrc/utils/log.hpp:21`）、`DEEPINFER_HF_MODEL_DIR`（例子/README 的本地模型目录，`examples/_config.py:42`）、`DEEPINFER_REPO`（`eval/common/paths.py:24`）、`DEEPINFER_FUSED_SAMPLE_STRESS_TEST` / `DEEPINFER_FUSED_SAMPLE_DETAIL_LOG`（`tilelang_ops/fused_sample.py:530,601`）。
- **三个陷阱**（附录 B 必须写）：① 布尔只认 `0/""/false/False` 为假（#10）；② 三个默认值是表达式（#11）；③ 首次访问即缓存，运行中改无效（#9）。
- **真正会被"生产调参"的只有十来个**：`USE_NEXTN_DECODE`、`ENABLE_MTP_VARLEN`、`YOCO_*` 两个、`OVERRIDE_NUM_EXTRA_EXPERTS`、`EXPERT_COUNT_WINDOW_SIZE`、`EXPERT_REARRANGEMENT_STEP_INTERVAL`、`BATCH_INVARIANT`、`JIT_CACHE_DIR`、`LOGGING_INTERVAL_STEP`、`ZERO_OVERHEAD_SCHEDULING`。其余是调试/profiling/多模态/结构化输出的专用开关。

### Q3 引擎参数（`arg_utils.py`、`config.py`）

见 **§5.2 表 B-2**（`EngineArgs` 26 项 + `LLM.__init__` 覆盖项 + `SamplingParams`）与 **§5.3 表 B-3**（4 个 config 类的派生规则与断言）。三个最该记住的点：

1. `EngineArgs` 的 26 项里，只有 `model` 必填；其余默认值见 #12 与表 B-2。`LLM.__init__` 会**再覆盖**四个：`disable_log_stats=True`、`stop_preempted_request=False`、`offline_inference=True`（强制）、`tokenizer=model`（`entrypoints/llm.py:54-81`）。
2. 四个派生规则（A，`config.py`）：`max_model_len` 缺省 **4096**（`:69`）；`max_num_batched_tokens` 缺省 `max(max_model_len, 4096, max_num_seqs*(1+nextn))`（`:219-228`）；`runtime_max_model_len = max_model_len + nextn + (nextn+1 if ZOS else 0) + nextn`（`:11-16`）；`data_parallel_size=0` → `max(1, ep//cp)`，`world_size = max(cp*dp, engram_ps)`（`:161-173`）。
3. 两条硬校验会成为"启动即失败"的原因：`not chunked_prefill and max_num_batched_tokens < max_model_len` → `ValueError`（`config.py:209-212`）；`max_num_batched_tokens < max_num_seqs*(1+nextn)` → `assert`（`:226`）。`gpu_memory_utilization > 1.0` 也直接抛（`:146-147`）。

### Q4 硬件门槛：单机 8 卡行不行？要多少显存？生产拓扑？

- **8 卡能跑**：`offline_inference.py` 默认就是 `--ep 8`，单机 8 卡 → dp=8、world=8、ep=8、cp=1（#16，A）。prefill 侧落在文档的 EP2–8 区间内（#17，B）。
- **8 卡不是高效形态**：decode 官方推荐 EP64–128，最小高效单元 64P+64D（#17，B）；decode 8 卡的对比点是文档自己给的 EP8 vs EP64 图（`doc:55-59`）。
- **非 SM100 的行为**：在 `LLMEngine.__init__` 就抛 `RuntimeError: unknown GPU type`（#15），不会走到 kernel；Ascend 走 432 TFLOPS 分支不受影响。
- **显存要多少**：**未能确认具体数字**（仓库与文档都没有给单卡 HBM 下限，也没给 V4.1-Flash 的参数量；`05_BASELINE_FACTS.md:87` 的 552B 是 D 级二手说法，需以 HF 卡为准）。可给的只有两条硬事实：(a) KV 显存由 `total_gpu_memory × gpu_memory_utilization − profile 峰值 − 2×state` 反推 block 数（#14，A）；(b) 官方口径是"prefill 用 HBM 允许的最小 EP/DP"（`doc:47`），即**卡数是为了装下权重而被 HBM 逼出来的**，不是反过来。工程上可用的估算式：单卡权重占用 ≈ 总参数量 × 每参数字节 / EP + 冗余专家开销，再留出 KV 与激活——但任何具体 GB 数都应由读者在自己卡上跑 `offline_inference.py` 并读那行 `Memory profiling results: ... kv_cache_size=...`（#14）。
- **生产拓扑要求**（B，`doc:45-67`）：P/D/VE 分离；prefill EP2–8/DP2–8、decode EP64–128/DP64–128（Wide-EP）、VE 单卡；不使用 TP/CP；1P+1D 可部署，比例约 1:1。存储与 KV 缓存、跨机 LB 不在开源范围内（`README.md:38-42`）。

### Q5 `benchmark.py` 与 profile 的用法；如何从 trace 看出用了哪些内核

**跑法**（A，`examples/README.md:31-63` + `benchmark.py` argv）：
1. 单机先设 `export WORLD_SIZE=1 RANK=0 CUDA_VISIBLE_DEVICES=0,1,...,7`；`ep_size = WORLD_SIZE × len(CUDA_VISIBLE_DEVICES)`，`--cp` 必须整除 `ep_size`。
2. `python examples/benchmark.py`（默认 `decode_num_requests=64, context_length=32768, prefill_batch_size=32768, non_cached_tokens=4096`）；`PROFILE=0` 关 trace（#18）；`--yoco_clip[=N]` 开 YoCo（裸给 140）；`--result_jsonl F` 落一行原始量；`--tracing_path D` 改 trace 目录（默认 `tracing/`）。
3. 多机：`RANK`/`WORLD_SIZE` 由 `launch_dp` 消费（`examples/_launch.py:15-48`），只有 `dp_rank==0` 写 trace 与 jsonl（`benchmark.py:489,519,570`）。trace 文件名把配置编进去了：`{model}-ep{}-cp{}-prefillbsz{}-noncached{}-input{}-bs{}-nextn{}-force_topk{}-so{}-zos{}-yoco-{mode}{keep}.json`（`benchmark.py:562-570`），事后能凭文件名反查这次是什么配置。
4. 扫点画 Pareto：`bash examples/sweep/run_sweep.sh decode|prefill|encoder <tag> ...`，产物 `sweeps/<tag>/results.jsonl` 与 `<mode>_pareto.png`。

**profile 的两条路**：
- `benchmark.py` 内建（默认开）：prefill/decode 各一段独立 profiler，导出后首尾相接成一条 trace，**固定含 2 个满 chunk prefill step + 3 个稳态 decode step**（`benchmark.py:103-110`；`examples/README.md:47`）。trace 内的时间被 profiler 抬高了近 50%（bsz=1），只能看构成、不能算 SPS（B）。
- 服务态 HTTP：`DEEPINFER_PROFILE_HTTP_ENABLE=1` → `POST http://<host>:6070+driver_gpu_id/profile`，body/query 支持 `num_steps`（默认 20、上限 50）与 `output_dir`（默认 `DEEPINFER_TORCH_PROFILER_DIR=/tmp`），阻塞返回路径（#26，A）。

**从 trace 认内核**（A，机制见 #20；官方也这么说 `doc:225-227`）：
1. 用 Chrome `chrome://tracing` 或 Perfetto 打开 `tracing/*.json`；
2. 先找 CPU 轨上的 `deepinfer:schedule` 切 step，再用 `deepinfer:run_workers` → `launch_forward_N?_B?` → `launch_sample` 定位一次 forward 的时间窗；
3. GPU 轨（`pid` = device index，driver 恒 0）里同一时间窗的 `cat=kernel` 事件即真实 kernel；名字是 mangled 的 C++ 符号，直接搜关键字可归属：`flash_mla`/`FlashMLA`（稀疏注意力）、`deep_gemm`/`mega`（GEMM、Mega MoE）、`deep_ep`/`dispatch`/`combine`（EP 通信）、`tilelang`/`_tl` 前缀（自研 TileLang 算子，如 fused_sample、store_kv_cache）、`nccl`（集合通信）、`elementwise`/`CUDAFunctor`（torch 原生）；
4. 交叉验证：`DEEPINFER_CPP_LOG_LEVEL`、`deepinfer/timer.py` 的 `cpu_timer` 聚合、以及 `PyTorch` 的 `key_averages()` 表（benchmark 里未用，但 `torch.profiler` 对象可直接打）。
- 注意：`examples/README.md` 与 `doc:225-227` 都没给 kernel 名字清单；书中若列名字，必须以读者自己的 trace 为准（E），或引用 R10 的算子清单。

### Q6 没有 SM100 / 昇腾 950 时能读什么、能跑什么（实验设计）

前提：本机**没有任何加速卡**，以下"能跑"分三档——**确定能跑（已实测）**、**需要 CUDA toolkit 但没有卡也能跑（E）**、**只能读**。

| 档 | 实验 | 怎么做 | 学到什么 | 证据 |
|---|---|---|---|---|
| ✅ 已实测 | E1 环境变量表自测 | `python3 -c "import deepinfer; from deepinfer import envs; print(envs.DEEPINFER_USE_NEXTN_DECODE, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS)"`；再改 `DEEPINFER_ACCELERATOR=ascend`、`DEEPINFER_USE_NEXTN_DECODE=3` 重跑 | 派生默认（#11）、缓存的坑（#9）；把附录 B 变成可执行文档 | #21 |
| ✅ 已实测 | E2 `_bool` 陷阱复现 | 用 `deepinfer.environ._bool` 试 `"0"/""/"false"/"False"/"off"/"no"/"FALSE"` | 为什么"设了 off 反而打开" | #10 |
| ✅ 只读 | E3 读构建脚本，画出依赖图 | 对照 §5.4 表，手画 `build.sh → build_vendor.sh → setup.py` 的产物与输入 | 三个 tag（accel/arch/py-buildtag）如何决定 `.so` 路径 | #1–#8 |
| ⚠️ 需 toolkit | E4 只编不跑 | 装 CUDA 13 toolkit（不需要驱动/显卡），`DEEPINFER_ACCELERATOR=nvidia bash scripts/build.sh build`；链接走 `lib64/stubs/libcuda.so` | `csrc/` 与 pybind 边界；`import deepinfer._C` 后可用 `dir()` 看暴露的 C++ 类（`Sequence`/`Scheduler`/`FlopsEstimator`） | #5（A）+ E |
| ⚠️ 需 toolkit | E5 跑 CPU 侧单测 | `pytest tests/engine/test_sequence.py tests/engine/test_flops_estimator.py`（后者需模型 config.json） | C++ `Sequence`/block 语义、FLOPs 估算器的口径 | #23（A/E） |
| ⚠️ 需一张任意 CUDA 卡 | E6 `yoco_equiv.py` L1（`mask-compact`） | `python examples/yoco/yoco_equiv.py mask-compact --keep-tokens 140 --window 128`；**不加载权重**，但要一张可见 GPU（`prepare_prompt_v4` 走 pinned 内存，`yoco_equiv.py:376-377`） | mask 与 compact 两种截断的 SWA pivot/可见 key/store slot/压缩 key range 完全一致 | `examples/yoco/README.md:7`（B）+ `yoco_equiv.py:376-377`（A） |
| ❌ 需 8 卡 + 权重 | E7 `yoco_equiv.py` L2 | `examples/yoco/run_yoco_equiv_l2.sh --model_name /path/to/v4.1-yoco --ep 8 --cp 1 [--mtp 5]` | 逐 token logprob / top-K 对齐；`--mtp` 还能逐位比 confidence（DFlash 主 cache 的探针） | `examples/yoco/README.md:8,29-33`（B） |
| ✅ 只读 | E8 TileLang 算子对照阅读 | 在 `deepinfer/model_executor/layers/tilelang_ops/` 选一个算子（建议 `fused_sample.py` 或 `store_kv_cache.py`），对照 `cuda/` 与 `asc/` 同名文件；用 `ops.py:15-22` 看分发（`*_tl` 后缀 + 符号冲突断言） | 同一算子两套实现的差异点；哪些算子 Ascend 缺失（`KernelUnavailableError`） | #22, R10 |
| ✅ 只读 | E9 与 vLLM 同名文件 diff | 23 个带 `Derived from vLLM` 头的文件（`grep -rl "Derived from vLLM" --include=*.py`），优先 diff 这四个：`deepinfer/engine/arg_utils.py`、`deepinfer/config.py`、`deepinfer/sequence.py`、`deepinfer/entrypoints/llm.py` 对 `vllm/engine/arg_utils.py`、`vllm/config.py`、`vllm/sequence.py`、`vllm/entrypoints/llm.py` | 哪些是"照抄"（SamplingParams 校验）、哪些被重写（Sequence 编码压缩、EngineArgs 的 EP/CP 维度）、哪些被删（TP/PP、分布式 executor） | `R03` §1 第 3 条（vLLM 血统 23 文件/6817 行）+ A |
| ✅ 只读 | E10 读 19 个测试当规格书 | 按 `pytestmark` 分三组：`ci`+`ci_npu`（`test_sequence`、`test_flops_estimator`、`test_state_cache_ops`、`test_shm_broadcast`、`test_offline_inference`）、`gpu`（`test_cuda_sync_guards`）、无标记的模型级（`test_kl`、`test_sampler`、`test_eplb_policy`、`test_batch_invariant`） | 每个测试的 docstring 就是一条设计约束（如 `test_flops_estimator.py:1-6` 说明估算器要按"逐层逐 query 枚举可见 key"对齐） | A |

**不推荐**给无卡读者的：`benchmark.py`（8 卡 + 权重 + 数十 GB trace）、`tests/models/*`（要权重与多卡）、`eval/`（要权重与数据集）。

**推荐代码阅读顺序**（E，但每站都给了可验证的落点）：`README` 特性表 → `examples/offline_inference.py`（看 26 个参数怎么变成一次 `LLM()`）→ `arg_utils.py`/`config.py`（参数派生）→ `deepinfer/engine/llm_engine.py`（`step()`：schedule → run_workers → process_outputs）→ `csrc/engine/scheduler.hpp`/`zos.cpp`（真正的调度）→ `worker/model_runner.py`（input metadata → forward → sample）→ `model_executor/models/v4_modules/*`（层）→ `tilelang_ops/`（算子）。每一站配一个"带着问题读"：见 §5.6。

## 4. 新发现的问题

1. **文档与代码在 `PROFILE` 默认值上不一致**（#18）：`examples/README.md:47` 用 "With `PROFILE=1`" 的措辞，代码默认却是 1。第 16 章应直接写"默认采 trace，`PROFILE=0` 关"。
2. **`gpu_memory_utilization` 的 block 数是全局平均而非最小**（#14 疑点）：多卡异构（或某卡被别的进程占用）时可能出现分配不均，代码没有断言。值得在"部署踩坑"里提一句。
3. **`data_parallel_size` 与 `ep_size` 的关系靠断言维持，错误信息不友好**：`config.py:176-182` 五条 `assert`，任一不满足只会抛裸 `AssertionError`。文档也没写"8 卡应当怎么填 dp/cp/ep"。
4. **开源版缺 TileLang 算子测试与基线数据**（#22）：`conftest.py` 引用的 `benchmark_baselines/`、`gpu_mem_profiles/` 目录不存在，`tile_kernels` 插件的回归阈值（0.15）也无从触发。第 16 章不要承诺"可以跑 kernel 单测"。
5. **`DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` 在 ascend 上被强制为 True，用户无法关**（#11）：`_bool(v) or accelerator=="ascend"`，即使显式设 0 也无效。附录 B 要写"ascend 上只读"。
6. **`SamplingParams.logprobs` 被 `del` 掉**（`sampling_params.py:63`）：传了不报错也不生效，属于 OpenAI 兼容层的空实现。书中讲 API 兼容性时可作为例子。
7. **`tokens_per_block` 只能校验不能设置**（`arg_utils.py:84-86`）：`-1` 表示由模型元数据推导，显式传值只做 `assert` 相等，不能覆盖。容易被误读成"可调 block size"。
8. **`DEEPINFER_CPP_LOG_LEVEL` 不在 `environ.py` 里**（`csrc/utils/log.hpp:21`），是唯一一个绕过 Python 配置层的日志开关；附录 B 若只从 `environ.py` 生成会漏掉它，以及另外 4 个仓内变量。

## 5. 可用于图表的材料

### 5.1 表 B-1 全量 `DEEPINFER_*` 环境变量（64 项）

说明：类型/默认值/出处均取自 `deepinfer/environ.py`（A）；"语义"列全部有实际调用点支撑（A），调用点写在"影响模块"列。布尔语义统一为"只认 `0`/空/`false`/`False` 为假"（#10）。

| 名称 | 类型 | 默认值 | 语义 | 影响模块 | 出处 |
|---|---|---|---|---|---|
| `DEEPINFER_ACCELERATOR` | str | `nvidia` | 加速器后端；决定 `_so` 子目录名、vendor 构建目标 | `deepinfer/__init__.py`、`build.sh`、`build_vendor.sh` | environ.py:101 |
| `DEEPINFER_COMPUTE_SUM_SMALLER_PROBS` | bool | `false` | 采样时是否走 sum-smaller-probs 路径 | `layers/sampler.py:104`、`tilelang_ops/sum_smaller_probs.py` | :104 |
| `DEEPINFER_JIT_CACHE_DIR` | str/None | `None` | JIT 缓存根；设了才改写 tilelang/deep_gemm/deep_ep 三个缓存目录 | `worker/runtime_config.py:45-51` | :105 |
| `DEEPINFER_NCCL_NUM_SMS` | int | `32` | NCCL 进程组的 `max_ctas=min_ctas`（占用 SM 数） | `platform.py:80-83` | :108 |
| `DEEPINFER_USERBUFFER_COMM_SMS` | int | `0` | DeepEP userbuffer dispatch/combine 使用的 SM 数 | `v4_modules/mlp.py:471,504` | :109 |
| `DEEPINFER_GLOO_TIMEOUT` | timedelta | `14400s`（4h） | Gloo 进程组超时 | `worker/base.py:339` | :110 |
| `DEEPINFER_REARRANGE_P2P_GROUPS` | int | `1` | EPLB 重排时跨卡 P2P 的分桶数 | `eplb/shuffle.py:114` | :111 |
| `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | int | `0` | 冗余专家数的下限（引擎按 ep 向上取整到可整除，写回 moe 配置） | `model_executor/model_loader.py:97,122` | :114 |
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | int | `100` | 专家计数滑动窗口步数（越小越省显存） | `eplb/__init__.py:102` | :115 |
| `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | int | `0`（不重排） | 每隔多少 step 做一次专家重排 | `eplb/__init__.py:103` | :116 |
| `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE` | bool | `false` | NORMAL 引擎在 prefill 后立即重排 | `eplb` | :117 |
| `DEEPINFER_MOE_LB_STAT_INTERVAL` | int | `128` | MoE 负载统计的 flush 间隔（step），太短会抖动 | `eplb:125`、`stats_logger.py:456` | :118 |
| `DEEPINFER_FORCE_IMBALANCE_RATIO` | float | `0.0` | profiling 开关：故意制造不均衡（须 ≥1.0） | `v4_modules/mlp.py:367-371` | :119 |
| `DEEPINFER_FORCE_LOAD_BALANCE` | bool | `false` | 强制均匀路由（benchmark 用它压抖动） | `model_metadata_utils.py:166`、`prepare_tensor_input_zos.py:366` | :120 |
| `DEEPINFER_SHOW_EPLB_DETAIL` | bool | `false` | 打印 EPLB 明细 | `eplb/__init__.py:327` | :121 |
| `DEEPINFER_PREFILL_MAX_SUM_SEQLEN_K` | int/None | `None`（C++ 侧封顶 `2**30`） | profile run 的 `sum(seqlen_k)` 上限 | `worker/model_runner.py:406-418` | :124 |
| `DEEPINFER_INDEX_ATTN_MEMORY_LIMIT` | int | `10 GiB` | index-attention 计算分块上限（`min(8192, limit/row/8*8)`） | `v4_modules/index_attn.py:117-118` | :126 |
| `DEEPINFER_USE_NEXTN_DECODE` | int | `5` | 最大投机长度；`0` 关闭投机解码 | MTP/DFlash、`config.py:13`、benchmark | :129 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | bool | `false` | 开启变长验证（动态验证长度） | MTP 调度与 kernel | :131 |
| `DEEPINFER_MAX_BACKTRACKING_TOKENS` | int | `max(设置值, nextn+1)` → `6` | 最大回退 token 数；决定 YoCo 保留窗口下界 | 调度、YoCo | :132 |
| `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` | float/tuple | `1.0` | DFlash 置信度头温度；逗号分隔时元素数须等于 nextn | DFlash | :135-144 |
| `DEEPINFER_ENGRAM_PAD_TOKEN` | int | `2` | Engram 的 pad token id | Engram | :146 |
| `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` | bool | `false`；**ascend 恒 true** | Engram 表放显存（ascend 只读） | `runtime_config.py:18-26` | :147 |
| `DEEPINFER_SAMPLER_FORCE_TOPK` | int | `0`（不强制） | 强制 top-k，绕过动态 top-k；例子用 1024/4096 | `layers/sampler.py:102-104` | :150 |
| `DEEPINFER_DECODE_TOKEN_BLACKLIST` | str | `""` | decode 禁用 token 列表（逗号分隔） | `layers/sampler.py:108-112`、`eval/run_eval.py:133` | :151 |
| `DEEPINFER_ENFORCE_EAGER` | bool | `false` | 全面关闭 CUDA Graph | `worker/model_runner.py` | :154 |
| `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | int | `8` | CUDA Graph 捕获的 batch 粒度 | 同上 | :155 |
| `DEEPINFER_DFLASH_ENFORCE_EAGER` | bool | `false` | 仅 DFlash 头关图 | DFlash | :156 |
| `DEEPINFER_DFLASH_CUDAGRAPH_BATCH_SIZE_INTERVAL` | int | `8` | DFlash 捕获粒度 | DFlash | :157 |
| `DEEPINFER_DECODE_PERF_CACHE_DIR` | str/None | `None` | decode 性能模型的缓存目录 | `worker/decode_perf_cache.py:34` | :158 |
| `DEEPINFER_DECODE_PERF_CACHE_TTL_SEC` | int | `3600` | 上述缓存 TTL | `decode_perf_cache.py:35` | :159 |
| `DEEPINFER_NUM_STATE_SLOTS_RATIO` | float | `1.2` | state slot 数 = `int(max_num_seqs×ratio)+1` | `worker/cache_engine.py:201-202` | :162 |
| `DEEPINFER_PREFILL_STORE_EXTRA_STATE` | bool | `false` | prefill 额外存 state（YoCo/DFlash 复用） | `cache_engine.py:107-110` | :163 |
| `DEEPINFER_PREFILL_STORE_EXTRA_SIZE` | int | `128` | 额外 state 的 token 数 | 同上 | :164 |
| `DEEPINFER_XGRAMMAR_DEBUG` | bool | `false` | xgrammar 调试日志 | `structure_output/` | :167 |
| `DEEPINFER_XGRAMMAR_TIMEOUT` | int | `20` s | 语法/自动机编译超时 | 同上 | :168 |
| `DEEPINFER_XGRAMMAR_DISABLE_CACHE` | bool | `false` | 关闭编译缓存 | 同上 | :169 |
| `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` | int | `10 GiB` | 编译缓存上限 | 同上 | :170 |
| `DEEPINFER_SO_BUILDER_CPUS` | int | `4` | 结构化输出编译进程的 CPU 数 | `structure_output/` | :171 |
| `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` | int | `70` | NORMAL 引擎每 N 个 decode step 做一次 ALL 投票 prefill | `config.py:193` | :174 |
| `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` | int | `100` | ANY 保底投票间隔（防饿死） | `config.py:194` | :177 |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | bool | 由 `EngineType` 决定（NORMAL/COMPLETION→true） | 零开销异步调度 | `llm_engine`、`arg_utils.py:51-55`、`config.py:14` | :179 |
| `DEEPINFER_LOGGING_INTERVAL_STEP` | int | `100` | 每 N step 打一次统计日志；`0` 关闭 | `engine/stats_logger.py` | :182 |
| `DEEPINFER_NCCL_MEMORY_SAMPLE_INTERVAL_STEPS` | int | `1000` | NCCL 显存采样间隔（step） | `parallel_utils/nccl_memory.py:19-21` | :183 |
| `DEEPINFER_RINGBUFFER_WARNING_INTERVAL` | int | `60` s | shm ring buffer 无可用块的告警节流 | `distributed/shm_broadcast.py:313-315` | :184 |
| `DEEPINFER_GC_INTERVAL_STEP` | int | `1000` | gc 被禁用后，每 N step 手动回收（COMPLETION×10） | `worker/base.py:102-106` | :185 |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | enum | `off`（`off`/`mask`/`compact`） | YoCo 截断模式；`mask` 只对齐可见域不省算 | 调度 + TileLang kernel | :188 |
| `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | int | `0`（off 时忽略） | 保留末尾 token 数；V4.1 官方值 `140` = 窗口 128 + 回退 | 同上 | :191 |
| `DEEPINFER_BATCH_INVARIANT` | bool | `false` | 批不变（NVIDIA only；会切确定性算法） | `sampler`、`linear`、`moe`、`runtime_config.py:63-68` | :197 |
| `DEEPINFER_DISABLE_SYNC_GUARD` | bool | `false` | 关闭"禁止意外 CUDA 同步"的守卫 | `deepinfer/utils.py:124-132` | :198 |
| `DEEPINFER_PROFILE_TORCH_CUDA_MEMORY` | str/None | `None` | 指定路径则导 torch 显存快照 | `benchmark.py:225-256` | :200 |
| `DEEPINFER_ENABLE_MEMORY_TRACING` | bool | `false` | profile run 后 dump 显存快照 | `worker/base.py:226-227` | :201 |
| `DEEPINFER_PROFILE_HTTP_ENABLE` | bool | `false` | 开启 `/profile` HTTP 服务 | `engine/profile_server.py` | :202 |
| `DEEPINFER_PROFILE_HTTP_PORT_BASE` | int | `6070` | 端口 = base + driver_gpu_id | `profile_server.py:5-8` | :203 |
| `DEEPINFER_TORCH_PROFILER_DIR` | str | `/tmp` | `/profile` 默认输出目录 | `profile_server.py:208` | :204 |
| `DEEPINFER_USE_MULTIMODAL_POOL` | bool | `false`（`EngineArgs` 按引擎类型反写） | 多模态共享内存池开关 | `ve/multimodal_processor.py:49`、`arg_utils.py:56-60` | :207 |
| `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` | int | `8` | 每 rank 的 pinned host 池大小（GB） | `ve/multimodal_processor.py:51-55` | :208 |
| `DEEPINFER_MULTI_MODAL_BUFFER_SIZE` | int | `512` | 预处理输出队列容量（**条数**，非字节） | `ve/multimodal_processor.py:96` | :209 |
| `DEEPINFER_MULTI_MODAL_WORKER_TIMEOUT` | float | `1800.0` s | 多模态 worker 超时 | `ve/multimodal_processor.py:47` | :210 |
| `DEEPINFER_CPP_LOG_LEVEL` | str | 未设=C++ 默认级别 | C++ 侧日志级别（唯一绕过 Python 配置层的开关） | `csrc/utils/log.hpp:21` | 仓内其它 |
| `DEEPINFER_HF_MODEL_DIR` | str/None | `None` | 只覆盖**默认模型 id** 的本地目录，不覆盖显式 `--model_name` | `examples/_config.py:42`、`README.md:68` | 仓内其它 |
| `DEEPINFER_REPO` | str/None | `None` | eval 脚本定位仓库根 | `eval/common/paths.py:24-35` | 仓内其它 |
| `DEEPINFER_FUSED_SAMPLE_STRESS_TEST` | int | `1` | fused_sample 自测轮数（×1024） | `tilelang_ops/fused_sample.py:530` | 仓内其它 |
| `DEEPINFER_FUSED_SAMPLE_DETAIL_LOG` | str/None | `None` | fused_sample 明细日志路径 | `tilelang_ops/fused_sample.py:601` | 仓内其它 |

统计：64 项 = `environ.py` 59 + 仓内其它 5。按首字母/功能分组的数量见 Q2。

### 5.2 表 B-2 启动参数（`EngineArgs` 26 项 + `LLM` 覆盖 + `SamplingParams`）

| 参数 | 默认值 | 语义 | 备注 |
|---|---|---|---|
| `model` | 必填 | HF 模型 id 或本地目录 | |
| `served_model_name` | `deepinfer-model` | 对外模型名 | |
| `tokenizer` | `None`→同 `model` | 分词器路径 | `__post_init__` 补默认 |
| `trust_remote_code` | `False` | 允许 HF 自定义代码 | |
| `load_format` | `auto` | `auto`=读 safetensors 索引；`dummy`=合成权重（profile 用） | `config.py:115-119` |
| `seed` | `0` | 采样随机种子 | |
| `max_model_len` | `None`→`4096` | prompt+output 总长上限 | `config.py:69` |
| `data_parallel_size` | `0`→`max(1, ep//cp)` | DP 组数 | `config.py:161-163` |
| `context_parallel_size` | `1` | CP 份数（生产不用，代码支持） | |
| `expert_parallel_size` | `1` | EP 组大小（benchmark 里 = 节点数×可见卡数） | |
| `engram_parallel_size` | `0`→`cp×dp` | Engram 并行度 | `config.py:172` |
| `engine_rank` | `0` | 本引擎在 DP 组内的 rank | 例子用 dp_rank 传 |
| `engine_group_rank` | `0` | 引擎组 rank（多组部署用） | |
| `gpu_memory_utilization` | `0.90` | 显存比例上限；KV block 由此反推 | `LLM` 里默认 0.9，例子 0.85，benchmark 0.95 |
| `max_num_batched_tokens` | `None`→`max(max_model_len, 4096, max_num_seqs×(1+nextn))` | 单步 token 上限 | `config.py:219-228` |
| `max_num_seqs` | `256` | 单步最大序列数 | 例子：4（offline）/ 512（测试） |
| `disable_log_stats` | `False`（`LLM` 里 `True`） | 关统计日志 | `entrypoints/llm.py:67` |
| `revision` / `tokenizer_revision` | `None` | HF 版本（branch/tag/commit） | |
| `stop_preempted_request` | `True`（`LLM` 里 `False`） | 被抢占的请求是否终止 | `entrypoints/llm.py:68` |
| `engine_type` | `NORMAL` | `NORMAL`/`PROMPT`/`COMPLETION` | 决定 ZOS 与多模态池默认值 |
| `chunked_prefill` | `False` | 分块 prefill | 关时要求 `batched_tokens ≥ max_model_len` |
| `ignore_len` | `128<<20` | 超长请求的忽略阈值（默认接受所有长度） | `arg_utils.py:43` |
| `tokens_per_block` | `-1` | 只能校验不能设置（须等于模型推导值） | `arg_utils.py:84-86` |
| `lookahead_tokens` | `0` | KV 预留的前瞻 token 数（投机用） | 例子 `min(1,nextn)` |
| `offline_inference` | `False`（`LLM` 强制 `True`） | 离线模式（无 HTTP 前端） | `entrypoints/llm.py:79` |
| `decode_sequence`（仅 `LLM`） | `True` | 是否边解码边输出文本 | 例子传 `False` |
| `SamplingParams.temperature` | `1.0` | `<1e-5` 时被抬到 `1e-5` 并强制 `top_p=1,top_k=-1` | `sampling_params.py:76-79` |
| `SamplingParams.top_p` / `top_k` | `1.0` / `-1` | 核采样 / top-k | `top_k=0` 非法 |
| `SamplingParams.max_tokens` | `16`（`0`→`1`） | 生成长度 | |
| `SamplingParams.seq_seed` | `None`→随机 64bit | 序列级种子（确定性复现用） | |
| `SamplingParams.so_args` | `None` | 结构化输出：`schema_type`、`schema`、可选 `trigger_token_ids`/`bound_token_id` | `sampling_params.py:39-43` |
| `SamplingParams.logprobs` | `False` | **被忽略**（`del logprobs`） | `:63` |

### 5.3 表 B-3 config dataclass 的派生与断言

| 类 | 字段 | 派生/校验 | 出处 |
|---|---|---|---|
| `ModelConfig` | model, tokenizer, trust_remote_code, load_format, seed, max_num_seqs, revision, tokenizer_revision, max_model_len, max_num_batched_tokens, ignore_len, served_model_name, engine_type | `max_model_len=None→4096`；`dtype` 硬编 `bfloat16`；`runtime_max_model_len = max_model_len + nextn + (nextn+1 if ZOS) + nextn`；`load_format ∈ {auto,dummy}` | `config.py:43-127` |
| `CacheConfig` | block_size, gpu_memory_utilization, lookahead_tokens=0 | `gpu_memory_utilization ≤ 1.0`；`num_gpu_blocks/num_state_seqs/bytes_per_block` 在 profile 后填 | `config.py:125-148` |
| `ParallelConfig` | dp, cp, ep, engram_ps, engine_rank, engine_group_rank, engine_type | `dp==0→max(1, ep//cp)`；`model_parallel_size=cp`；`world_size=max(cp×dp, engram_ps)`；5 条断言（含 `ep>0`、`cp×dp % ep == 0`） | `config.py:150-182` |
| `SchedulerConfig` | max_num_batched_tokens, max_num_seqs, max_model_len, stop_preempted_request=False, chunked_prefill=False, prefill_attempt_interval=70, prefill_any_attempt_interval=100 | 两个 interval 的默认值在**函数定义时**就从 `envs` 求值（之后再改环境变量无效）；关 chunked prefill 时要求 `batched ≥ max_model_len` | `config.py:185-216` |

### 5.4 表 B-4 构建流水线（可直接画成第 16 章的流程图）

```
build.sh
 ├─ 0 校验 DEEPINFER_ACCELERATOR ∈ {nvidia, ascend}
 ├─ 1 pip install -r requirements-<accel>.txt      [PyPI + download.pytorch.org]
 ├─ 2 (nvidia) pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
 ├─ 3 scripts/build_vendor.sh
 │    ├─ git submodule update --init (6)
 │    ├─ for pkg in (DeepSelect|FlashMLA|DeepEP|DeepGEMM[-Ascend]):
 │    │     git submodule update --init --recursive; python setup.py bdist_wheel
 │    └─ pip install --no-deps --no-index --find-links build/<tag>/vendor-wheels <4 包>
 └─ 4 scripts/build.sh build
      ├─ MAX_JOBS=${MAX_JOBS:-16}
      ├─ python setup.py build --build-base build/<accel>-<arch>
      ├─ ln -fs build/<tag>/lib.<py-buildtag>/deepinfer/*.so deepinfer/_so/<tag>/
      ├─ [ -n "$(ls -A deepinfer/_so/<tag>)" ] || exit 1
      └─ python -m compileall deepinfer -j $MAX_JOBS

运行期: deepinfer/__init__.py 用同一 <accel>-<arch> 标签把 deepinfer/_so/<tag> 挂进 __path__
```

| 产物 | 位置 | 谁消费 |
|---|---|---|
| vendor wheels | `build/<tag>/vendor-wheels/*.whl` | 第 3 步的 pip（`--no-index`） |
| C++ 扩展 | `build/<tag>/lib.<py-tag>/deepinfer/_C*.so` | 软链到 `deepinfer/_so/<tag>/` |
| 软链 | `deepinfer/_so/<accel>-<arch>/` | `import deepinfer._C` |
| pyi stub | 仓库根（`make stubgen` → `scripts/gen_stub.sh`） | 类型检查/IDE，非运行必需 |

### 5.5 表 B-5 硬件门槛（"能跑" vs "高效"）

| 维度 | 能跑（开源例子默认） | 高效（官方生产口径） | 出处 |
|---|---|---|---|
| 加速器 | NVIDIA SM100 / Ascend 950 | 同 | `README.md:33-36`（B） |
| 单机卡数 | 8（`--ep 8`） | prefill 机 ≤8 卡；decode 需 8–16 个 8 卡节点（EP64–128） | A #16 / B #17 |
| prefill 并行 | EP8 / DP8（单机满卡） | EP2–8 / DP2–8（HBM 允许的最小） | `doc:47` |
| decode 并行 | EP8（同左） | EP64–128 / DP64–128（Wide-EP） | `doc:49` |
| VE | 离线例子用 `--prompt_mode image` 在同一脚本里跑多模态 | 单卡独立部署，用便宜卡 | `doc:51` / `benchmark.py:53` |
| TP / CP | CP 可设（`--cp`）；代码里没有 TP 参数（`model_parallel_size = cp`） | "暂不使用" | `doc:53` / `arg_utils.py:29` / `config.py:171` |
| 最小完整服务 | 1P + 1D（可部署，不高效） | 64P + 64D | `doc:65-67` |
| 单卡算力常量（BF16 dense） | SM100 x86 `2250` / aarch64 `2500` TFLOPS；Ascend `432` | 同 | `device.py:152-158`（A） |
| 单卡 HBM 下限 | **未能确认**（仓库/文档无数字） | 由"最小 EP/DP that HBM allows"反推 | Q4 |

### 5.6 表 B-6 无卡学习路线（每站带一个问题）

| 站 | 读什么 | 带着什么问题 | 可验证的落点 |
|---|---|---|---|
| 1 | `README.md` + `doc/inference_deployment_qa.md` | 哪些是"开源了"，哪些是"只在文档里" | `README.md:38-42` 列的"not included"三项 |
| 2 | `examples/offline_inference.py` | 一次最小推理要设几个东西？ | `--ep 8` 默认、`BATCH_INVARIANT=1`、`max_num_seqs=4` |
| 3 | `arg_utils.py` / `config.py` | 哪些参数是派生的？哪些会启动即失败？ | 表 B-3 的四条派生、两条 ValueError |
| 4 | `engine/llm_engine.py` 的 `step()` | 一步里 CPU 做什么、GPU 做什么、谁在等谁 | `cpu_timer.record("schedule"/"run_workers"/"bubble")` |
| 5 | `csrc/engine/scheduler.hpp` + `zos.cpp` | 调度为什么在 C++？ZOS 的"晚一步"在哪体现 | `llm_engine.py:826-827`（ZOS 注释） |
| 6 | `worker/model_runner.py` | input metadata 怎么变成 kernel 参数 | `prepare_input`、`launch_forward_N?_B?` |
| 7 | `models/v4_modules/{attn,mlp,index_attn}.py` | 哪些算子来自 vendor、哪些是 TileLang | #22 的 `*_tl` 工厂清单 |
| 8 | `tilelang_ops/{cuda,asc}/<op>.py` 对照 | 同一算子在两平台差在哪 | `ops.py` 的 `KernelUnavailableError` |
| 9 | 23 个 `Derived from vLLM` 文件 diff | 哪些是照抄、哪些被重写 | `sequence.py` 只剩薄壳（`Sequence` 在 `csrc/engine/`）、`config.py` 多了 EP/CP 维度 |
| 10 | `tests/` 19 个文件当规格书 | 每个测试在守护哪条不变量 | `test_flops_estimator.py:1-6` 的"逐 query 枚举可见 key" |

---

### 附：本文用到的实测命令（可复现）

```bash
cd <repo>
python3 -c "import deepinfer; from deepinfer import envs; \
print(envs.DEEPINFER_ACCELERATOR, envs.DEEPINFER_USE_NEXTN_DECODE, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS)"
# nvidia 5 6

DEEPINFER_ACCELERATOR=ascend DEEPINFER_USE_NEXTN_DECODE=3 python3 -c "from deepinfer import envs; \
print(envs.DEEPINFER_ENGRAM_FORCE_GPU_STORAGE, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS)"
# True 4

python3 -c "from deepinfer.environ import _bool; print([ _bool(v) for v in ['0','','false','False','off','no','FALSE'] ])"
# [False, False, False, False, True, True, True]
```
