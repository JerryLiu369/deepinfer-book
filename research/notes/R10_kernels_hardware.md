# R10 底层内核库、TileLang 算子分发、双硬件（NVIDIA / 昇腾）适配与低精度格式

## 0. 元信息

- 代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21 12:55 +0800）。本文所有 `路径:行号` 均指该 commit。
- vendor submodule 版本（本 commit 锁定的 gitlink）：
  - `vendor/DeepEP` `f6aa38f`（`v1.2.1-43-gf6aa38f`，2026-10-21）
  - `vendor/DeepEP-Ascend` `3b25377`（main，2026-09-29）
  - `vendor/DeepGEMM` `1fc9c66`（`v2.1.1.post3-25-g1fc9c66`，2026-10-14）
  - `vendor/DeepGEMM-Ascend` `3a35326`（main，2026-10-14）
  - `vendor/DeepSelect` `bfa4507`（main，2026-09-30）
  - `vendor/FlashMLA` `2e5429f`（2026-09-30）
- 查阅的主要来源：
  - `vendor/{DeepEP,DeepEP-Ascend,DeepGEMM,DeepGEMM-Ascend,DeepSelect,FlashMLA}/README.md` 与前三者的 `csrc/`、`deep_gemm/mega/`
  - `deepinfer/platform.py`、`deepinfer/__init__.py`、`deepinfer/worker/runtime_config.py`、`deepinfer/worker/cache_engine.py`
  - `deepinfer/model_executor/models/v4_modules/{attn.py,mlp.py,index_attn.py,hc.py,engram.py}`
  - `deepinfer/model_executor/layers/{attention_backend.py,linear.py}`
  - `deepinfer/model_executor/layers/tilelang_ops/`（根目录、`cuda/`、`asc/`、`ops.py`、`cache_common.py`、`conftest.py`）
  - `deepinfer/model_executor/model_metadata_utils.py`、`deepinfer/hugging_face/{config.py,weights.py}`
  - `deepinfer/npu_ops/{__init__.py,patch_imports.py,deep_select.py,vit_attn.py}`
  - `csrc/{pybind.cpp,cuda_utils.cpp,asc_utils.cpp}`、`setup.py`、`scripts/build_vendor.sh`、`scripts/build.sh`
  - `doc/inference_deployment_qa.md`、`README.md`、`requirements*.txt`
  - `vendor/FlashMLA/docs/20260930-ascend-prefill-deep-dive.md`
- 说明：本机无 GPU/NPU，未运行任何 kernel 或 benchmark；所有性能数字均为官方文档引用，代码执行顺序均由静态阅读得出。正文中 `doc:NNN` 均指 `doc/inference_deployment_qa.md` 的行号。

## 1. 三个最重要的发现

1. **DeepInfer 的算子面由 5 个外部库 + 1 套自研 TileLang 算子拼成，调用面主要落在 `v4_modules`（engram/CP 的原语在并行工具层）**：FlashMLA 负责稀疏注意力，DeepGEMM 负责 GEMM/MQA logits/Mega 系列，DeepEP 负责 EP all-to-all，TileKernels 负责门控/SwiGLU/mHC/Engram，DeepSelect 负责 TopK；`tilelang_ops/` 补齐其余 43 个 `*_tl` 工厂。
2. **"Mega MoE" 是 DeepGEMM 的单内核融合（dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine，通信与张量核重叠）**，DeepInfer 只在 EP 组位于 scale-up 域内时启用（`mlp.py:184-188`），域外回退到 DeepEP + grouped GEMM 的常规路径；官方文档"Mega MoE 在 scale-up 域内总是更快"（`doc:231`）没有给任何数字与配置。
3. **双硬件差异被压到很薄的一层**：同一套 Python 调用、同名包（`deep_ep`/`deep_gemm`/`flash_mla`/`deep_select`）由 `scripts/build_vendor.sh` 在构建期二选一；TileLang 算子层 `cuda/` 与 `asc/` 各有 43 个同名 `*_tl` 工厂，由 `ops.py` 按 `PLATFORM.tilelang_ops_backend` 纯名字级重导出，低精度 KV 布局则单点定义在 `cache_common.py`（288 B / 528 B / 68 B）。

## 2. 事实条目

### #1 官方 README 明确列出 5 个外部内核库 + 自研 TileLang 算子

- 证据：`README.md:23`
  > `**Optimized kernels:** [FlashMLA](...), [DeepGEMM](...), [DeepEP](...), [TileKernels](...), [DeepSelect](...), and custom [TileLang kernels](deepinfer/model_executor/layers/tilelang_ops).`
- 等级：B
- 意味着什么：这 6 项就是全书讲"内核"时的完整边界，没有第七个隐藏库。
- 疑点：无

### #2 通信与 GEMM 两个库按加速器二选一，FlashMLA/DeepSelect 同仓按构建目标编两版，且 Ascend 版复用同一 Python 包名

- 证据：`scripts/build_vendor.sh:7-22`
  > `nvidia) BUILD_TARGET=CUDA; PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM); INSTALL=(deep_select flash_mla deep_ep deep_gemm)`
  > `ascend) BUILD_TARGET=ASCEND; PACKAGES=(DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend); INSTALL=(deep_select flash_mla deep_ep deep_gemm)`
- 证据：`vendor/DeepGEMM-Ascend/README.md:9`（"DeepGEMM Ascend 和 DeepGEMM 使用相同的包名"）；`vendor/DeepEP-Ascend/README.md:62`（"share the same `EPBuffer` interface and the `deep_ep` Python package name"）
- 等级：A（脚本）+ B（README）
- 意味着什么：上层 `import deep_gemm` / `from deep_ep import EPBuffer` 完全不需要平台分支，平台差异在 wheel 安装时已经决定。
- 疑点：`DeepGEMM-Ascend` 的 Python 包名虽同，但导出符号并非全集（见 #12、Q4 表）。

### #3 FlashMLA 的职责：DSA 稀疏 prefill/decode + 融合 norm-RoPE-attn-RoPE-cast

- 证据：`vendor/FlashMLA/README.md:7`
  > `This repo contains token-level sparse attention kernels for prefill and for decoding with an FP8 / FP4 KV cache, a fused kernel that combines Q-norm, Q-RoPE, attention, O-RoPE (conjugate) and the cast to FP8, and dense attention kernels for prefill and backward.`
- 证据：`deepinfer/model_executor/layers/attention_backend.py:2-3`
  > `from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, fused_norm_rope_attn_rope_cast`
  > `from flash_mla.flash_mla_interface import FlashMLASchedMeta`
- 等级：A
- 意味着什么：DeepInfer 只用 FlashMLA 的稀疏 DSA 路径 + 融合 kernel；dense MHA 只用于 ViT（且走 `flash_attn` 包，见 #20）。
- 疑点：无

### #4 FlashMLA 在模型中的三个调用点：prefill、decode、DFlash 草稿 decode

- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:617-625`（prefill）
  > `context, _ = prefill_attn_varlen_sparse(q_b, kv, kv_index, self.softmax_scale, attn_sink=self.attn_sink_params, index_length=dense_index_len, fused_norm_rope_cast_kwargs=fused_kwargs)`
- 证据：`attn.py:660-669`（主模型 decode）；`attn.py:994-1007`（DFlash/MTP 草稿 decode）
- 证据：`attention_backend.py:64-71`（`flash_mla_sparse_fwd`）、`:125-142`（`flash_mla_with_kvcache`，`is_fp8_kvcache=True`、`indices=` 走 token 级稀疏）
- 等级：A
- 意味着什么：每层每次前向最多一次 FlashMLA 调用；prefill 与 decode 走不同 API，DFlash 复用 decode API 但 KV 只读 SWA cache。
- 疑点：`is_fp8_kvcache=True` 是硬编码（`attention_backend.py:134`），FP4 主 KV 也走同一标志；实际格式由 `k_cache` 最后一维字节数判定（见 #6），标志名与语义不完全对应。

### #5 融合 kernel `fused_norm_rope_attn_rope_cast` 仅 NVIDIA 启用

- 证据：`attn.py:87-92`
  > `self.use_fused_norm_rope_attn_rope_cast = (((not envs.DEEPINFER_BATCH_INVARIANT and cfg.engine_type == EngineType.PROMPT) or self.is_sparse or not self.has_packed_kv) and self.num_heads in (64, 128) and PLATFORM.is_gpu)`
- 证据：`vendor/FlashMLA/README.md:43`
  > `This kernel currently supports CUDA only; Ascend is not supported.`
- 等级：A + B
- 意味着什么：昇腾上 Q-norm/Q-RoPE/O-RoPE/FP8 cast 是分开的小算子，这本身构成一个可量化的平台差异。
- 疑点：无

### #6 FlashMLA 的 KV cache 是"按 token 自描述"的 528 B / 288 B 两种格式

- 证据：`vendor/FlashMLA/README.md:138-141`
  > `the format is detected from the last dimension of k_cache (i.e. the bytes per token): 528 (V4.1) or 288 (V4.1 fp4).`
  > `V4.1 (528 bytes per token): the first 512 bytes are float8_e4m3 ... The last 16 bytes are float8_e8m0 scales, one for every 32 ...`
  > `V4.1 fp4 (288 bytes per token): the first 256 bytes contain 512 e2m1 values ... The last 32 bytes are float8_e4m3 scales, one for every 16 ... This format is only valid for extra_k_cache when k_cache is in the V4.1 format.`
- 证据：`attn.py:640-647`（SWA 层把 `swa_cache` 当主 `key_cache`，把压缩主 KV `ca_cache` 当 `key_cache_extra`）
- 等级：B（格式）+ A（角色分工）
- 意味着什么：注意两套命名错位——在 FlashMLA 调用里 528 B 是 `k_cache`、288 B 是 `extra_k_cache`；在 DeepInfer 命名里 288 B 的 `ca_cache` 才是"main（compress）KV"，528 B 是 SWA KV。这不是随意选择，而是 FlashMLA 的接口约束：FP4 只能作为 `extra_k_cache` 出现。
- 疑点：若某层 `use_swa=False` 且 `ratio>0`，`ca_cache`（288 B）会被当主 `key_cache` 传入，与 README 的约束冲突；V4.1 所有层都有 SWA，因此该路径在当前模型不触发（`attn.py:93-96` 断言 `use_swa or has_packed_kv`）。

### #7 DeepGEMM 的职责：dense/grouped GEMM、indexer MQA logits、Mega 系列

- 证据：`vendor/DeepGEMM/README.md:3`
  > `DeepGEMM is a unified, high-performance tensor core kernel library that brings together ... GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more`
- 证据（调用点）：
  - `deepinfer/model_executor/layers/linear.py:165`：`deep_gemm.fp8_fp4_gemm_nt(x, weight, out, out_acc, recipe=(1, 1, get_fp8_casting_granularity()))`（所有 FP8 线性层）
  - `v4_modules/mlp.py:119` / `:130`：`m_grouped_fp8_fp4_gemm_nt_contiguous` / `m_grouped_bf16_gemm_nt_contiguous`（MoE 专家 GEMM）
  - `v4_modules/index_attn.py:299`（prefill `fp8_fp4_mqa_logits`）、`:317`（sparse prefill）、`:434`（paged decode）、`:447`（paged sparse decode）
  - `v4_modules/mlp.py:549`（`self.mega_moe_func`，见 #11）、`:398`（`bf16_mega_gate`）
  - `v4_modules/hc.py:270`（`mega_mhc`，仅 GPU 启用，见 Q4）
- 等级：A
- 意味着什么：DeepGEMM 不只是"矩阵乘库"，它同时承担 indexer 的打分（MQA logits）与 MoE/HC 的融合算子，是覆盖面最广的一个库。
- 疑点：`linear.py:165` 的 `recipe=(1, 1, 32)` 把激活按 per-32 量化写死，与 `get_fp8_casting_granularity()` 返回 32 一致（`linear.py:13-14`）。

### #8 DeepEP 的职责：EP dispatch/combine，另含 Engram/PP/Bucket 原语

- 证据：`vendor/DeepEP/README.md:3`
  > `DeepEP (DeepEveryParallel) is a high-performance communication library ... It provides high-throughput and low-latency expert-parallel (EP) all-to-all GPU kernels for MoE dispatch and combine, including FP8 dispatch, and NVLink weight and gradient exchange ...`
- 证据（调用点）：`v4_modules/mlp.py:469`（`self.ep_buffer.dispatch(...)`）、`:501`（`self.ep_buffer.combine(...)`）；`v4_modules/engram.py:181`（`userbuffer.allocate_engram`）；`parallel_utils/cp.py:17`（`BucketBuffer`）
- 等级：A + B
- 意味着什么：MoE 的"通信"与"计算"在非 Mega 路径上是两次独立的库调用，中间由 `EventOverlap` 与计算流重叠（`mlp.py:578-610`）。
- 疑点：`DeepEP/README.md:39` 明确 "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported"，即通信要占 SM。

### #9 TileKernels 不是 submodule，而是 pip 依赖 `tile-kernels==2.1.0`

- 证据：`requirements.txt:14`
  > `tile-kernels==2.1.0`
- 证据（用途）：`v4_modules/mlp.py:18-19`
  > `from tile_kernels.moe import moe_topk_gate`
  > `from tile_kernels.quant import swiglu_forward_and_per_token_cast`
- 证据（调用点）：`mlp.py:400`（`moe_topk_gate`）、`:436`/`:521`（`swiglu_forward_and_per_token_cast`）、`attn.py:200`（`tile_kernels.quant.per_token_cast`）、`v4_modules/mhc_ops.py:8-14`（7 个 `tile_kernels.mhc.*` kernel）、`v4_modules/engram.py:134/151`、`model_executor/models/deepseek.py:180`（`engram_hash`）
- 等级：A
- 意味着什么：TileKernels 是"小而杂"的算子集合（门控、SwiGLU+量化、mHC、Engram、通用 quant），它与 `tilelang_ops/` 的分界是"是否与模型结构强绑定"。
- 疑点：仓库内没有 `vendor/TileKernels`，无法在本 commit 内核对实现；只能确认调用面。

### #10 DeepSelect 的职责：indexer 与 sampler 的 TopK

- 证据：`vendor/DeepSelect/README.md:3`
  > `DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) ... and the sampler. ... achieves 2 ~ 20x speedup compared to vanilla torch.topk.`
- 证据（调用点）：`v4_modules/index_attn.py:12`（`from deep_select.interface import topk`）、`:216`（candidate block topk）、`:361`、`:494`（512-token topk，`return_value=False`）；`layers/sampler.py:12`、`layers/sampler_utils.py:4`
- 等级：A + B
- 意味着什么：DSA 的"选 512 个 token"与采样的"vocab topk"共用一个库，但两条路径的 dtype/约束不同。
- 疑点：`vendor/DeepSelect/README.md:37` 说 Ascend 只支持 bf16、fp32 sampling 仅 CUDA；DeepInfer 的 NPU 兜底见 #20。

### #11 Mega MoE 是什么：把 MoE 的五步融成一个 persistent kernel

- 证据：`vendor/DeepGEMM/README.md:114`
  > `Mega MoE fuses and overlaps EP dispatch, linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory.`
- 证据：`vendor/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（`BlockPhase` 枚举含 `Linear1/Linear2/SharedLinear1/SharedLinear2`，任务以 `pool_block_idx`、`m_block_idx`、`n_cluster_idx` 在核内调度）；`vendor/DeepGEMM/csrc/apis/mega_moe.hpp`（`get_symm_buffer_size_for_mega_moe`、ring/pool 容量计算）
- 等级：B + A（vendor 源码）
- 意味着什么：Mega MoE 不是"更快的 grouped GEMM"，而是把通信、两段 GEMM、激活、combine 放进同一个 kernel 的流水线，消除了 kernel 边界与全局 barrier。
- 疑点：名字里的 "Mega" 也用于 `bf16_mega_gate`、`mega_mhc`，三者是同一"融合多个小算子"设计思路的不同算子，不是同一个 kernel。

### #12 DeepInfer 启用 Mega MoE 的条件：域内 + 平台/MMA 类型

- 证据：`v4_modules/mlp.py:184-188`
  > `self.use_mega_moe = ((PLATFORM.is_gpu or self.mma_type == "fp8xfp4") and self.mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16") and PG.is_within_nvl_domain(PG.expert_parallel_group()))`
- 证据：`parallel_utils/parallel_state.py:131-134`
  > `def is_within_nvl_domain(group: ProcessGroup) -> bool:`
  > `    if PLATFORM.is_npu:`
  > `        return True`
  > `    return group.size() == 1 or group.size() <= deep_ep.get_physical_domain_size(group)[1]`
- 等级：A
- 意味着什么："scale-up 域"在代码里就是"EP 组不超过 `deep_ep` 报的物理域大小"；NPU 上该判断恒真，但 NPU 只允许 `fp8xfp4` 走 Mega（因为 DeepGEMM-Ascend 只导出 `fp8_fp4_mega_moe`）。
- 疑点：`PG.is_within_nvl_domain` 在 NPU 上直接 `return True`，名字里的 NVLink 在昇腾上不对应任何硬件；这是命名遗留。

### #13 Mega MoE 的缓冲区与权重布局在模型加载期准备

- 证据：`v4_modules/mlp.py:93-98`（`transform_weights_for_mega_moe`，L1 gate/up 行交织 + SF 转置）
- 证据：`mlp.py:303-315`（`userbuffer.allocate_sym_buffer(...)`）；`userbuffer.py:221`（`deep_gemm.SymmBuffer(**spec._asdict(), base=base)`）
- 证据：`mlp.py:549-559`（`self.mega_moe_func(out, l1_weights, l2_weights, sym_buffer, **shared_weights_kwargs, cumulative_local_expert_recv_stats=..., activation=..., activation_clamp=..., fast_math=True)`）
- 等级：A
- 意味着什么：Mega MoE 的代价前移到 load 期（权重重排、对称内存分配），decode 步内只剩一次 kernel 调用。
- 疑点：`transform_weights_for_mega_moe` 要求 `mn % 128 == 0`（`vendor/DeepGEMM/docs/scaling-factor-format.md:399`），因此并非任意模型形状都能开启。

### #14 文档"Mega MoE 在 scale-up 域内总是更快"的原文

- 证据：`doc/inference_deployment_qa.md:229-231`
  > `- When is Mega MoE used?`
  > `    - Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE.`
- 等级：B
- 意味着什么：官方把 Mega MoE 定位为域内默认解——是否启用不按 EP 规模或 batch 做启发式判断（block 大小仍由 `get_block_m_for_mega_moe` 按形状选择，`mlp.py:293-300`）。
- 疑点：没有给出"快多少"、对比基线（DeepEP+grouped GEMM？）、硬件型号、EP 规模、token 数；"always faster" 是定性断言。代码侧唯一能核对的是启用条件（#12），无法核对性能结论。

### #15 `tilelang_ops/` 是三层结构：根目录 API/参考/测试 + 两个后端实现 + 一个名字级分发器

- 证据（统计，`find`/`wc` 于本 commit 目录）
  - 根目录：33 个 `.py`，其中 29 个是"公开 API + PyTorch 参考实现 + pytest 测试"合一模块，4 个是基础设施（`ops.py`、`common.py`、`cache_common.py`、`conftest.py`）；根目录合计 10,945 行。
  - `cuda/`：30 个 `.py`、4,968 行、43 个 `*_tl` 工厂。
  - `asc/`：30 个 `.py`、7,418 行、43 个 `*_tl` 工厂。
  - 27 个文件名在三个目录完全同名；`prepare_v4_decode_metadata.py`（根/asc）与 `prepare_decode_metadata.py`（cuda）文件不同名但导出同名工厂。
- 证据：`tilelang_ops/ops.py:1-6`（模块 docstring）
  > `Kernel dispatcher: interface modules do ``from deepinfer.model_executor.layers.tilelang_ops import ops`` and call ``ops.xxx_tl``; backend selection happens only in this file.`
- 等级：A
- 意味着什么：根目录不是"再包一层薄壳"，它同时是规范（参考实现）和验收（测试）；后端只负责"同一语义的高性能实现"。
- 疑点：根目录的 `compress.py`/`norm.py` 仍 `import tilelang`（仅用于 `PassConfigKey` 与类型），并非 100% 后端无关。

### #16 分发机制：`ops` 动态导入所选后端并重导出所有 `*_tl`

- 证据：`tilelang_ops/ops.py:13-22`
  > `from deepinfer import PLATFORM`
  > `_pkg = importlib.import_module(f"{__package__}.{PLATFORM.tilelang_ops_backend}")`
  > `for _info in pkgutil.iter_modules(_pkg.__path__): ...`
  > `if _name.startswith("_") or isinstance(_obj, ModuleType) or not _name.endswith("_tl"): continue`
  > `assert globals().get(_name, _obj) is _obj, f"backend symbol collision: {_name}"`
- 证据：`ops.py:25-32`（`KernelUnavailableError(AttributeError)` + 模块级 `__getattr__`）
- 证据：`tilelang_ops/conftest.py:100-113`
  > `"""Skip tests hitting kernels the selected backend has not implemented."""`
  > `if type(e).__name__ != "KernelUnavailableError" or accelerator_kind() == "nvidia": raise`
  > `pytest.skip(str(e))`
- 等级：A
- 意味着什么：选择发生在 import 时、按符号名而非文件名；缺算子不会在 import 期炸掉，而是调用时抛 `AttributeError` 子类，测试自动 skip。
- 疑点：同名符号冲突用 `assert` 处理，若两个后端模块导出同名但语义不同的工厂，只有运行时断言能发现。

### #17 根目录模块被生产代码直接 import，不存在"另一套运行时封装"

- 证据：`v4_modules/attn.py:39`（`from ...tilelang_ops.store_kv_cache import store_kv_cache`）；`v4_modules/compress.py:16`；`layers/layernorm.py:10`（`norm`）；`layers/sampler.py:25`；`input_metadata/input_metadata.py:14`
- 证据：`tilelang_ops/store_kv_cache.py:63` 与 `:77-79`
  > `ops.store_kv_cache_tl(layout, kv_cache.shape[1])(kv, cache, slot_mapping)`
  > `store_kernel = ops.store_kv_cache_tl(layout, block_size, **kernel_kwargs)`
- 等级：A
- 意味着什么：读算子实现只需跟"根目录函数 → `ops.xxx_tl` → cuda/asc 同名工厂"三跳。
- 疑点：根目录文件混放 API 与测试（29/33 含 `def test_`），生产 import 会连带加载 `pytest`（`store_kv_cache.py:4`），是一个轻微的启动开销。

### #18 平台层：后端名、通信后端、SF dtype 三个常量决定大部分差异

- 证据：`deepinfer/platform.py:55-63`（GPU：`tilelang_ops_backend="cuda"`、`distributed_backend="nccl"`、`fp8_sf_type=torch.int32`）
- 证据：`platform.py:86-89`、`:138`（NPU：`"asc"`、`"hccl"`、`fp8_sf_type=torch.int16`）
- 证据：`platform.py:226`（`platform_class = PlatformNPU if accelerator_kind() == "ascend" else PlatformGPU`）
- 等级：A
- 意味着什么：`fp8_sf_type` 是"一个 int 里塞几个 UE8M0"的抽象：GPU 4 个/int32，NPU 2 个/int16；量化 SF buffer 与 GEMM 接口都以它为 dtype。
- 疑点：`linear.py:164` 用 `assert x[1].dtype == weight[1].dtype == PLATFORM.fp8_sf_type` 把这一差异变成硬约束，第三方权重若不按平台打包会直接断言失败。

### #19 构建与运行时：同名扩展、按平台分目录、NPU 才打的 import 补丁

- 证据：`setup.py:38-53`（`IS_NPU` 时用 `NpuExtension`，sources 加 `csrc/asc_utils.cpp`、`graph_timer_asc.cpp`）；`setup.py:59-72`（CUDA 时加 `cuda_utils.cpp`、`graph_timer_cuda.cpp`）
- 证据：`csrc/pybind.cpp:7-11`（无论 CUDA/NPU，子模块都叫 `cuda_utils`）；`csrc/asc_utils.cpp:14`（NPU 实现仍在 `namespace deepinfer::cuda::utils`）
- 证据：`deepinfer/__init__.py:8-29`（`deepinfer/_so/<accelerator>-<arch>/` 加入 `__path__`，同 arch 的 nvidia/ascend `.so` 可共存）
- 证据：`platform.py:140-141`
  > `# TODO(NPU): Intended to be removed.`
  > `import deepinfer.npu_ops  # noqa: F401`
- 等级：A
- 意味着什么：二进制层通过"同 ABI 名、不同目录"共存；Python 层只用 `PLATFORM` 与少量 monkey patch 分流。
- 疑点：`asc_utils.cpp` 保留 `cuda` 命名空间与 `cuda_utils` 子模块名，阅读时容易误判为 CUDA 代码。

### #20 NPU 的 Python 补丁：DeepSelect 回退与 ViT attention 替换

- 证据：`deepinfer/npu_ops/__init__.py:3-12`（`patch_imports({... "deep_select.interface": "topk" ... "flash_attn.cute.interface": "_flash_attn_fwd"})`）
- 证据：`npu_ops/deep_select.py:20-26`
  > `# TODO(NPU): To be removed when topk for sampler is ready`
  > `if kwargs.get("return_value", True): return _torch_topk(*args, **kwargs)`
  > `return _topk(*args, **kwargs)`
- 证据：`npu_ops/vit_attn.py:9-30`（`torch_npu.npu_fusion_attention(..., "TND", ...)`）
- 等级：A
- 意味着什么：昇腾上"DSA 的 TopK"用 DeepSelect，而"采样器的 TopK"退回 `torch.topk`；ViT 的 `flash_attn` 被替换成 `npu_fusion_attention`。
- 疑点：`patch_imports` 只拦截 `from ... import ...`（`patch_imports.py:62-64`），对 `import deep_select` 后属性访问无效；当前调用点都是 from-import，所以能生效。

### #21 TileLang 实现对比：CUDA 用 TMA+PDL，Ascend 用 persistent+UB+SIMD

- 证据（对源码做字符串计数，非精确语义统计；`cuda/` vs `asc/`）
  - `cuda/`：`T.pdl_sync()` 56 处、`TL_DISABLE_WARP_SPECIALIZED` 29 处、`TMA` 29 处、`T.alloc_shared` 37 处。
  - `asc/`：`tilelang.ascend` 36 处、`T.SimtVF` 43 处、`annotate_buffer_versions`（多版本 UB 缓冲）41 处、`T.Persistent` 53 处、`S.v*` 向量内建 623 处、`tile_kernels.get_num_vec_cores()` 50 处。
- 证据（同一算子两版实现）：`cuda/flash_mla_utils.py:27-38`（`T.Kernel(s, b, threads=128)` + `T.pdl_sync()` + fragment 寄存器内重映射） vs `asc/flash_mla_utils.py:20-51`（`k` 必须是 2 的幂且 ≤4096；`T.Kernel(num_cores)` 持久化；`alloc_shared` + `annotate_buffer_versions(num_stages=4)`；`T.SimtVF(threads=k)`）
- 等级：A
- 意味着什么：CUDA 版把"并行"交给 block/warp 与 TMA，Ascend 版把"并行"交给固定数量的 vector core + UB 多版本缓冲 + 手写 SIMD；代码量更大（7,418 vs 4,968 行）与这一范式差异一致。
- 疑点：`asc/` 的 `k <= 4096` 是硬约束，CUDA 版没有；若未来 topk > 4096，两平台行为会分叉。

### #22 Ascend 的数值补偿：`rstd_1ulp` 与寄存器量化 LUT

- 证据：`asc/rstd_1ulp.py:1-10`
  > `"""Compensated RSTD, matching hai_llm.tilelang_kernels.ascend.helper.rstd_1ulp.`
  > `Shared by norm and fused_transform; keep the FP32/FMA operation order identical so both inference paths follow the hai-llm numerical standard."""`
- 证据：`cuda/norm.py:260`（`rstd_buf[i] = T.rsqrt(rstd_buf[i] / d + eps)`） vs `asc/norm.py:10`（`from .rstd_1ulp import make_rstd_impl`）
- 证据：`asc/kv_quant.py:1-8`（"Register-only BF16 KV quantization primitives for Ascend SIMD"，scale/inverse 与 sticky-bit FP4 packing 改编自 `tile_kernels/quant/per_token_cast_asc.py`）
- 等级：A
- 意味着什么：Ascend 用补偿 Newton 迭代替代 CUDA 的 `T.rsqrt`，把 FP32/FMA 运算顺序钉死以对齐内部 `hai-llm` 的数值标准（动机是可复现性，非性能；机制部分为 E 级推断）。
- 疑点：`rstd_1ulp.py` 注释直接提到内部库 `hai_llm`，说明 Ascend 算子是内部 TileLang kernel 的开源移植。

### #23 Ascend 算子的功能缺口有明确代码位置

- 证据：`asc/store_kv_cache.py:79-80`
  > `if layout.bf16_elements or layout.sf_format == "fp32": raise NotImplementedError("Ascend V4 mixed/fp32-scale KV cache store is not supported")`
- 证据：`asc/fused_transform.py:850-851`
  > `assert sin is None, "Ascend RoPE requires full cos_sin_cache in cossin and sin=None; split inputs unsupported"`
- 证据：`asc/` 没有 `outlier_detection.py`（该 kernel 仅 `cuda/outlier_detection.py` 提供；根目录 `outlier_detection.py:1` 标注 "CUDA outlier detection"）
- 等级：A
- 意味着什么：同一算子名在 Ascend 上可能只支持当前模型用到的格式子集；老格式（V4 mixed/fp32 SF）与异常值检测在昇腾不可用。
- 疑点：`detect_outlier_and_abort` 在生产代码中没有任何调用点（仅测试），因此这个缺口当前无实际影响。

### #24 低精度 #1：主压缩 KV 是 FP4、per-16、e4m3 scale，288 B/token

- 证据：`tilelang_ops/cache_common.py:119-128`
  > `V41_FP4_FORMAT = KVCacheFormat(quantized_dtype="e2m1", quantized_elements=512, elements_per_sf=16, sf_format="e4m3", note="V41, FP4 main kv.", token_interleaved=True)`
- 证据：`cache_common.py:100-101`（`bytes_per_token = quantized_bytes + sf_bytes + bf16_elements*2`）→ 512×4bit=256 B + 512/16×1 B=32 B = **288 B**
- 证据：`model_metadata_utils.py:37-39`
  > `"""FP4 主压缩 KV 每 16 个值共用 scale；MXFP8 主 KV 和 SWA 每 32 个值共用 scale。"""`
  > `return 16 if not is_swa and cfg.use_fp4_sparse_kv else 32`
- 证据：`worker/cache_engine.py:327-332`（主 KV 用 `derive_main_kv_cache_num_per_channels(is_swa=False)` + `use_fp4=model_metadata.use_fp4_sparse_kv`）
- 等级：A
- 意味着什么：这是"类 NVFP4 但没有 global scale"的格式（`doc:217`），比 MXFP8 的 528 B 省 45% 带宽，代价是 scale 更密（per-16）。
- 疑点：`use_fp4_sparse_kv` 默认 True（`hugging_face/config.py:303`），即默认模型走 288 B。

### #25 低精度 #2：SWA KV 固定 MXFP8、per-32、UE8M0，528 B/token/层

- 证据：`cache_common.py:109-118`
  > `V41_FORMAT = KVCacheFormat(quantized_dtype="e4m3", quantized_elements=512, elements_per_sf=32, sf_format="ue8m0", note="V41, main/swa kv.", token_interleaved=True)`
- 证据：`cache_engine.py:261-263`（SWA 容量计算固定 `use_fp4=False`）
- 证据：`attn.py:82`（`self.swa_cache_num_per_channels = derive_main_kv_cache_num_per_channels(cfg, is_swa=True)`，恒为 32）
- 等级：A
- 意味着什么：SWA 不走 FP4，永远是 512 B e4m3 + 16 B UE8M0 = 528 B；因此"主 KV 是否 FP4"不影响 SWA 的带宽（R02a 已算出 ≈2.8 MB/seq 的固定开销）。
- 疑点：文档 `doc:217` 只讲"Main KV: per-16, e4m3"，没讲 SWA 的 528 B/MXFP8；需要本书补上。

### #26 低精度 #3：indexer K 是 MXFP4、per-32、UE8M0，68 B/token

- 证据：`cache_common.py:129-137`
  > `FP4_INDEX_KV_FORMAT = KVCacheFormat(quantized_dtype="e2m1", quantized_elements=128, elements_per_sf=32, sf_format="ue8m0", note="V4/V41, index kv.")`
- 证据：`cache_engine.py:73-78`
  > `"""获取这一层 indexer 的 FP4 KV bytes 数（包含 UE8M0 scale）"""`
  > `return ic.index_head_dim // 2 + ic.index_head_dim // 32`
  → 128/2 + 128/32 = **68 B**
- 证据：`v4_modules/index_attn.py:85-86`
  > `# Indexer Q/K always use FP4 (E2M1 + UE8M0 scale); weights and psum use BF16.`
  > `self.cache_format = FP4_INDEX_KV_FORMAT`
- 等级：A
- 意味着什么：indexer 的 K cache 每 token 只有 68 B，是 DSA 能用 512 topk 的经济性基础；它不经过 FlashMLA——paged decode 时由 `deep_gemm.fp8_fp4_paged_mqa_logits` 直接读这份 cache（`index_attn.py:434`），prefill 则先经 `load_paged_kv_cache` 解出 `(value, sf)`（`index_attn.py:190/199`）。
- 疑点：index cache 不是 token-interleaved（`token_interleaved` 默认 False），而是块内 data/SF 分区（`cache_common.py:31-32` 注释）。

### #27 低精度 #4：MoE 路由专家权重是 MXFP4、per-32、UE8M0

- 证据：`hugging_face/config.py:253-264`
  > `"expert_dtype": "fp4", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0"`
  > `mxfp4 = quant["expert_dtype"] == "fp4"`
- 证据：`hugging_face/weights.py:318-325`（HF scale dtype 必须是 `torch.float8_e8m0fnu`，`packed_sf = scale.view(param.sf.dtype)`，`param.mxfp4 = True`）
- 证据：`v4_modules/mlp.py:59-71`
  > `assert ((deepinfer_w.view(torch.int32) & 0x03030303) == 0).all(), "Only support FP4-QAT checkpoint"`
  > `_deepinfer_w, _deepinfer_sf = tile_kernels.quant.per_token_cast(..., "e2m1", 32, x_block_size=(fp8_block_size, fp8_block_size), round_sf=True, use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True)`
- 证据：`mlp.py:124-125`（grouped GEMM `recipe_b=(1, 32 if fp4_b else ...)`）
- 等级：A
- 意味着什么：专家权重是"预量化 checkpoint"（QAT），加载时直接读 packed FP4 + packed UE8M0 SF，不在运行时做 FP8→FP4 转换；激活仍是 FP8（`use_fp8_act`）→ `fp8xfp4` MMA。
- 疑点：`per_token_cast` 的 `use_tma_aligned_col_major_sf=True` 让 SF 布局与 `linear.py` 的 FP8 SF 布局一致，但 Ascend 要求 MN-major int16（见 #28），这里靠平台相关的 `deep_gemm` 接口吸收。

### #28 SF 打包宽度的平台差异被一个 dtype 常量吸收

- 证据：`vendor/DeepGEMM/README.md:68`
  > `SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single torch.int.`
- 证据：`vendor/DeepGEMM-Ascend/README.md:56`
  > `The scaling factor format on Ascend differs from NVIDIA's: each pair of UE8M0 scaling factors along the K dimension is packed into an int16, and the packed values are stored in MN-major order for optimal hardware efficiency.`
- 证据：`platform.py:63` / `:138`（`torch.int32` vs `torch.int16`）；`linear.py:164`（断言 SF dtype == `PLATFORM.fp8_sf_type`）
- 等级：B + A
- 意味着什么：同一份 bf16 权重/激活在两种硬件上会生成不同字节布局的 SF，"可移植的模型权重 + 平台相关的 SF 重排"是这套代码的基本假设。
- 疑点：`transform_sf_into_required_layout` 在 `mlp.py:73-81` 只处理 FP8 路径，FP4 路径依赖 HF 已 packed 的 SF，二者在 Ascend 上是否都满足 MN-major 未见断言。

### #29 官方文档关于 FlashMLA 开源版与内部版的原文

- 证据：`doc/inference_deployment_qa.md:233-237`
  > `- Are there better internal versions of kernel libraries such as DeepGEMM / FlashMLA?`
  > `    - They are broadly the same; the indexer and mega series differ somewhat. FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.`
  > `    - End-to-end performance is close — at most 5% apart.`
- 等级：B
- 意味着什么：官方承认的差距有两层：单 kernel（decode attention 慢 5%–10%）与端到端（≤5%）；同时点名 indexer 与 mega 系列也有差异——从代码看，"mega 系列"对应 DeepGEMM 的 Mega MoE / Mega Gate / Mega mHC 这一组融合算子（E 级对应关系）。
- 疑点：原文没有给"5%–10%"的硬件、batch、topk、KV 格式与基线版本；也没有说明"SASS optimization"具体优化了哪条指令序列。5%–10% 是 kernel 时间还是 kernel TFLOPS 也未说明。

### #30 对 "5%–10% / ≤5%" 的可行性分析（推断）

- 证据：`vendor/FlashMLA/README.md:41`（融合 kernel 在 B200 上 prefill 1460 / decode 950 TFlops）、`:55-63`（非融合 sparse decode 1024 TFlops）
- 证据：`vendor/FlashMLA/csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/phase1.cuh:37`（`asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 ...")`）——开源版是 CUDA C++ + inline PTX，仓库内没有任何 `.sass` 文件（`find vendor/FlashMLA -name '*.sass'` 为空）
- 等级：E（推断）
- 推理：decode attention 是带宽/延迟受限（topk gather + KV dequant），手写 SASS 最可能优化的是 load/issue 调度与寄存器分配，因此"decode kernel 慢 5%–10%"在量级上合理；端到端 ≤5% 意味着 attention 只占 decode step 的一部分（MoE 与 indexer 也占时间），5%–10% 的 kernel 退化被其他阶段摊薄。若 attention 占 step 的 30%–50%，则端到端退化为 1.5%–5%，与文档一致。
- 意味着什么：书中应把这两个数字分开表述："decode attention kernel 慢 5%–10%，端到端 ≤5%"，不要合并成"FlashMLA 慢 5%–10%"。
- 疑点：没有 profiler 数据能验证 attention 占 decode step 的比例；该比例随 EP 规模、batch、KV 长度变化。

## 3. 必答问题逐条回答

### Q1 FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect 各自职责与调用行号

| 库 | 职责（一句话） | 在模型哪一步被调用 | 调用行号 | 事实条目 |
|---|---|---|---|---|
| **FlashMLA** | DSA token 级稀疏 attention 的 prefill/decode kernel，以及融合 norm-RoPE-attn-RoPE-cast | 每个 attention 层的 core attention（prefill、decode、DFlash 草稿 decode） | `attn.py:617`（prefill）、`attn.py:660`（decode）、`attn.py:994`（DFlash decode）；底层 `attention_backend.py:64`、`:125`；融合 kernel `attention_backend.py:51`、`:108` | #3 #4 #5 |
| **DeepGEMM** | FP8/FP4/BF16 GEMM（dense + grouped）、indexer MQA logits、Mega MoE/Gate/mHC | 所有线性层；MoE 专家两段 GEMM；indexer 打分；MoE 融合路径；HC 融合 | `linear.py:165`；`mlp.py:119`/`:130`（grouped）；`index_attn.py:299`/`:317`/`:434`/`:447`（MQA logits）；`mlp.py:549`（Mega MoE）、`:398`（Mega Gate）；`hc.py:270`（Mega mHC） | #7 #11 #12 |
| **DeepEP** | EP all-to-all：MoE dispatch/combine（含 FP8 dispatch），另有 Engram/PP/Bucket 原语 | MoE 层路由后 dispatch、专家计算后 combine；Engram 表；CP 的 all-gather | `mlp.py:469`（dispatch）、`mlp.py:501`（combine）；`engram.py:181`；`cp.py:17` | #8 |
| **TileKernels** | MoE 门控 topk、SwiGLU+per-token 量化、mHC 七个 kernel、Engram、通用 quant；pip 包 | MoE gate；专家 GEMM 间激活；HC 子层；Engram 查表；attention 输出 FP8 cast | `mlp.py:400`（gate）、`mlp.py:436`/`:521`（SwiGLU）、`attn.py:200`（cast）、`mhc_ops.py:8-14`、`engram.py:134`/`:151`、`deepseek.py:180` | #9 |
| **DeepSelect** | DSA indexer 与 sampler 的 TopK（替代 `torch.topk`） | indexer 选 candidate block / 512 token；采样 topk | `index_attn.py:216`、`:361`、`:494`；`sampler.py:12`、`sampler_utils.py:4` | #10 |

补充：`tilelang_ops/` 不是第六个库，而是每个后端 43 个 `*_tl` 工厂（KV 存取、compress、norm、sampler、gather、DFlash 元数据等），与上述 5 个库并列构成完整算子面（#15）。

### Q2 Mega MoE 是什么？在代码哪里？文档为什么说它在 scale-up 域内总是更快？

**是什么**（#11）：DeepGEMM 的单个 persistent kernel，融合 EP dispatch、L1 GEMM（gate/up）、SwiGLU、L2 GEMM（down）、EP combine，并在核内把 NVLink 通信与张量核计算重叠。shared expert 的 L1/L2 也可一起融合（`BlockPhase::SharedLinear1/SharedLinear2`）。

**代码位置**：
- 库实现：`vendor/DeepGEMM/csrc/apis/mega_moe.hpp`（host 校验、对称内存大小、权重变换）；`vendor/DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`（kernel 主体）；`vendor/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（核内调度、pool/ring）；Python 封装 `vendor/DeepGEMM/deep_gemm/mega/__init__.py`（`SymmBuffer`）。
- DeepInfer 侧：`v4_modules/mlp.py:184-188`（启用条件）、`:254-257`（选择 `fp8_fp4_mega_moe` / `bf16_mega_moe`）、`:303-315`（分配 `SymmBuffer`）、`:528-563`（`mega_moe_forward`，`:549` 调用）、`:571-572`（`forward` 分流）。
- 权重准备：`mlp.py:93-98`（`transform_weights_for_mega_moe`）。

**文档为什么说"域内总是更快"**（#14）：`doc:231` 的原话是 "Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE." 机制上（推断，E）：
1. 融合消除了 dispatch→GEMM→combine 之间的 kernel launch 与全局同步，SM 不会在阶段边界空转；
2. 通信与 GEMM 在同一个 kernel 内以 ring/pool 流水重叠，NVLink 传输时间被张量核计算遮住，而不是串行相加；
3. 它要求 symmetric memory，即所有 rank 必须在同一 NVLink scale-up 域内（`DeepGEMM/README.md:114`），因此"域内"这个前提同时是性能前提和可用性前提；
4. 代码用 `deep_ep.get_physical_domain_size` 把"域内"变成可判定条件（`parallel_state.py:131-134`），域外直接回退 DeepEP + grouped GEMM（`mlp.py:571-610`）。

需要注意：文档的"always faster"是定性断言，没有给 EP 规模、batch、硬件与基线数字；代码只能证明"域外不用"，不能证明"域内总是更快"。

### Q3 `tilelang_ops/` 根目录、`cuda/`、`asc/` 的架构关系与分发机制

- **根目录 = 规范层**：29 个模块同时提供 (a) 公开 Python API（断言、形状推导、缓存格式解析）、(b) PyTorch 参考实现（`*_ref`）、(c) pytest 测试；4 个基础设施文件（`ops.py`、`common.py`、`cache_common.py`、`conftest.py`）。合计 10,945 行（#15）。
- **`cuda/` 与 `asc/` = 实现层**：各 30 个文件、43 个 `*_tl` 工厂，27 个文件名完全同名；CUDA 多 `quant.py`、`outlier_detection.py`、`prepare_decode_metadata.py`（后者与根/asc 的 `prepare_v4_decode_metadata.py` 导出同名工厂）；Ascend 多 `kv_quant.py`、`rstd_1ulp.py`（辅助宏，不是工厂）（#15、#21）。
- **分发 = `ops.py`**：`PLATFORM.tilelang_ops_backend`（`"cuda"`/`"asc"`）→ `importlib.import_module` → `pkgutil.iter_modules` 遍历该包所有子模块 → 把所有不以 `_` 开头、不是模块、以 `_tl` 结尾的顶层符号复制到 `ops` 的 globals，并用 `assert` 防冲突（`ops.py:13-22`）。缺符号时模块级 `__getattr__` 抛 `KernelUnavailableError`（`ops.py:25-32`），`conftest.py:100-113` 把它转成非 CUDA 平台的测试 skip（#16）。
- **共享 ABI**：`cache_common.py` 的 `KVCacheFormat` 定义所有 KV 布局与 `bytes_per_token`；`common.py` 定义 FP4/FP8 amax margin、`DRAFT_PROB_PENALTY` 等常量。两个后端都 import 同一份定义（#24–#27）。
- **构建期还有一层分发**：`scripts/build_vendor.sh` 按 `DEEPINFER_ACCELERATOR` 决定装 DeepEP/DeepGEMM 还是 DeepEP-Ascend/DeepGEMM-Ascend；`deepinfer/_so/<accelerator>-<arch>/` 让两套 `.so` 共存（#2、#19）。

一句话：根目录定义"算什么"，`cuda/`/`asc/` 定义"怎么算"，`ops.py` 在 import 时按名字把"怎么算"接到"算什么"上。

### Q4 NVIDIA 与昇腾 950 的功能差异与算子实现对比

**功能差异**

| 维度 | NVIDIA（SM100/SM103） | 昇腾 950 | 证据 |
|---|---|---|---|
| 构建工具链 | CUDA 13.1+、nvcc、CUTLASS | CANN 9.2/9.20、Bisheng、Ascend C++、`torch_npu` | `FlashMLA/README.md:79-88`；`DeepGEMM-Ascend/README.md:21-29` |
| 通信后端 | NCCL（Gin）、NVLink + RDMA | HCCL/HCOMM、UBMEM/URMA | `platform.py:58/89`；`DeepEP-Ascend/README.md:5` |
| SF 打包 | 4×UE8M0 / `int32` | 2×UE8M0 / `int16`，MN-major | `DeepGEMM/README.md:68`；`DeepGEMM-Ascend/README.md:56`；`platform.py:63/138` |
| Mega MoE | `fp8_fp4_mega_moe` + `bf16_mega_moe` | 仅 `fp8_fp4_mega_moe` | `DeepGEMM-Ascend/deep_gemm/__init__.py:137-141`；`mlp.py:185` |
| Mega Gate / Mega mHC | 有（`bf16_mega_gate`、`mega_mhc`） | 无导出 → 回退 TileKernels | `DeepGEMM/deep_gemm/__init__.py:66/80`；`mlp.py:228/397-400`；`hc.py:99-100` |
| 融合 norm-RoPE-attn-RoPE-cast | 支持 | 不支持 | `FlashMLA/README.md:43`；`attn.py:91` |
| 稀疏 prefill / decode 峰值 | 1350 / 1024 TFlops（B200） | 410 TFlops（95%）/ 360 TFlops（83%） | `FlashMLA/README.md:51-63` |
| DeepSelect dtype | bf16 + fp32 | 仅 bf16（fp32 sampling 回退 `torch.topk`） | `DeepSelect/README.md:37`；`npu_ops/deep_select.py:20-26` |
| Batch-invariant / 确定性 | 支持（env 开关） | 断言关闭 batch-invariant；确定性算法强制开启 | `README.md:27`；`runtime_config.py:61-69` |
| PDL | 开启（`deep_gemm.set_pdl(True)`） | 不开启 | `runtime_config.py:64-66` |
| EP 限制 | zero-SM RDMA EP 不支持 | `do_expand=True`、`allow_multiple_reduction=True`、`do_handle_copy=False`；topk_idx 固定 int64；SL/QP 不可指定 | `DeepEP/README.md:39`；`DeepEP-Ascend/README.md:72-74` |
| 未实现项 | — | `lb_prefetch_weights`/`lb_reduce_grads` 通信 kernel、reduce-scatter/all-reduce、hybrid comm、CPU Engram、graph capture | `DeepEP-Ascend/README.md:76-84` |

**实现对比（同一算子的两版写法）**

| 算子 | CUDA 版 | Ascend 版 | 证据 |
|---|---|---|---|
| `convert_kv_index` | `T.Kernel(s,b,threads=128)`；`T.pdl_sync()`；fragment 内做 `block_table` 重映射 | `T.Kernel(num_cores)` persistent；`k` 必须 2 的幂且 ≤4096；UB 4 级多版本；`T.SimtVF(threads=k)` | `cuda/flash_mla_utils.py:27-38`；`asc/flash_mla_utils.py:20-51` |
| `store_kv_cache` | `T.copy` + ue8m0 宏；支持所有 `KVCacheFormat` | 寄存器内量化 + e4m3 倒数 LUT + 每行一次 DMA；V4 mixed/fp32 SF 直接 `NotImplementedError` | `cuda/store_kv_cache.py:11-24`；`asc/store_kv_cache.py:1-60`、`:79-80`；`asc/kv_quant.py:24-45` |
| `norm` | `T.rsqrt`（硬件近似） | `rstd_1ulp` 补偿 Newton，对齐 hai-llm 1 ulp | `cuda/norm.py:260`；`asc/norm.py:10`；`asc/rstd_1ulp.py:1-10` |
| 代码风格统计 | PDL 56 处、TMA 29 处、warp-specialized 关闭 29 处 | `tilelang.ascend` 36 处、`T.SimtVF` 43 处、`S.v*` 623 处、`num_vec_cores` 50 处 | #21 |

**结论**：两平台的"算子契约"（函数名、参数、KV 布局、语义）基本一致，差异集中在 (a) 平台常量（SF dtype、distributed backend）；(b) 少数融合算子是否可用（Mega Gate/mHC、融合 attention）；(c) 实现范式（TMA/PDL/warp 级并行 vs persistent vector core + UB + SIMD）；(d) 若干老格式与边界功能在 Ascend 未实现。

### Q5 低精度格式在代码中的具体位置

| 用途 | 格式 | 每 token 字节 | 代码定义 | 使用/分配位置 |
|---|---|---|---|---|
| **主压缩 KV（ca_cache）** | FP4 E2M1，per-16，e4m3 scale（"NVFP4 无 global scale"） | 256 + 32 = **288 B** | `cache_common.py:119-128`（`V41_FP4_FORMAT`）；`cache_common.py:100-101`（字节公式） | `cache_engine.py:327-332`（`use_fp4=use_fp4_sparse_kv`）；`model_metadata_utils.py:37-39`；`attn.py:644-646`（作为 `key_cache_extra`）；`doc:217` |
| **SWA KV（swa_cache）** | MXFP8 E4M3，per-32，UE8M0 | 512 + 16 = **528 B** | `cache_common.py:109-118`（`V41_FORMAT`） | `cache_engine.py:261-263`（固定 `use_fp4=False`）；`attn.py:82`；`attn.py:642-643`（作为主 `key_cache`） |
| **Indexer K（ica_cache）** | MXFP4 E2M1，per-32，UE8M0 | 64 + 4 = **68 B** | `cache_common.py:129-137`（`FP4_INDEX_KV_FORMAT`）；`cache_engine.py:73-78`（`index_head_dim//2 + index_head_dim//32`） | `index_attn.py:85-86`（`self.cache_format = FP4_INDEX_KV_FORMAT`）；`index_attn.py:165`（index Q 也量化到 FP4） |
| **MoE 路由专家权重** | MXFP4 E2M1，per-32，UE8M0 | 权重，非 KV | `hugging_face/config.py:253-264`；`mlp.py:59-71`（`per_token_cast(..., "e2m1", 32, ...)`）；`weights.py:318-325` | `mlp.py:119`（`m_grouped_fp8_fp4_gemm_nt_contiguous`，`recipe_b=(1,32)`）；`mlp.py:178-182`（`use_fp4_weight` 与 `mma_type`） |
| （对照）FP8 主 KV / 非 FP4 模式 | MXFP8，per-32，UE8M0 | 528 B | `cache_common.py:109-118` | `use_fp4_sparse_kv=False` 时 |
| （对照）V4 index KV fp32 scale | E4M3，per-128，fp32 | 128 + 4 = 132 B | `cache_common.py:138-146` | 仅在老 V4 路径；Ascend 明确不支持（`asc/store_kv_cache.py:79-80`） |

关键点：
- 288 B 与 528 B 的差别不只是位宽，还有 scale 粒度（per-16 vs per-32）与 scale 类型（e4m3 vs UE8M0）；`doc:217` 的 "like NVFP4 but without the global scale" 指的就是 288 B 这份。
- 68 B 的 indexer K 是 `128` 维：64 B 数据 + 4 B UE8M0，正好 68；它不经过 FlashMLA：paged decode 由 DeepGEMM 的 `fp8_fp4_paged_mqa_logits` 直接读 cache，prefill 先经 `load_paged_kv_cache` 解包（#26）。
- 三份 cache 的角色在 `attn.py:640-653` 一次看清：SWA 层 → FlashMLA 的 `k_cache` 是 `swa_cache`（528）+ `ca_cache`（288）作 extra；非 SWA 层 → 只有 `ca_cache`。注意 DeepInfer 把 288 B 的 `ca_cache` 叫 "main KV"，把 528 B 叫 "SWA KV"，与 FlashMLA 形参名相反。

### Q6 官方文档关于"开源版 FlashMLA 少一个 SASS 优化"的原文与分析

**原文**（`doc/inference_deployment_qa.md:233-237`）：

> `- Are there better internal versions of kernel libraries such as DeepGEMM / FlashMLA?`
> `    - They are broadly the same; the indexer and mega series differ somewhat. FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.`
> `    - End-to-end performance is close — at most 5% apart.`

**逐句解读**：
1. "They are broadly the same"：开源版与内部版整体同源，不是两个不同实现。
2. "the indexer and mega series differ somewhat"：差异不只在 FlashMLA；indexer（MQA logits）与 mega 系列也与内部版有出入——这与代码里 Mega 系列存在平台/类型门控（#12）相呼应。
3. "FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower"：缺口是**一个手写 SASS 优化**，影响面是**decode kernel**，幅度 5%–10%。注意主语是 decoding kernel，不是整个 FlashMLA，也不是端到端。
4. "End-to-end performance is close — at most 5% apart"：端到端差距 ≤5%，说明 decode attention 只占整个 decode step 的一部分，kernel 级 5%–10% 被摊薄。

**可核对与不可核对**：
- 可核对：开源 FlashMLA 仓库内没有任何 `.sass` 文件，kernel 是 CUDA C++ + inline PTX（`vendor/FlashMLA/csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/phase1.cuh:37` 的 `ld.global.nc...v8.s32` 就是典型 PTX 级手写 load）。这与"内部版多一个手写 SASS 优化"在结构上自洽（E 级推断）。
- 不可核对：文档没给硬件、batch、topk、KV 格式、基线 commit、测量口径（kernel 时间 vs TFLOPS），也没说是哪条指令序列。仓库里没有内部版，无法对比。
- 数量级合理性（E）：若 decode attention 占 step 时间的 30%–50%，kernel 慢 5%–10% 对应端到端 1.5%–5%，与"≤5%"一致；这也反过来说明 decode attention 不是唯一瓶颈（MoE/indexer 占大头）。
- 写作建议：不要把两句合并成"FlashMLA 慢 5%–10%"，应写成"decode attention kernel 慢 5%–10%，端到端 ≤5%"，并注明这是官方定性说法、无公开测量条件。

## 4. 新发现的问题

1. **`tilelang_ops/flash_mla_utils.py` 与 FlashMLA 无关**：它是把 topk 逻辑索引经 `block_tables` 映射成物理 KV slot 的自研 TileLang 算子（`flash_mla_utils.py:10-37`，调用点 `index_attn.py:508`），只因服务于 FlashMLA 的 `indices_in_kvcache` 而得名。书中若按文件名归类会误导。
2. **`detect_outlier_and_abort` 只有 CUDA 实现且无生产调用**：`outlier_detection.py` 根目录是 "CUDA outlier detection and PyTorch reference"，`grep` 全仓库除测试外无调用点；它更像调试/安全网算子，不是推理路径的一部分。
3. **DeepGEMM-Ascend 缺 `bf16_mega_moe` / `bf16_mega_gate` / `mega_mhc`**：`vendor/DeepGEMM-Ascend/deep_gemm/__init__.py:137-141` 只导出 `SymmBuffer`、`transform_weights_for_mega_moe`、`fp8_fp4_mega_moe`。因此 DeepInfer 在 NPU 上把 Mega 限制为 `fp8xfp4`，gate 回退 `tile_kernels.moe.moe_topk_gate`，mHC 回退 `tile_kernels.mhc.*`。文档没有解释这一非对称性。
4. **`is_within_nvl_domain` 在 NPU 恒真**：`parallel_state.py:131-133` 直接 `return True`，函数名与语义在昇腾上不成立；它把"是否 scale-up 域"简化为"是否 NPU"，掩盖了昇腾 supernode 内外带宽差异。
5. **文档的 Mega MoE 性能断言缺条件**：`doc:231` 的 "significantly faster ... regardless of EP size or batch size" 没有数字、基线、硬件；而代码的启用条件（#12）说明它并非"任何情况"都可用。书中引用时必须补"域内 + 平台/MMA 限制"。
6. **288 B FP4 主 KV 只能作 `extra_k_cache`**：FlashMLA README:141 明说该格式仅在 `k_cache` 为 V4.1 格式时对 `extra_k_cache` 有效。V4.1 所有层都有 SWA，所以 DeepInfer 恰好满足；若未来出现"无 SWA + FP4 主 KV"的层，当前路径会违反该约束（`attn.py:648-653` 会把 288 B 直接当主 cache）。
7. **`is_fp8_kvcache=True` 的命名与 FP4 现实不符**：`attention_backend.py:134` 对所有量化 KV 都传 True，真正格式由字节数判定（README:138）；读者容易误以为 FP4 走的是另一条 API。
8. **TileKernels 不可审计**：它是 pip 包而非 submodule，`vendor/` 下没有源码；本书只能写"调用了什么"，不能写"内部怎么实现"。这与其他五个库的证据强度不对等。

## 5. 可用于图表的材料

### 5.1 内核库 → 职责 → 调用点（可直接做表）

| 库 | 一句话职责 | 调用点（文件:行） |
|---|---|---|
| FlashMLA | DSA 稀疏 attention（prefill/decode）+ 融合 norm/rope/cast | `attn.py:617`、`:660`、`:994` |
| DeepGEMM | GEMM（dense/grouped）+ indexer logits + Mega 系列 | `linear.py:165`；`mlp.py:119/130/549/398`；`index_attn.py:299/317/434/447`；`hc.py:270` |
| DeepEP | EP dispatch/combine + Engram/PP/Bucket | `mlp.py:469/501`；`engram.py:181`；`cp.py:17` |
| TileKernels | gate、SwiGLU+quant、mHC、Engram | `mlp.py:400/436/521`；`attn.py:200`；`mhc_ops.py:8-14` |
| DeepSelect | indexer / sampler TopK | `index_attn.py:216/361/494`；`sampler.py:12` |
| tilelang_ops | 每后端 43 个 `*_tl` 工厂（KV、compress、norm、sampler…） | `ops.py:15`；根目录 API |

### 5.2 TileLang 算子层规模（可做柱状/表格）

| 目录 | 文件数 | 行数 | `*_tl` 工厂数 |
|---|---:|---:|---:|
| 根目录（API+参考+测试） | 33 | 10,945 | —（调用 `ops.*`） |
| `cuda/` | 30 | 4,968 | 43 |
| `asc/` | 30 | 7,418 | 43 |

同名文件 27 对；CUDA 独有 `quant.py`、`outlier_detection.py`、`prepare_decode_metadata.py`；Ascend 独有 `kv_quant.py`、`rstd_1ulp.py`。

### 5.3 一份 token 的 KV 字节账（可做堆叠条形图）

| 组成 | 数据字节 | scale 字节 | 合计 |
|---|---:|---:|---:|
| 主压缩 KV（FP4 per-16 e4m3） | 256 | 32 | **288 B** |
| SWA KV（MXFP8 per-32 UE8M0） | 512 | 16 | **528 B** |
| Indexer K（MXFP4 per-32 UE8M0） | 64 | 4 | **68 B** |
| 对照：FP8 主 KV（per-32 UE8M0） | 512 | 16 | 528 B |

### 5.4 MoE 层两条执行路径（可做流程图）

- **非 Mega（域外 / NPU 非 fp8xfp4）**：gate（`mlp.py:400`，TileKernels 或 DeepGEMM Mega Gate）→ DeepEP dispatch（`:469`）→ grouped GEMM L1（`:119`）→ SwiGLU+quant（`:436`）→ grouped GEMM L2（`:449`）→ DeepEP combine（`:501`）。
- **Mega（域内 + 平台允许）**：gate（`:398`/`:400`）→ 写入 `SymmBuffer`（`fused_norm_cast_kwargs`，`mlp.py:274-301`）→ 单次 `mega_moe_func`（`:549`）完成 dispatch/L1/SwiGLU/L2/combine（+ shared expert）。

### 5.5 双硬件对照卡（可做两列表）

- 平台常量：`cuda`/`nccl`/`int32` vs `asc`/`hccl`/`int16`（`platform.py:57-63`、`:86-138`）。
- 构建：`CUDAExtension` + `cuda_utils.cpp` vs `NpuExtension` + `asc_utils.cpp`（`setup.py:38-72`）。
- 算子范式：TMA/PDL/warp-specialized vs persistent vector core/UB 多版本/SIMD（#21）。
- 功能缺口：融合 attention、Mega Gate、Mega mHC、fp32 sampler、V4 mixed KV、outlier detection（#23、Q4 表）。
- 性能口径：B200 sparse prefill 1350 / decode 1024 TFlops；Ascend 950 410（95%）/ 360（83%）TFlops（FlashMLA README:51-63）。
