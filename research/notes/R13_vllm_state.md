# R13 vLLM 现状（与 DeepInfer 对比）

## 0. 元信息

- 使用的代码版本：`vllm-project/vllm` @ `2d40c0609fd672e845ddfef0324ab6d8dd616af4`（main，提交时间 2026-10-22 19:41:07 -0700，即北京时间 2026-10-23 10:41）
- 本地仓库是**浅克隆，只有 1 个提交**（`git log` 仅返回 2d40c06），因此"从哪个版本开始有"这类问题多数无法用 git 历史确认，只能引用代码内注释、文档中的版本字样或 PR 号
- 版本号：仓库无 tag、无 `_version.py` 实体。代码内弃用提示指向"will be removed in v0.33.0"（`vllm/engine/arg_utils.py:457,463,470`），PR #62840 正文称"TileLang ROCm/XPU support is unstable on vLLM 0.32"，据此判断当前 main ≈ 0.32 之后、0.33 之前的开发版（E 级）
- 查阅的主要来源：
  - 源码：`vllm/v1/engine/`、`vllm/v1/core/sched/`、`vllm/v1/core/kv_cache_*.py`、`vllm/v1/kv_cache_interface.py`、`vllm/config/*.py`、`vllm/models/deepseek_v41/`、`vllm/v1/spec_decode/`、`vllm/v1/worker/gpu/spec_decode/`、`vllm/v1/structured_output/`、`vllm/distributed/eplb/`
  - 文档：`docs/design/arch_overview.md`、`docs/design/model_runner_v2.md`、`docs/design/prefix_caching.md`、`docs/design/paged_attention.md`、`docs/design/hybrid_kv_cache_manager.md`、`docs/design/nixl_kv_push_connector.md`、`docs/configuration/optimization.md`、`docs/serving/*.md`、`docs/features/*.md`、`docs/usage/v1_guide.md`、`docs/usage/reproducibility.md`
  - 配方：`recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`
  - 外部一手：vLLM 官方 X `@vllm_project` 2026-10-21 推文串（status 2112703311933747347）；vLLM 博客 2026-10-07《DeepSeek-V4.1-Flash on vLLM》；DeepInfer 仓库 README 与 `doc/images/decode_pareto.png`；vLLM issue #57448、#62839、PR #62840
  - 二手：AICoder/AGI Hunt 对推文的转述；第三方单机实测（dev.to 8×B300、GPUStack 8×B200，C 级）

## 1. 三个最重要的发现

1. **V0 已被彻底删除，V1 是唯一引擎**：`vllm/engine/llm_engine.py` 只剩 5 行别名，`VLLM_USE_V1` 环境变量在代码里已不存在；进程模型固定为"API server（可多个，ZMQ 多对多）—EngineCore（每 DP rank 一个）—GPU worker（每卡一个）"，且模型执行层正在从 V1 Model Runner 迁到默认开启的 MRV2。
2. **异步调度在 vLLM 里已经是默认行为，不是可选优化**：`async_scheduling` 默认按兼容性自动置 True（`vllm/config/vllm.py:1665-1699`），机制是调度器为下一步预填 `-1` 占位 token（`AsyncScheduler`），与 DeepInfer 的 ZOS"CPU 永远晚一步"是同一类设计；但文档仍有"async scheduling (experimental)"的滞后表述（`docs/serving/expert_parallel_deployment.md:214`）。
3. **"6%" 不是博客，是 vLLM 官方 X 推文**（2026-10-21 00:31 UTC，DeepInfer 开源后数小时）。它的对比方式是"vLLM 在 GB200 上跑 DeepSeek 的 decode 配置（V4.1-Flash、131K、nextn=3、batch 8–256），对 DeepInfer 仓库公布的 decode Pareto 曲线"——而那条曲线是 **64×SM100、EP64DP64、per-token acceptance 固定 75%** 下测的。推文没有公布 vLLM 侧的 GPU 数、TP/EP 配置，所以这是"跨集群对公布数字"，不是同机对跑。

## 2. 事实条目

### #1 V0 引擎已删除，`LLMEngine` 只是 V1 的别名
- 证据：`vllm/engine/llm_engine.py:1-8`
  > `from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine`
  > `LLMEngine = V1LLMEngine  # type: ignore`
  > `"""The `LLMEngine` class is an alias of [vllm.v1.engine.llm_engine.LLMEngine][]."""`
  `vllm/engine/async_llm_engine.py` 同理别名到 `vllm.v1.engine.async_llm.AsyncLLM`。全仓库 `grep VLLM_USE_V1` 无结果。
- 等级：A
- 意味着什么：读 vLLM 时不必再区分 V0/V1，"V1 架构"就是 vLLM 架构本身；老博客/老脚本里的 `VLLM_USE_V1=1` 已无意义（但 `--enforce-eager` 等运行开关仍有效，不要混为一谈）。
- 疑点：无。

### #2 进程模型：API server / EngineCore / GPU worker / DP coordinator
- 证据：`docs/design/arch_overview.md:67-110`
  > API server 处理 HTTP、tokenization、多模态加载，默认 1 个，DP 时自动扩到 DP 大小，也可 `--api-server-count` 手动设；与 engine core 通过 ZMQ 多对多连接
  > "There is **1 engine core process per data parallel rank**."
  > "There is **1 worker process per GPU**. The total number of GPU worker processes equals `tensor_parallel_size x pipeline_parallel_size` per engine core."
  > 进程总数表：`A + DP + N`（+1 if DP>1），例：4 卡 TP=4 → 1 API + 1 core + 4 worker = 6 进程
- 等级：B（与 A 级代码一致：`vllm/v1/engine/utils.py:237-289` 的 `CoreEngineProcManager` 用 `context.Process(target=EngineCoreProc.run_engine_core, name="EngineCore_DP{i}")` 起进程）
- 意味着什么：vLLM 的 CPU 侧开销是"每 DP rank 一个调度进程"；DeepInfer 的 ZOS 把调度放进 C++ 扩展（`csrc/engine/zos.cpp`），vLLM 用多进程 + ZMQ 解耦，二者都在解决同一个问题（调度不能挡住 GPU）。
- 疑点：无。

### #3 EngineCore 是忙等循环，PP/异步靠 batch queue 填流水
- 证据：`vllm/v1/engine/core.py:1609-1619`
  > `def run_busy_loop(self):`
  > `while self._handle_shutdown():`
  > `self._process_input_queue()`
  > `self._process_engine_step()`
  `vllm/v1/engine/core.py:698` `def step_with_batch_queue(...)`；`vllm/config/vllm.py:588-599`
  > "PP requires PP-size concurrent batches to fill the pipeline. Async scheduling requires 2 concurrent batches to overlap."
  > `if self.use_v2_model_runner: return pp_size + 1` / `if pp_size <= 1: return 2`
- 等级：A
- 意味着什么：vLLM 的"异步"是**批级流水**（同时有 2 个 batch 在飞），不是 token 级；PP 时按 PP size 预排 batch，MRV2 下再多留 1 个。
- 疑点：`step_with_batch_queue` 与 `max_concurrent_batches` 的对应关系只在 PP/异步路径生效，单卡默认仍是同步 `step()`。

### #4 模型执行层默认已切到 MRV2（Model Runner V2），V1 runner 是回退
- 证据：`vllm/config/vllm.py:714-749`
  > `use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER`
  > `if use_v2_model_runner is not None: return use_v2_model_runner`
  > `unsupported = self._get_v2_model_runner_unsupported_features()`
  > `if unsupported: ... return False`
  > `return True`
- 等级：A
- 意味着什么：`VLLM_USE_V2_MODEL_RUNNER` 不设时，只要 Triton 可用且没有不支持的特性，就走 MRV2。V4.1-Flash 的配方在 Blackwell 上显式设 `VLLM_USE_V2_MODEL_RUNNER: "1"`（`recipes/.../DeepSeek-V4.1-Flash.yaml:148,340,420,442`）。
- 疑点：`docs/design/model_runner_v2.md:9` 自述"not yet feature-complete, not rigorously tested"，与"默认开启"并存，说明默认开关是按平台+特性白名单做的，不是全模型全平台。

### #5 continuous batching：调度器里没有 prefill/decode 阶段之分
- 证据：`vllm/v1/core/sched/scheduler.py:677-685`
  > `# NOTE(woosuk) on the scheduling algorithm:`
  > `# There's no "decoding phase" nor "prefill phase" in the scheduler.`
  > `# Each request just has the num_computed_tokens and num_tokens_with_spec.`
  > `# At each step, the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens_with_spec.`
  > `# This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the "jump decoding" optimization in the future.`
  文档同义：`docs/usage/v1_guide.md:69-74`（"unified scheduler treats both prompt and output tokens the same way by using a simple dictionary ... without a strict separation between prefill and decode phases"）
- 等级：A
- 意味着什么：continuous batching、chunked prefill、投机解码、前缀缓存在同一个 token 预算循环里统一处理；这是 V1 相对 V0 最核心的简化。
- 疑点：无。

### #6 chunked prefill 默认开启，且 decode 优先
- 证据：`vllm/config/scheduler.py:134` `enable_chunked_prefill: bool = True`；`docs/configuration/optimization.md:53`
  > "In V1, **chunked prefill is enabled by default whenever possible**. With chunked prefill enabled, the scheduling policy prioritizes decode requests. It batches all pending decode requests before scheduling any prefill operations. ... If a pending prefill request cannot fit into `max_num_batched_tokens`, it automatically chunks it."
- 等级：A（默认值）/ B（策略描述）
- 意味着什么：默认部署下 TTFT 会被 decode 抢占，长 prompt 被切片；要控制尾巴延迟可调 `max_num_batched_tokens` 或 `prefill_schedule_interval`（`vllm/config/scheduler.py:203-208`，默认 1，即每步都可进 prefill）。
- 疑点：encoder-decoder 模型会强制关掉 chunked prefill 与前缀缓存（`vllm/config/scheduler.py:297-305`）。

### #7 异步调度的实现：调度下一步时用 `-1` 占位 token
- 证据：`vllm/v1/core/sched/async_scheduler.py:14-50`
  > `class AsyncScheduler(Scheduler):`
  > `self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens`
  > `# Add placeholders for the new draft/spec tokens.`
  > `# We will update the actual spec token ids in the worker process.`
  > `request.spec_token_ids = self._spec_token_placeholders`
  > `request.num_output_placeholders += (self.num_sampled_tokens_per_step + cur_num_spec_tokens)`
  `:72-73` `request.num_output_placeholders -= num_new_tokens` / `assert ... >= 0`
- 等级：A
- 意味着什么：与 DeepInfer ZOS 完全同构——CPU 用上一步的"占位"结果继续排下一步，真实 token id 由 worker 回填。代价也一样：结构化输出、停止条件、KV 计数都要为"未定 token"做特殊处理（`async_scheduler.py:36-40` 对 `use_structured_output` 单独开 `pending_structured_output_tokens`）。
- 疑点：`_update_request_with_output(..., is_stale=True)` 的存在说明存在"迟到/过期输出"路径，具体触发条件未逐一追。

### #8 异步调度默认开，但有明确的兼容性黑名单
- 证据：`vllm/config/vllm.py:1648-1699`
  > 显式 `--async-scheduling` 时，遇到 ROCm DeepEP high-throughput DBO、V1 runner 下的非 EAGLE/Ngram/draft_model/dspark/dflash 投机方法、`disable_padded_drafter_batch=True`、`pipeline_parallel_size > 1` 都会直接 `raise ValueError`
  > `elif self.scheduler_config.async_scheduling is None: ... else: self.scheduler_config.async_scheduling = True`
  > pooling 模型、executor 不支持（如 Ray 某些后端）、ROCm DeepEP DBO 时自动置 False
- 等级：A
- 意味着什么：默认部署就是异步调度；限制集中在"非 EAGLE 系的投机解码 + PP + ROCm DBO"。MRV2 下 PP 也能异步（`max_concurrent_batches = pp_size + 1`）。
- 疑点：文档口径滞后：`docs/serving/expert_parallel_deployment.md:214` 仍写"Async scheduling (experimental): Try `--async-scheduling`"。以代码为准。

### #9 PagedAttention：概念还在，`paged_attention.md` 已自我声明是历史文档
- 证据：`docs/design/paged_attention.md:3-6`
  > "!!! warning"
  > "This is a historical document based on the original paper for vLLM."
  > "It no longer describes the code used in vLLM today."
  实际实现：KV 以 block 为单位池化（`vllm/v1/core/block_pool.py`、`kv_cache_manager.py`），attention 由 FlashAttention / FlashInfer / FlashMLA / Triton 等 backend 消费 block table（`docs/README.md:48`）。
- 等级：B（文档自述）+ A（文件存在）
- 意味着什么：书里不能再用 2023 年那套"paged_attention_kernel"描述今天的 vLLM；分页仍是内存管理基石，但 kernel 已经是各家 backend。
- 疑点：`csrc/attention/attention_kernels.cu` 是否仍被某 backend 使用，未逐一核对。

### #10 前缀缓存：block 哈希 = 父哈希 + 本块 token + extra keys，默认开启
- 证据：`vllm/v1/core/kv_cache_utils.py:707-736`
  > `def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None) -> BlockHash:`
  > `if not parent_block_hash: parent_block_hash = NONE_HASH`
  > `return BlockHash(hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys)))`
  `vllm/config/cache.py:254-256`：`enable_prefix_caching: bool = True`；`prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"`
  `docs/design/prefix_caching.md:5-31`：components = 父哈希 + block tokens + extra hashes（LoRA ID、多模态 hash、cache salt）；只缓存整块；"As of v0.11, the default hashing algorithm is `sha256`"；可选 `sha256_cbor` / `xxhash` / `xxhash_cbor`
- 等级：A（代码）/ B（默认算法版本）
- 意味着什么：与 DeepInfer 的"KV cache 系统未开源"形成直接对照——前缀缓存是 vLLM 的开箱能力，且是跨请求、跨实例（配合 KV connector）都可用的。
- 疑点：`sha256` 默认走 pickle，官方明确说"may not be reproducible across different Python or vLLM versions"；要跨环境可复现得用 `sha256_cbor`。

### #11 混合 KV：单内存池 + 多种 KV spec（Full / SWA / ChunkedLocal / Circular / MLA / Mamba / HiSparse）
- 证据：`vllm/v1/kv_cache_interface.py` 类清单
  > `:604 FullAttentionSpec`、`:714 MLAAttentionSpec`、`:801 RSWASpec`、`:846 ChunkedLocalAttentionSpec`、`:886 SlidingWindowSpec`、`:953 CircularBufferSpec`、`:471 HiSparseHotSpec`、`:505 HiSparseResidentSpec`、以及 `MambaSpec`
  `:886-905`
  > `class SlidingWindowSpec(AttentionSpec): sliding_window: int`
  > `extra_retained_tokens: int = 0`（"needed for multi-module spec decoding which can re-prefill the last num_spec_prefill_tokens - 1 tokens"）
  `docs/design/hybrid_kv_cache_manager.md:5-25`：一个内存池，按 attention 类型给不同层分配不同 block 数；full attention 保留全部 token，SWA 只保留最近 `sliding_window_size`；前缀命中规则也按层类型区分
- 等级：A（代码）/ B（设计文档）
- 意味着什么：V4.1-Flash 这类"全局压缩 KV + 40 层 SWA"的混合模型能直接跑，靠的就是这套 per-layer KV spec 与统一 block pool；DeepInfer 用 YOCO/SWA bounded replay 处理同一问题，vLLM 用 KV spec + 前缀缓存边界处理。
- 疑点：`hybrid_kv_cache_manager.md` 开头仍写"feature is still in its early stage"（基于 commit 458e74），与代码里已支持 7+ 种 spec 的成熟度不符，属文档滞后。

### #12 KV Connector：9 类连接器，NIXL 是主力，已有 push 模式
- 证据：`docs/features/disagg_prefill.md:19-60`
  > "Now supports 9 types of connectors:" ExampleConnector / LMCacheConnectorV1 / NixlConnector / MooncakeConnector / MoRIIOConnector (ROCm only) / MultiConnector / OffloadingConnector / FlexKVConnectorV1（正文列 8 类 + LMCache MP 变体）
  `vllm/distributed/kv_transfer/kv_connector/v1/` 目录含 `nixl/`、`mooncake/`、`moriio/`、`lmcache_integration/`、`hf3fs/`、`offloading/`、`hisparse/`、`umbp/`、`multi_connector.py`、`flexkv_connector.py`
  `docs/design/nixl_kv_push_connector.md:3-6`
  > "The default NIXL connector is **pull-based**: the decode (D) instance reads KV blocks from the prefill (P) instance via `NIXL READ` ... `NixlPushConnector` adds a **push-based** alternative in which P writes the KV blocks directly into D's pre-allocated memory via `NIXL WRITE`."
- 等级：A（目录）/ B（文档）
- 意味着什么：KV 传输层是 vLLM 相对 DeepInfer（开源版明确不含 KV cache 系统与 P/D 传输）最明显的"整套 vs 缺件"差距；生产配方里 prefill 侧用 `MultiConnector`（NIXL push + SimpleCPUOffload），decode 侧只用 NIXL consumer（`recipes/.../DeepSeek-V4.1-Flash.yaml:1072-1080`）。
- 疑点：文档说"9 类"但正文只列 8 个条目（含 LMCache MP 变体算一类）；`FlexKV` 与 `umbp` 的定位未查。

### #13 并行：TP 走 Megatron 式、PP 按层切、默认 mp/Ray 两种 executor
- 证据：`docs/serving/parallelism_scaling.md:5-12, 31-33`
  > "The implementation includes Megatron-LM's tensor parallel algorithm."
  > "The default distributed runtimes are Ray for multi-node inference and native Python `multiprocessing` for single-node inference."
  `vllm/config/vllm.py:588-591`：PP 需要 `pp_size` 个并发 batch 填流水
- 等级：B
- 意味着什么：TP/PP 是"单副本内"切分；PP 在 V1 runner + 异步下有明确限制（`vllm/config/vllm.py:1646` 直接报错），MRV2 才放开。
- 疑点：PP 的实际吞吐收益与 bubble 处理未在文档给出数字。

### #14 DP-Attention：每个 DP rank 一个 EngineCore，可与 TP/EP 组合
- 证据：`docs/serving/data_parallel_deployment.md:13`
  > "In vLLM, each DP rank is deployed as a separate 'core engine' process that communicates with front-end process(es) via ZMQ sockets. Data Parallel attention can be combined with Tensor Parallel attention, in which case each DP engine owns a number of per-GPU worker processes equal to the configured TP size."
  `docs/serving/parallelism_scaling.md:27`
  > "vLLM supports large-scale deployment combining Data Parallel attention with Expert or Tensor Parallel MoE layers."
- 等级：B
- 意味着什么：MoE 模型常用 "Attention DP + MoE EP"：attention 按请求分到 DP rank，专家按 EP 切。DeepInfer 的 `--ep 2 --dp 8` 是同一思路，只是它把 attention DP 与 balanced CP 并列。
- 疑点：DP 之间的 dummy run / 同步开销由 `dp_sync_interval`（默认 16，`vllm/config/parallel.py:163`）控制，实际影响未测。

### #15 EP 与 all2all 后端：DeepEP 的 HT/LL 两套 kernel 都在
- 证据：`vllm/config/parallel.py:190, 220-232`
  > `enable_expert_parallel: bool = False` / "Use expert parallelism instead of tensor parallelism for MoE layers."
  > `all2all_backend: All2AllBackend = "allgather_reducescatter"`，可选 `deepep_high_throughput`、`deepep_low_latency`、`mori_high_throughput`、`mori_low_latency`、`moonep`、`nixl_ep`、`flashinfer_nvlink_one_sided/two_sided`、`passthrough`
  `vllm/distributed/device_communicators/all2all.py:203-209` `class DeepEPAll2AllManagerBase` — "All2All communication based on DeepEP High-Throughput kernels."
- 等级：A
- 意味着什么：vLLM 直接复用 DeepEP（DeepSeek 开源通信库），所以"两套引擎的 MoE 通信 kernel 基本相同"是 vLLM 敢说 6% 的技术前提之一。
- 疑点：`all2all_backend` 默认是 `allgather_reducescatter`，不是 DeepEP；要用 DeepEP HT/LL 必须显式指定。V4.1 配方里**没有**设 `--all2all-backend`（`grep deepep` 无命中），即走默认；`passthrough` 只自动绑定给 FlashInfer MoE-EP megakernel（`vllm/config/kernel.py:194-215`），不适用于 `deep_gemm_mega_moe`。因此 vLLM 侧的 EP 通信是否与 DeepInfer 完全一致，未能确认。

### #16 EPLB：运行时统计 + 周期性重排 + 冗余专家，异步执行
- 证据：`docs/serving/expert_parallel_deployment.md:137-180`
  > "When enabled, vLLM collects load statistics with every forward pass and periodically rebalances expert distribution."
  > 参数表：`window_size` 默认 1000、`step_interval` 默认 3000、`num_redundant_experts` 默认 0、`use_async` 默认 true、`policy` 默认 `"default"`、`communicator` 可选 `torch_nccl`/`torch_gloo`/`pynccl`/`nixl`/null
  > 内存开销公式与例子："For DeepSeekV3, this is approximately `2.4 GB` for one redundant expert per EP rank."
  代码：`vllm/distributed/eplb/eplb_state.py:234 class EplbState`、`rebalance_execute.py`、`async_worker.py`、`policy/`
- 等级：B（文档）/ A（代码）
- 意味着什么：EPLB 是"Wide-EP 能跑满"的关键；DeepInfer 默认 32 个冗余专家（`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`），vLLM 默认 0、要显式开，配方里 GB300 DEP16 用 16、DEP32 用 32（`recipes/.../DeepSeek-V4.1-Flash.yaml:1060`）。
- 疑点：文档的 communicator 列表没有 `symm_mem`，但配方用了 `{"communicator":"symm_mem"}`（`:1084`），文档滞后。

### #17 Wide-EP：不是独立开关，而是"一 pod 一 rank + 外部 LB"的部署形态
- 证据：`vllm/config/parallel.py:169-176`
  > `data_parallel_external_lb: bool = False`
  > `"""Whether to use "external" DP LB mode. ... This is useful for a "one-pod-per-rank" wide-EP setup in Kubernetes. Supported only for MoE deployments ..."""`
  另有 `data_parallel_hybrid_lb`（每节点 AsyncLLM + 本地 DP LB，节点间外部 LB，`:177-186`）
  `docs/serving/expert_parallel_deployment.md:83-128` 多机 EP 部署（headless 从节点）
- 等级：A
- 意味着什么：vLLM 的 Wide-EP = 多节点 DP+EP + 外部路由（vllm-router/llm-d），没有单独 `--wide-ep` 参数；这解释了为什么 6% 的对比必须限定在 64 卡级拓扑——单机 8 卡上这套形态用不上。官方博客 2026-02-03《vLLM WideEP ... on NVIDIA GB200》已把 WideEP 作为独立主题写过（B 级，`vllm.ai/blog` 索引可见）。
- 疑点：仓库内没有"Wide-EP"专门文档；该词只在 `moriio_common.py:252` 等注释和 X 推文里出现。

### #18 投机解码方法清单：EAGLE 系 + MTP 系 + DFlash + DSpark + ngram/suffix/draft_model
- 证据：`vllm/config/speculative.py:74-82`
  > `SpeculativeMethod = Literal["ngram","draft_model","suffix","custom_class", EagleModelTypes, NgramGPUTypes, DSparkModelTypes]`
  `:37-67` `MTPModelTypes` 含 `"mtp"`、`deepseek_mtp`、`qwen3_next_mtp`、`kimi_k3_mtp`、`glm5_next_mtp` 等 28 种；`EagleModelTypes = Literal["eagle","eagle3","extract_hidden_states", MTPModelTypes, DFlashModelTypes]`
  `docs/features/speculative_decoding/README.md:9-23` 用户可见清单：EAGLE、LiLiCorr、MTP、Draft Model、PARD、N-Gram、Suffix、Hidden State Extraction、Custom Proposer、Dynamic SD、Adaptive Verification、FLy、Per-Request Acceptance Metrics
- 等级：A（枚举）/ B（文档）
- 意味着什么：vLLM 的投机解码是"全家桶"，DeepInfer 只有一个 DSpark；但 V4.1 这个 checkpoint 上 vLLM 同样只用 DSpark（见 #26）。
- 疑点：18 种 MTP 变体多为模型专属；`step3p5`、`gemma4`、`inkling_mtp` 等未逐一核对实现是否共用 `mtp` 路径。

### #19 变长验证：有，且是"跨请求全局预算"式的变长，不只按 batch size 调 K
- 证据：`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:39-46`
  > `def resolve_adaptive_cudagraph_mode(mode, *, piecewise_capture_available):`
  > `"""Select a separate decode route for variable-length verification."""`
  `:48-70`
  > `def _assign_draft_token_budget(confidence_probs, idx_mapping, capacities, draft_budget, num_steps):`
  > `"""Admit the globally best `draft_budget` draft slots, in place. Every (request, step) slot is scored by its survival probability ... and the highest scores win."""`
  `docs/features/speculative_decoding/adaptive_verification.md:9-17`
  > "Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability* ... and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests ..."
  > "The budget itself comes from a cost model profiled at startup."
- 等级：A（代码）/ B（文档）
- 意味着什么：这正是 DeepInfer `DEEPINFER_ENABLE_MTP_VARLEN=1`（按置信度安排验证）的对应物，且 vLLM 版本支持 EAGLE/MTP/DFlash/DSpark/draft_model 五类 drafter（文档 Support 段），不只 DSpark。V4.1 配方在 NVIDIA 上默认开（`recipes/...:74-79`）。
- 疑点：ROCm 上不可用（配方 `:80-93` 明确列出两个失败检查：`supports_device_cpu_query_lens_mismatch()` 为 False、`DeepseekV41ROCMAiterSparseSWABackend` 只报 `UNIFORM_BATCH`）。

### #20 粗粒度"动态 K"：按并发区间切 `num_speculative_tokens`
- 证据：`docs/features/speculative_decoding/dynamic_speculative_decoding.md:9-24`
  > "add `num_speculative_tokens_per_batch_size` ... an entry is `[start_bs, end_bs, optimal_K]`"
  > 例子：`[[1,64,3],[65,128,1],[129,512,0]]`
- 等级：B
- 意味着什么：这是"按负载调 K"，粒度是并发区间，不是每请求；与 #19 的每 (request, position) 置信度打分不是一回事。书里若写"vLLM 支持变长验证"，应指 #19。
- 疑点：`num_speculative_tokens_per_batch_size` 与 adaptive verification 同时开时的优先级未查。

### #21 结构化输出：4 个后端，一个引擎只能选一个
- 证据：`vllm/config/structured_outputs.py:12, 20`
  > `"auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"`
  > `backend: StructuredOutputsBackend = "auto"`
  `vllm/v1/structured_output/__init__.py:154-186`
  > `# NOTE: We only support a single backend. We do NOT support different backends on a per-request basis in V1 (for now, anyway...)`
  > `if request_backend == "xgrammar": self.backend = XgrammarBackend(...)` / `elif "guidance"` / `elif "outlines"` / `elif "lm-format-enforcer"`
  `docs/features/structured_outputs.md:4-5, 36-38`：xgrammar 或 guidance 为文档主推；`auto` 会"try to choose an appropriate backend based on the details of the request"
- 等级：A（代码）/ B（文档）
- 意味着什么：DeepInfer 的 `so_args={"schema_type": ..., "schema": ...}` 是自有 API；vLLM 走 OpenAI 风格 `response_format`/`structured_outputs`，正则语法随 backend 不同（xgrammar/guidance/outlines 用 Rust 风格，lm-format-enforcer 用 Python `re`，`docs/features/structured_outputs.md:64`）。
- 疑点：`auto` 的具体判定逻辑在 `_validate_structured_output` 里，未逐行追；`outlines` 与 `guidance` 是否仍被积极维护未查。

### #22 批不变：`VLLM_BATCH_INVARIANT=1`，beta，硬件受限
- 证据：`docs/features/batch_invariance.md:3-4`
  > "Batch invariance is currently in beta. Some features are still under active development. Track progress ... at <https://github.com/vllm-project/vllm/issues/27433>"
  `:19-28` 支持平台：NVIDIA compute capability ≥ 8.0；AMD gfx90a/gfx942/gfx950；Intel XPU（需 Triton attention）
  `:152-160` 实现要点：确定性 kernel、固定 reduction 顺序的自定义 all-reduce（1-stage kernel 固定、大输入按固定 chunk 规约）、关闭 FlashInfer/AITER/QuickReduce all-reduce、对调优过的 matmul 用 breakable CUDA graphs 跟随运行时 batch、LoRA shrink 拆 8 个固定 K chunk
  环境变量定义：`vllm/envs.py:93, 637`
- 等级：A（环境变量）/ B（文档）
- 意味着什么：与 DeepInfer `DEEPINFER_BATCH_INVARIANT=1`（仅 NVIDIA）同级；vLLM 覆盖面更广（含 AMD/XPU），但自述 beta。
- 疑点：`docs/features/batch_invariance.md:152+` 的 kernel 级做法是否覆盖 V4.1 的所有自定义 kernel（MegaAttention/Mega-mHC 等）未确认——文档的 tested models 列表里没有 V4.1。

### #23 确定性推理：默认不保证；靠 seed + 批不变 + 关多进程
- 证据：`docs/usage/reproducibility.md:3-6`
  > "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance."
  > "In offline mode, you can either set `VLLM_ENABLE_V1_MULTIPROCESSING=0` which makes scheduling deterministic, or enable batch invariance ... In online mode, you can only enable batch invariance."
  `:20-26`
  > "In V1, the `seed` parameter defaults to `0` ... It is impossible to un-specify a seed for V1 because different workers need to sample the same outputs for workflows such as speculative decoding."
  `vllm/sampling_params.py:604-612`
  > `if envs.VLLM_BATCH_INVARIANT and self.temperature >= _SAMPLING_EPS and self.seed is None: logger.warning_once("Random sampling without an explicit seed may not be batch invariant. ...")`
- 等级：A（代码/警告）/ B（文档）
- 意味着什么：DeepInfer 的"确定性采样 = BATCH_INVARIANT=1 + MTP_VARLEN=0 + 固定 seq_seed"在 vLLM 的对应写法是 `VLLM_BATCH_INVARIANT=1` + 固定 `seed`（可选 `VLLM_ENABLE_V1_MULTIPROCESSING=0`）。注意 vLLM 的 `seed` 默认 0，但**仅 seed 不足以保证批不变**。
- 疑点：工具调用/结构化输出在批不变模式下的行为未测；多进程关闭只对 offline 有效。

### #24 P/D 分离：官方仍标 experimental，连接器最多、但"不提升吞吐"
- 证据：`docs/features/disagg_prefill.md:1, 6`
  > "# Disaggregated Prefilling (experimental)"
  > "This feature is experimental and subject to change."
  `:16`
  > "Disaggregated prefill DOES NOT improve throughput."
  理由（`:9-13`）：分开调 TTFT/ITL、控制尾巴 ITL；"Chunked prefill with a proper chunk size also can achieve the same goal, but in practice it's hard to figure out the correct chunk size value."
- 等级：B
- 意味着什么：DeepInfer 说"P/D 分离的大部分引擎功能已实现，但未含跨机传输"；vLLM 反过来——传输与连接器齐全（NIXL/Mooncake/LMCache/FlexKV…），但官方对收益的表述很克制（不提升吞吐，只调延迟结构）。
- 疑点：`experimental` 标签与配方里已给出 GB200 1P1D / GB300 DEP4+DEP16 的"verified"配置并存，成熟度介于两者之间。

### #25 硬件覆盖：CUDA / ROCm / TPU / XPU / CPU + 一堆 OOT 插件
- 证据：`docs/usage/v1_guide.md:84-95` 表格：NVIDIA 🟢、AMD 🟢、INTEL GPU 🟢、TPU 🟢、CPU 🟢；插件：vllm-ascend、vllm-spyre、vllm-gaudi、vllm-openvino
  `vllm/platforms/` 只有 `cpu.py`、`cuda.py`、`rocm.py`、`tpu.py`、`xpu.py`、`zen_cpu.py` —— 昇腾不在树内
  `docs/README.md:64`：插件还列 Google TPUs、Intel Gaudi、IBM Spyre、Huawei Ascend、Rebellions NPU、Apple Silicon、MetaX
- 等级：A（目录）/ B（文档）
- 意味着什么：vLLM 对昇腾是"官方插件"而非"内置后端"；这与 DeepInfer 把 Ascend 950 作为一等公民（`DEEPINFER_ACCELERATOR=ascend`、`vendor/DeepGEMM-Ascend`、`DeepEP-Ascend`）不同。
- 疑点：vllm-ascend 与本仓库的版本对应关系（vLLM 0.32 ↔ vllm-ascend 哪个版本）未确认。

### #26 昇腾上的 P/D：由 vllm-ascend + Mooncake 提供，不在本仓库
- 证据：vllm-ascend 文档（B 级，非本仓库）
  > "vLLM-Ascend now supports prefill-decode (PD) disaggregation."（`docs.vllm.ai/projects/ascend/.../pd_disaggregation_mooncake_single_node.html`）
  > 支持 `MooncakeConnector`（D 拉 P）、`MooncakeLayerwiseConnector`（P 逐层推 D）、`MooncakeD2RHConnectorV1`（先拉到 host 再上卡）（`.../developer_guide/Design_Documents/disaggregated_prefill.html`）
  > 单机 1P1D 示例用 vLLM-Ascend v0.23.0（with vLLM v0.23.0）
- 等级：B（外部官方文档）；本仓库内**无**昇腾 P/D 代码（A 级否证：`vllm/platforms/` 无 ascend）
- 意味着什么：写"vLLM 在昇腾上支持 P/D"必须限定为"通过 vllm-ascend 插件 + Mooncake"，不能说成主仓能力；vLLM 主仓的 NIXL 路径在昇腾上不可用。
- 疑点：vllm-ascend 文档版本号（v0.23.0）与本地 vLLM（≈0.32）差距很大，说明该文档页可能是旧版或版本号体系不同——未能确认对应关系。

### #27 V4.1-Flash 配方：min vLLM 0.31.0，架构 0.30.0 就有，2026-10-21 更新
- 证据：`recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml:6-8, 41`
  > `description: "DeepSeek V4.1 Flash vision-language MoE (552B backbone; 8B acti..."`
  > `date_added: 2026-09-09`
  > `date_updated: 2026-10-21`
  > `min_vllm_version: "0.31.0"`
  （配方 `:36`、`:693` 另称架构在 0.30.0 已 ship，PR #56228）
- 等级：B（配方是官方维护的文档型资产）
- 意味着什么：vLLM 对 V4.1 的支持是 day-0 级别（模型 2026-09-09 发布，同日有配方）；DeepInfer 10-20 开源时，vLLM 已经迭代了 6 周。
- 疑点：`date_updated: 2026-10-21` 恰在 DeepInfer 开源次日，推测这次更新加入了 Wide-EP/EPLB 的 parity 配置，但浅克隆无法 diff，标 E。

### #28 V4.1 的投机解码：只有 DSpark，5 token/轮，NVIDIA 上开自适应验证
- 证据：`recipes/.../DeepSeek-V4.1-Flash.yaml:613-615`
  > "DSpark is the only speculative method for this checkpoint. V4.1 dropped the MTP module that V3 and V4 trained alongside the backbone. **Speculative decoding** enables a 5-token block. On NVIDIA it also turns on adaptive verification."
  `:74-79` 生成参数
  > `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}`
  `:89-93` AMD 覆盖：`enable_adaptive_verification:false`，验证整块 5 token
- 等级：B
- 意味着什么：vLLM 的 MTP 全家桶在 V4.1 上用不上；"vLLM 支持 MTP"与"V4.1 用 MTP"是两件事。DeepInfer 的 `DEEPINFER_USE_NEXTN_DECODE` 默认 5，与 vLLM 的 5-token block 对齐。
- 疑点：配方说 "rejection_sample_method: block"（Sun et al. 的块验证），与 DeepInfer 的 per-token 置信度验证不同——这可能是两边 acceptance 曲线不完全可比的一个原因（E）。

### #29 V4.1 集成的 DeepSeek 内核：MegaAttention / MegaMoE / Mega-mHC / Mega-Gate / DeepSelect / DeepEP / FlashMLA
- 证据：`recipes/.../DeepSeek-V4.1-Flash.yaml:353, 384, 434`
  > `'{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}'`
  > `'{"moe_backend":"deep_gemm_mega_moe"}'`
  > `'{"moe_backend":"deep_gemm_mega_moe","sparse_indexer_topk_backend":"deep_select"}'`
  `vllm/config/kernel.py:126-135, 150, 309-321`：`"deep_gemm"`、`"deep_gemm_mega_moe"`、`"flashinfer_moe_ep_mega_deep_gemm"`，并注明 "DeepGEMM consumes the DeepSeek-V4 MXFP4 checkpoint recipe"
  `vllm/models/deepseek_v41/attention.py:665` 使用 `FLASHMLA_MEGA_ATTN_DSV41`；`vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py:34` 有 `FLASHINFER_MLA_SPARSE_DSV41`
  博客（2026-10-07）："We integrated DeepSeek's newly released kernels, including MegaAttention, Mega-mHC, Mega-Gate, and DeepSelect." 并称 SWA bounded replay + CUDA graphs 带来 ~30% TTFT 降低
  tracking issue #57448 列出：MegaAttention (#56935)、sparse logits indexer (#56254)、DeepSelect (#56464)、Mega mHC (#56962)、Mega gate (#56266)、SWA bounded replay encoder (#56227)/decoder (#56752)
- 等级：A（代码/配方）/ B（博客、issue）
- 意味着什么：vLLM 集成了 DeepSeek 的**内核库**（FlashMLA/DeepGEMM/DeepEP/DeepSelect），但**没有**集成 DeepInfer 的调度器、KV 系统、P/D 实现——这是理解"6%"的关键：同内核、不同外壳。
- 疑点：`deep_ep` 是否对 V4.1 默认启用未在配方里确认（配方用的是 `moe_backend: deep_gemm_mega_moe`，EP 通信后端另配）。

### #30 V4.1 的 KV 形态：`fp8_ds_mla` + indexer `mxfp4`，全局 KV 约 890 B/token
- 证据：`vllm/config/cache.py:52` `"fp8_ds_mla"` 在合法 KV dtype 列表
  `recipes/.../DeepSeek-V4.1-Flash.yaml:289, 309`
  > `'{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
  > `'{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
  博客（2026-10-07）："It combines several techniques to shrink the KV cache size: Compressed Sparse Attention 2 (CSA2), FP4 KV cache, and inter-layer KV cache sharing, pushing the global KV footprint to 890 bytes per token."；"Sliding-window (SWA) KV is uncompressed in FP8, covering the last 128 positions in each of the 40 layers."
- 等级：A（dtype）/ B（890 B/token 来自官方博客转述技术报告）
- 意味着什么：V4.1 的 KV 是"全局压缩 FP4 + 40 层 SWA FP8"两套并存；vLLM 用 #11 的混合 KV spec 承载，DeepInfer 用 YOCO bounded replay 承载 SWA 的重算代价。
- 疑点：890 B/token 是模型层数字（技术报告），不是 vLLM 实测；DeepInfer 的 ROCm PR 提到其 cache 有 "FP8 (528 B) / FP4 (288 B)" 两种形态（单位与是否 per-token/per-layer 未在摘要中说明），与 vLLM `fp8_ds_mla` 的对应关系需 R06 复核。

### #31 V4.1 的 P/D 验证配置：GB200 1P1D（TP4+NIXL）、GB300 DEP4+DEP16
- 证据：`recipes/.../DeepSeek-V4.1-Flash.yaml:633-640`
  > "The **Prefill/Decode Disaggregation** strategy is the verified 1P1D layout on GB200 NVL4: one tray (4 GPUs) per role, TP4 in each pool, KV handed over through NIXL, fronted by `vllm-router --vllm-pd-disaggregation`."
  > "Both pools disable FlashInfer autotune plus JIT and CuTeDSL warmup ... and cap `--max-num-seqs` at 32. On 8-GPU nodes the same layout becomes TP8 per role."
  `:422-423`
  > `# Verified 1P1D on GB200 NVL4: one tray per role, TP4 each, NIXL transfer.`
  > `# GB300: DEP4 prefill + DEP16 decode (SemiAnalysisAI/InferenceX#3941).`
  `:1053-1084`：B300 用 NIXL `LIBFABRIC` over EFA，GB300 用 `UCX` over MNNVL；prefill 侧 `MultiConnector`(NIXL push + SimpleCPUOffload，host KV 70–80 GiB/rank)，decode 侧 `NixlConnector` consumer + `--enable-eplb --eplb-config {"communicator":"symm_mem","num_redundant_experts":16,...}`
- 等级：B
- 意味着什么：这是"vLLM 能复现 DeepInfer"的具体部署形态：不是单机，而是多托盘 P/D + NIXL + EPLB。也说明 6% 的对比只在这种拓扑下成立。
- 疑点：`verified` 的主体是配方作者（vLLM/Inferact），不是独立第三方；具体吞吐数字未写进配方。

### #32 "6%" 的原始出处：vLLM 官方 X，不是博客
- 证据：`https://x.com/vllm_project/status/2112703311933747347`（2026-10-21T00:31:36Z 发布，X 页面显示 "8:31 PM · Oct 20, 2026" 美西时间）
  > "vLLM is all you need to reproduce DeepSeek's performance."
  > "With Wide-EP and EPLB, vLLM can match DeepInfer's decode performance within 6% and generate $39/gpu/hr in revenue at DeepSeek's off-peak pricing, at 200 P90 TPS on the AgentX workload."
  > "Under DeepSeek's decode configuration (DeepSeek-V4.1-Flash, 131K context, nextn=3), vLLM on GB200 achieves throughput within 6% of DeepInfer's published results across batch sizes 8–256."
  跟帖 1：`https://x.com/vllm_project/status/2112703314026709282`
  > "First, we compared vLLM's performance with DeepSeek's published decode throughput across batch size 8-256. We find that vLLM matches the performance within 6%. This is because vLLM has continuously integrated DeepSeek's kernel libraries (MegaAttn, MegaMoE, Mega-mHC, etc) ... Since the kernels are mostly the same, the performance of the two engines roughly match."
- 等级：B（官方账号一手发言）
- 意味着什么：书里若写"vLLM 官方博客声称"，属于**出处错误**；应写"vLLM 官方 X 账号 2026-10-21 的推文"。核对 `vllm.ai/blog` 索引（已含 2026-10-21 的 Flint、Humming 两篇）：`DeepInfer` 命中 0 次、`6%` 命中 0 次，说明没有对应博客文章。
- 疑点：推文没有给论文/博客链接，只给 InferenceX 的交互式面板；"within 6%"未说明是 vLLM 低 6% 还是高 6%（E：从"match ... within"语气看，应是 vLLM 略低、差距 ≤6%）。

### #33 6% 的对照条件：DeepInfer 公布曲线 = 64×SM100 / EP64DP64 / 131,072 / nextn=3 / acceptance 75%
- 证据：`DeepInfer/doc/images/decode_pareto.png` 图内文字（tesseract OCR 可直接读出）
  > `EP64DP64 | 384+64 experts | context 131,072 | nextn=3 fixed | per-token acceptance 75% → τ=3.25`
  > 纵轴 `Output Throughput (tokens/s/GPU)`，横轴 `Per-User Speed (tokens/s/user)`，图注 `Point labels: decode_num_requests (requests per DP rank)`
  生成该副标题的代码：`DeepInfer/examples/sweep/plot_pareto.py:85-93`
  > `f"EP{head['ep_size']}DP{head['dp_size']}{experts} | context {head['context_length']:,} | nextn={head['nextn']} fixed | per-token acceptance {draft_acceptance:.0%} → τ={tau:.2f}"`
  复现命令：`DeepInfer/examples/README.md:105`
  > `bash examples/sweep/run_sweep.sh decode ce-128k 131072 3 0.75`（注释：sweep `decode_num_requests` 8…256, nextn=3, plot scaled by a draft acceptance rate of 0.75）
  `DeepInfer/README.md:83`
  > "Measured on 4 SM100 GPUs for prefill and 64 for decode — each within one scale-up domain."
  `DeepInfer/examples/sweep/run_sweep.sh:30-39`：decode sweep 的 batch 值就是 `(8 16 32 64 128 256)`，参数为 `<context_length> <nextn> <draft_acceptance>`，并固定 `--non_cached_tokens 4096`
  EP64DP64 即 64 个 DP rank × 1 卡（TP=1），与 README 的"64 GPUs for decode"一致（E 级算术）
- 等级：A（图与脚本在 DeepInfer 仓库内）
- 意味着什么："batch sizes 8–256"不是随便选的，正是 DeepInfer sweep 的六个点；"131K / nextn=3" 也是 sweep 的入参。vLLM 是照着 DeepInfer 的复现脚本参数去对标的。
- 疑点：acceptance 是**设定值**（75%），不是真实流量；两边若 rejection 策略不同（vLLM 用 block 验证，DeepInfer 用 per-token 置信度），实际 acceptance 会有差异。

### #34 6% 的方法学缺口：跨集群、GPU 数未公布、方向未说明
- 证据：推文全文（#32）只写 "vLLM on GB200"，未给 GPU 数量、TP/EP/DP 配置、是否 64 卡；对照对象是 DeepInfer "published results"（64×SM100 曲线）
- 等级：E（推断，基于推文缺失信息）
- 意味着什么：严格说这是"在 GB200 上用 DeepSeek 的 decode 配置复现出与 DeepInfer 公布曲线相差 ≤6% 的吞吐"，不是"同一集群 A/B"。书里引用时必须带这四个条件：GB200、131K、nextn=3、batch 8–256，并注明"对公布数字"。
- 疑点：DeepInfer 侧是 64 卡、TP=1、EP=64、DP=64；vLLM 若用 TP4/TP8 的托盘或不同 EP 规模，per-GPU 吞吐的可比性要打问号（EP 规模不同，专家并行通信开销不同）。这是最需要向 vLLM/Inferact 确认的一点。

### #35 第三方单机实测与 6% 不矛盾，但结论方向相反
- 证据（仅读摘要，未逐段核对方法）：
  > dev.to / warmctl《DeepSeek V4.1 Flash on 8×B300: vLLM vs DeepInfer》：单节点 8×B300 上 "vLLM is 2.0× faster on decode (1K → 1K), 4.3× at 8K → 1K"
  > GPUStack《DeepInfer vs. vLLM on 8×B200》：DeepInfer 无 TP 路径，只能配 CP1/DP8/EP8 来对 vLLM 的 TP
- 等级：C
- 意味着什么：单机 8 卡上 vLLM 反而大幅领先，因为 DeepInfer 的 Wide-EP 形态在 8 卡上施展不开、且没有 TP。这反向印证 #17/#34：6% 是"64 卡 wide-EP 对 64 卡"的结论，不能外推到单机。
- 疑点：两篇均为第三方博客，硬件/负载/版本各异（vLLM 0.31 vs 本仓库 0.32+），不能作为定量依据，只能作为"条件敏感性"的旁证。

### #36 DeepInfer 开源后 vLLM 社区的回应：官方推文 + 一个 RFC/PR，尚无合入
- 证据：
  - 官方回应：2026-10-21 推文串（#32），末尾 "Thanks @deepseek_ai for releasing DeepInfer. Together with the community, we'll keep pushing the frontier of performance and cost."
  - RFC：`https://github.com/vllm-project/vllm/issues/62839`（2026-10-21 开，Open）——"[RFC]: Fuse DSpark confidence-head GEMV into a single TileLang kernel"，正文直接引用 DeepInfer 源码 `deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py:12-72` 与 `.../v4_modules/heads.py:37`，称 "DeepInfer (DeepSeek's own engine) already ships an equivalent fused kernel via `AcceptRatePredictor` and the same PDL trick; this is a port-with-FP32-weight."
  - PR：`https://github.com/vllm-project/vllm/pull/62840`（2026-10-21 开，Open，draft）——"Draft implementation. Hold merge until the RFC discussion converges."；29 tests pass；kernel 在 `vllm/model_executor/kernels/dspark/`，custom op `torch.ops.vllm.dspark_confidence_tilelang`；门控 `(n_embed, markov_rank) == (7168, 32)` 且 batch ∈ [32, 2048]，平台限 SM90+；实测 "1.92× on RTX 5090 (SM120) and 1.90× on H200 (SM90)"，但 "TPOT-level gain is < 0.05%"
  - 本仓库 `grep -ri deepinfer`（py/md/yaml/cu/h）**零命中**
- 等级：A（grep）/ B（issue、PR 正文）
- 意味着什么：vLLM 没有"合并 DeepInfer"这件事，也没有引入 DeepInfer 的调度/KV 代码；回应方式是（a）发布 parity 数据，（b）把 DeepInfer 里值得抄的小 kernel 逐个 port。这与 DeepInfer 的致谢"originally built primarily on vLLM"构成一个闭环：DeepInfer 抄 vLLM 的骨架，vLLM 再抄 DeepInfer 的算子。
- 疑点：#62840 是个人 contributor（panpan0000）发起，不是 vLLM 核心团队；"hold merge"意味着合入时间未知。另有搜索结果显示 issue #62724（deepseek-recipe 契约）与 PR #62667，但未逐条核实。

### #37 版本号与"从哪个版本开始"的可得性
- 证据：仓库无 tag、浅克隆 1 个提交；代码内弃用提示 "will be removed in v0.33.0"（`vllm/engine/arg_utils.py:457-470`、`vllm/config/model.py:318`）；`docs/features/structured_outputs.md:10` 提到 "removed in v0.12.0"；`docs/design/prefix_caching.md:26` 提到 "As of v0.11"
- 等级：A/E
- 意味着什么：本笔记只能可靠地说"当前 main ≈ 0.32/0.33-dev"；各特性"从哪个版本开始"需另查 release notes（本次未做）。
- 疑点：无法用 git blame 定位引入版本，是本次调研最大的方法学缺口。

## 3. 必答问题逐条回答

### 1. 引擎架构：V1 引擎（EngineCore、scheduler、worker 进程模型）
- V0 已删除，`vllm.engine.LLMEngine` 是 `vllm.v1.engine.llm_engine.LLMEngine` 的别名（#1）。
- 进程模型：API server（1 个，DP 时 = DP，可 `--api-server-count`）× EngineCore（每 DP rank 1 个）× GPU worker（每卡 1 个，= DP×PP×TP），DP>1 时 +1 个 DP coordinator；总数 `A + DP + N`（#2）。
- EngineCore 跑 `run_busy_loop()`：poll 输入队列 → `step_fn()` → 输出队列（#3）；`max_concurrent_batches` 决定同时有几个 batch 在飞（PP=pp_size，异步=2，MRV2=pp_size+1）（#3）。
- 调度器在 `vllm/v1/core/sched/scheduler.py`（167 KB），异步版 `async_scheduler.py`；worker 在 `vllm/v1/worker/gpu_worker.py` + `multiproc_executor.py`（#2、#7）。
- 额外：模型执行层默认已切 MRV2（#4），这是 2026 年的新变化，很多 2025 年的架构图不包含。

### 2. 调度机制：continuous batching、chunked prefill、异步调度
- continuous batching：调度器没有 prefill/decode 阶段，统一按 `num_computed_tokens` 追 `num_tokens_with_spec`（#5）。
- chunked prefill：默认开启，decode 优先，prefill 放不进 `max_num_batched_tokens` 就切片（#6）；encoder-decoder 模型禁用。
- 异步调度：`AsyncScheduler` 用 `-1` 占位 token 调度下一步（#7）；默认按兼容性自动开启（#8）。
- 成熟度：功能上默认开、有完整的不兼容矩阵与回退；但（a）文档仍有一处写 "experimental"（#8 疑点），（b）MRV2 自述未 feature-complete（#4 疑点），（c）V1 runner 下 PP + 异步直接报错（#8）。综合判断：**生产可用、仍在演进**，不是实验特性。

### 3. KV 管理：PagedAttention、前缀缓存、混合 KV/SWA、KV Connector/NIXL
- PagedAttention：概念仍是基础，但 `docs/design/paged_attention.md` 已声明是历史文档，实际 kernel 由 FlashAttention/FlashInfer/FlashMLA 等 backend 提供（#9）。
- 前缀缓存：默认开启，block hash = 父哈希 + 本块 token + extra keys，默认 `sha256`，可选 `sha256_cbor`/`xxhash`（#10）。
- 混合 KV：单内存池 + 多种 spec（Full/MLA/SWA/ChunkedLocal/Circular/HiSparse/Mamba），SWA 只保留窗口内 token，并支持 `extra_retained_tokens` 供多模块投机解码回填（#11）。
- KV Connector：NIXL（pull 默认 + push 可选）、Mooncake、LMCache、FlexKV、MoRIIO（ROCm）、Offloading、MultiConnector 等（#12）。
- 与 DeepInfer 的异同：DeepInfer 开源版**不含** KV cache 系统（README 明说），vLLM 这块是完整实现；DeepInfer 的 SWA 用 YOCO bounded replay 重算，vLLM 用混合 KV spec + 前缀缓存边界，思路不同（#11）。

### 4. 并行支持：TP、PP、DP-Attention、EP、EPLB、Wide-EP
- TP：Megatron 式，单机默认 mp、跨机默认 Ray（#13）。
- PP：按层切；V1 runner + 异步下不支持，MRV2 放开（#8、#13）。
- DP-Attention：每 DP rank 一个 EngineCore，可与 TP/EP 组合（#14）。
- EP：`--enable-expert-parallel`，all2all 后端含 DeepEP HT/LL、MoRI、MoonEP、NIXL-EP、FlashInfer NVLink 单/双向（#15）。
- EPLB：`--enable-eplb`，窗口统计 + 周期重排 + 冗余专家，默认异步执行（#16）。
- Wide-EP：无独立开关，靠 `--data-parallel-external-lb`（one-pod-per-rank）+ 多机 EP + 外部 router（#17）。
- 与 DeepInfer 的异同：DeepInfer 是 "Attention DP / balanced CP + MoE EP + EPLB"；vLLM 有 DP-Attention + EP + EPLB，CP 侧有 `prefill_context_parallel_size`（PCP，`vllm/config/parallel.py:144`）与 `decode_context_parallel_size`（DCP，`:389`），但 DCP 与 SWA 互斥（`vllm/v1/kv_cache_interface.py:935` `assert ... "DCP not support sliding window."`），对 V4.1 这种 SWA 模型意味着 DCP 不可用。

### 5. 投机解码：支持的方法、是否支持变长验证
- 方法：EAGLE/eagle3、MTP（28 种模型变体）、DFlash、DSpark、draft_model、PARD（parallel drafting）、ngram/ngram_gpu、suffix、custom_class、extract_hidden_states（#18）。
- 变长验证：**支持**。`enable_adaptive_verification` 按 (request, position) 的 survival probability 做全局 top-k 预算分配，配合 varlen decode CUDA graphs；支持 EAGLE/MTP/DFlash/DSpark/draft_model（#19）。另有粗粒度的 `num_speculative_tokens_per_batch_size`（#20）。
- V4.1 上：只用 DSpark（5 token/轮）+ NVIDIA 开 adaptive verification + block rejection（#28）。
- 限制：ROCm 上 adaptive verification 不可用（#19 疑点）；V1 runner 下部分投机方法不能与异步调度同开（#8）。

### 6. 结构化输出后端（xgrammar / outlines 等）
- 4 个后端：xgrammar、guidance（llguidance）、outlines、lm-format-enforcer；默认 `auto`；一个引擎只能用一个后端（#21）。
- 语法能力：JSON schema、regex、EBNF/CFG、structural tag（xgrammar 的 token-level 格式）；regex 方言随后端不同（#21）。
- 与 DeepInfer 的异同：DeepInfer 用自有 `so_args`，vLLM 走 OpenAI 兼容字段 + 后端选择；两者都支持 regex/JSON schema/EBNF（DeepInfer README 自述）。

### 7. 批不变与确定性推理支持情况
- 批不变：`VLLM_BATCH_INVARIANT=1`，beta；NVIDIA CC≥8.0 / AMD gfx90a-gfx950 / Intel XPU（需 Triton attention）；实现包括确定性 kernel、固定顺序 all-reduce、breakable CUDA graphs 等（#22）。
- 确定性：默认不保证；offline 可 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 或批不变；online 只能批不变；`seed` 默认 0，但随机采样不给 seed 时批不变会告警（#23）。
- 与 DeepInfer 的异同：DeepInfer 是 `BATCH_INVARIANT=1 + MTP_VARLEN=0 + 固定 seq_seed`（仅 NVIDIA）；vLLM 对应 `VLLM_BATCH_INVARIANT=1 + 固定 seed`，硬件面更广但自述 beta，且 tested models 列表里没有 V4.1（#22 疑点）。

### 8. P/D 分离方案与硬件覆盖范围（含昇腾插件）
- 方案：`--kv-transfer-config` 选连接器，P 侧 `kv_producer`、D 侧 `kv_consumer`、单机可 `kv_both`；官方仍标 experimental，且明确"不提升吞吐"（#24）。
- 连接器：NIXL（pull/push）、Mooncake（含 EC 变体）、LMCache（含 MP）、FlexKV、MoRIIO（仅 ROCm）、Offloading、MultiConnector、Example（#12、#24）。
- 硬件：NIXL 覆盖 CUDA + ROCm（`docs/features/nixl_connector_usage.md:11-18`）；MoRIIO 仅 ROCm；昇腾走 vllm-ascend + Mooncake（MooncakeConnector / MooncakeLayerwiseConnector / MooncakeD2RHConnectorV1），**不在主仓**（#25、#26）。
- 与 DeepInfer 的异同：DeepInfer README 明说"P/D 分离的大部分引擎功能已实现，但未含跨机传输与 DP 副本间负载均衡"；vLLM 是传输层齐全、官方对收益克制（#24）。

### 9. 对 DeepSeek-V4.1-Flash 的支持：recipes 与 DeepSeek 内核
- 配方：`recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`，min vLLM 0.31.0，date_added 2026-09-09，date_updated 2026-10-21（#27）。
- 模型代码：`vllm/models/deepseek_v41/`（`attention.py`、`sparse_mla.py`、`compressor.py`、`decoder_replay_layers.py`、`quant_config.py`，NVIDIA/AMD 子目录），注册名 `DeepseekV41ForCausalLM`（`vllm/model_executor/models/registry.py:384`）。
- 集成内核：FlashMLA（`FLASHMLA_MEGA_ATTN_DSV41`、`FLASHINFER_MLA_SPARSE_DSV41`）、DeepGEMM（`deep_gemm_mega_moe`、Mega-Gate、Mega-mHC）、DeepSelect（sparse indexer top-k）、DeepEP（可选 all2all 后端，配方未显式启用，见 #15 疑点）（#29）。
- 关键配置：DSpark 5-token + adaptive verification（NVIDIA）、`fp8_ds_mla` KV、indexer `mxfp4`、MRV2、Engram CPU offload（#28、#29、#30）。
- 与 DeepInfer 的异同：**同内核、不同外壳**——vLLM 用 DeepSeek 的算子库，但调度/KV/P/D 是自己的；DeepInfer 的引擎骨架来自 vLLM，但换成了自己的 ZOS/VE/P/D 设计。

### 10. "解码吞吐在 DeepInfer 6% 以内"的原始出处、硬件、负载、方法、前置条件
- 原始出处：vLLM 官方 X `@vllm_project`，2026-10-21T00:31:36Z（status 2112703311933747347），**不是博客**；vLLM 博客索引里没有对应文章（#32）。
- 硬件：vLLM 侧 "on GB200"（GPU 数未公布）；对照侧 DeepInfer 公布曲线是 64×SM100，EP64DP64（#33、#34）。
- 测试负载：DeepSeek 的 decode 配置 —— DeepSeek-V4.1-Flash、131K context、nextn=3、batch 8–256（每 DP rank 的 `decode_num_requests`）（#32、#33）。
- 方法：拿 vLLM 在同一组 batch 点上的 per-GPU 输出吞吐，对 DeepInfer 公布的 decode Pareto 曲线（`decode_pareto.png`，纵轴 tokens/s/GPU，横轴 tokens/s/user）逐点比，得到"within 6%"（#32、#33）。
- 前置条件：Wide-EP + EPLB；已集成 MegaAttn/MegaMoE/Mega-mHC 等 DeepSeek 内核；nextn=3（注意配方默认是 5）；acceptance 按 75% 设定（DeepInfer 侧）（#29、#32、#33）。
- 不能说的：不能说"同机 A/B 差 6%"，不能外推到单机 8 卡（第三方单机实测方向相反，#35），也不能说"vLLM 快 6%"（方向未说明，#32 疑点）。

### 11. DeepInfer 开源后 vLLM 社区的最新回应与合入情况
- 回应：开源后数小时发推文串，主张"vLLM 足以复现 DeepSeek 性能"，给出 6% / $39 per GPU-hour / 826–829 TPS per user 等数字（#32）。
- 合入：截至 2d40c06（2026-10-22），**没有任何**引用 DeepInfer 的代码或文档合入（grep 零命中）；唯一直接相关的是 RFC #62839 + draft PR #62840（2026-10-21），把 DeepInfer 的 DSpark confidence-head TileLang 融合 kernel port 到 vLLM，状态 Open、"hold merge"，29 tests pass，TPOT 收益 <0.05%（#36）。
- 间接合入：V4.1 的内核集成早在 9 月就开始（#57448 tracking issue：#56935/#56254/#56464/#56962/#56266/#56227/#56752），10-07 博客总结了 SWA bounded replay + 内核集成带来的 1.9×/5.3× 提升（#29）。
- 判断：vLLM 的路线是"不合并引擎、只吸收算子 + 用 parity 数据回应"，而不是"兼容 DeepInfer"（E，基于 grep 与 PR 状态）。

## 4. 新发现的问题

1. **6% 的 GPU 数缺失**：推文只说 "on GB200"，若 vLLM 只用了 8 卡而 DeepInfer 是 64 卡 EP64DP64，per-GPU 吞吐不可直接比。这是全书引用该数字前最需要向 vLLM/Inferact 确认的一点。
2. **nextn=3 vs 配方默认 5**：推文用 3 对齐 DeepInfer 的公布曲线，但 vLLM 配方默认 `num_speculative_tokens: 5`（训练块大小）。书里写"vLLM 默认 5"与"6% 用的是 3"要分开说。
3. **rejection 策略不同**：vLLM 用 `block` 验证，DeepInfer 用 per-token 置信度（`DEEPINFER_ENABLE_MTP_VARLEN`）。同样 75% acceptance 设定下，两边实际接受长度可能不同，6% 的可比性受影响（E）。
4. **文档漂移三处**：`paged_attention.md` 自认历史文档；`hybrid_kv_cache_manager.md` 说 "early stage"；`expert_parallel_deployment.md` 说 async scheduling "experimental"；`v1_guide.md` 说 "Concurrent Partial Prefills: In Progress"。书里引用文档时要标注"文档可能滞后于代码"。
5. **MRV2 默认开启但自述未完成**：这是一个"默认值领先于文档"的典型，读者若按 2025 年的 V1 架构理解会漏掉 MRV2。
6. **vLLM 的 EPLB 默认 0 冗余专家**（`num_redundant_experts` 默认 0），DeepInfer 侧由 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 控制（默认值未能确认）；同样的 EP 配置下 vLLM 需要显式调参才能达到 DeepInfer 的负载均衡水平（#16）。
7. **`grep -ri deepinfer` 零命中**：说明 vLLM 没有做任何"DeepInfer 兼容层"，书里不应暗示两者可以互换配置。
8. **昇腾侧信息只能靠 vllm-ascend 文档**，且其版本号（v0.23.0）与本地 vLLM（≈0.32）对不上，需 R12/R14 复核版本对应关系。

## 5. 可用于图表的材料

### 5.1 vLLM V1 进程模型（表）

| 进程 | 数量 | 职责 | 代码 |
|---|---|---|---|
| API server | `A`（默认 = DP） | HTTP、tokenization、多模态加载、流式返回 | `vllm/entrypoints/launchers/api_server` |
| EngineCore | `DP` | 调度、KV 管理、协调 worker | `vllm/v1/engine/core.py` |
| GPU worker | `DP × PP × TP` | 加载权重、forward、显存管理 | `vllm/v1/worker/gpu_worker.py` |
| DP coordinator | 1（DP>1） | DP 负载均衡、MoE 同步 forward | `vllm/v1/engine/coordinator.py` |

### 5.2 异步调度时序（可与 DeepInfer ZOS 图并列）

| 步 | CPU（scheduler） | GPU（worker） |
|---|---|---|
| N | 用上一步的占位 token 排 batch N+1（`spec_token_ids = [-1]*k`） | 执行 batch N |
| N+1 | 收到 batch N 的真实 token，回填 `num_output_placeholders` | 执行 batch N+1 |
| 代价 | 停止条件/结构化输出/KV 计数需为"未定 token"留占位 | — |

### 5.3 vLLM vs DeepInfer 逐维度对照（草稿）

| 维度 | vLLM（2d40c06） | DeepInfer（开源版） |
|---|---|---|
| 引擎骨架 | V1 EngineCore + 多进程 + ZMQ | C++ 调度（ZOS，`csrc/engine/zos.cpp`）+ Python 前端 |
| 调度 | 统一 token 预算，chunked prefill 默认开，异步默认开 | 零开销批调度器（ZOS）默认开 |
| KV | PagedAttention + 前缀缓存（sha256）+ 混合 KV spec | 开源版不含 KV cache 系统；SWA 用 YOCO bounded replay |
| KV 传输 | NIXL/Mooncake/LMCache/FlexKV/MoRIIO… | 未开源（README 明说） |
| 并行 | TP/PP/DP-Attention/EP/EPLB/Wide-EP | Attention DP / balanced CP + EP + EPLB |
| 投机 | EAGLE/MTP/DFlash/DSpark/ngram/suffix + adaptive verification | DSpark + confidence-scheduled verification |
| 批不变 | `VLLM_BATCH_INVARIANT=1`（beta，NVIDIA/AMD/XPU） | `DEEPINFER_BATCH_INVARIANT=1`（仅 NVIDIA） |
| 结构化输出 | xgrammar/guidance/outlines/lm-format-enforcer | 自有 `so_args`（regex/JSON schema/EBNF） |
| P/D | 连接器齐全，官方标 experimental | 引擎骨架有 PE/CE 区分，传输未实现 |
| 硬件 | CUDA/ROCm/TPU/XPU/CPU + 插件（Ascend 等） | SM100 + Ascend 950 |
| V4.1 | day-0 配方（2026-09-09），集成 DeepSeek 内核 | 唯一支持模型 |

### 5.4 6% 声明的条件卡（可直接做图注）

- 谁说的：vLLM 官方 X（@vllm_project），2026-10-21
- vLLM 侧：GB200，Wide-EP + EPLB，131K context，nextn=3，batch 8–256（GPU 数未公布）
- 对照侧：DeepInfer 公布曲线，64×SM100，EP64DP64（64 个 DP rank × TP1），384+64 experts，131,072，nextn=3，per-token acceptance 75% → τ=3.25
- 指标：per-GPU 输出吞吐（tokens/s/GPU）vs per-user speed（tokens/s/user）
- 结论口径：within 6%（方向未说明）

### 5.5 V4.1 配方关键参数（可做"一键复现"表）

| 项 | 值 | 出处 |
|---|---|---|
| min vLLM | 0.31.0 | 配方 `:41` |
| 投机 | `dspark`, 5 tokens, probabilistic, block rejection, adaptive verification (NVIDIA) | `:74-79` |
| Attention backend | `FLASHMLA_MEGA_ATTN_DSV41` / `FLASHINFER_MLA_SPARSE_DSV41` | `:353, 309` |
| MoE backend | `deep_gemm_mega_moe` + `sparse_indexer_topk_backend: deep_select` | `:384, 434` |
| KV dtype | `fp8_ds_mla`，indexer `mxfp4` | `:289`、`vllm/config/cache.py:52` |
| Model runner | `VLLM_USE_V2_MODEL_RUNNER=1`（Blackwell） | `:148, 340, 420, 442` |
| P/D（GB200） | 1P1D，每池 TP4，NIXL | `:633-640` |
| P/D（GB300） | DEP4 prefill + DEP16 decode | `:423` |
| EPLB（decode） | `{"communicator":"symm_mem","num_redundant_experts":16,"use_async":true}` | `:1084` |
