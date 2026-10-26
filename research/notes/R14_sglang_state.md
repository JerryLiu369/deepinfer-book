---
# R14 SGLang 现状：RadixAttention / HiCache / overlap 调度器 / DP-Attention / V4.1-Flash / 路由器

## 0. 元信息

- 代码版本：`sgl-project/sglang` @ `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`（main，2026-10-22 20:01:42 -0700，commit `[CI] [NVIDIA] gb300: Restore the aarch64 rust...`）
- 本地仓库是**浅克隆，只有 1 个提交**（`git log --oneline | wc -l` = 1），因此"从哪个版本开始有"这类问题无法用本地 git 历史回答；本笔记用 GitHub Release / PR 元数据补齐，凡属外部信息的都标 B 或 E
- 版本判断：当前 main 在 **v0.5.22（2026-10-19）之后**；本地安装文档仍 pin `v0.5.21`（`docs/docs/get-started/install.mdx:122`），V4.1 的 ROCm 镜像是 `v0.5.22-rocm10-mi35x-20261021`（`docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx:61`）。没有本地 tag/`_version.py` 实体，版本号标 E
- 查阅的主要来源：
  - 源码：`python/sglang/srt/entrypoints/engine.py`、`srt/managers/{scheduler.py,tp_worker.py,data_parallel_controller.py,overlap_utils.py,cache_controller.py}`、`srt/layers/dp_attention.py`、`srt/mem_cache/{radix_cache.py,unified_radix_cache.py,evict_policy.py,hicache_storage.py}`、`srt/mem_cache/unified_cache/*`、`srt/mem_cache/storage/*`、`srt/speculative/{spec_info.py,ragged_verify.py,dspark_components/*}`、`srt/models/deepseek_v4*.py`、`python/sglang/kernels/ops/attention/dsv4/*`、`sgl-router/`、`sgl-model-gateway/`
  - 文档：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`、`docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx`、`docs/src/snippets/agentx/deepseek-ai/deepseek-v4_1.jsx`、`docs/docs/advanced_features/{hicache_design,hicache_best_practices,dp_dpa_smg_guide,sgl_model_gateway,speculative_decoding,attention_backend,server_arguments}.mdx`、`docs/docs/hardware-platforms/*`
  - 外部一手：SGLang v0.4 博客（2024-12-04，https://lmsys.org/blog/2024-12-04-sglang-v0-4/）；HiCache 博客（2025-09-10，https://lmsys.org/blog/2025-09-10-sglang-hicache/）；DeepSeek-V4.1 Day-0 博客（2026-09-09，https://www.sglang.io/blog/deepseek-v4.1-day0-support）；GitHub PR #1970、#2067、#35081、#39627、#40177、#40323、#43228；GitHub Release v0.4.0 / v0.4.5 / v0.5.2 / v0.5.19 / v0.5.21 / v0.5.22
  - DeepInfer 对照：`research/sources/DeepInfer/doc/inference_deployment_qa.md:274`、`README.md:95`；本套笔记 R06（KV/前缀缓存）、R08（DP/EP 并行）

## 1. 三个最重要的发现

1. **SGLang 对 DeepSeek-V4.1-Flash 不是"社区适配"，而是 Day-0 全套方案**：官方 DSpark 实现、P/D 分离、HiCache（明确声明与 DSpark 兼容）、`sglang-router` 全部有文档；2026-09-09 先以 `lmsysorg/sglang:dev-dsv41` 预览镜像发布，v0.5.21（2026-10-02）进入正式 release，v0.5.22（2026-10-19）补上 H200 2.3× 与 MI350X/MI355X 的 DSpark 1.9×。
2. **"RadixAttention" 在代码里已经换代**：默认路径不再是 `radix_cache.py` 的单一 RadixCache，而是 `UnifiedRadixCache`（Full/SWA/Mamba 组件化，v0.5.19 起所有模型默认），且树核心默认用 Rust（v0.5.21）；HiCache 是它的三层扩展（L1 GPU / L2 host / L3 storage），L2 是**实例私有**，跨实例共享只能靠 L3。
3. **overlap 调度器是 v0.4.0（2024-12-04）起默认开启的"零开销"实现**：Python scheduler 用 `schedule_stream` / `forward_stream` / `copy_stream` 三条流 + `FutureMap` 把 `schedule(N+1)` 藏在 `forward(N)` 下面；DeepInfer 官方 QA 明确写"Following existing implementations (SGLang, NanoFlow)"，但把 DSpark 的每请求验证长度改到 GPU TileLang 上算，SGLang 则用 host 侧 confidence planner + ragged verify。

## 2. 事实条目

### #1 进程模型：主进程 + 每 TP rank 一个 Scheduler 子进程 + Detokenizer 子进程
- 证据：`python/sglang/srt/entrypoints/engine.py:247-255`
  > `1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.`
  > `2. Scheduler (subprocess): ... forwards them, and sends the output tokens to the Detokenizer Manager.`
  > `3. DetokenizerManager (subprocess): ...`
  > `Note: 1. The HTTP server, Engine, and TokenizerManager all run in the main process. 2. IPC ... via ZMQ`
  `engine.py:927-956` 对 `pp_rank` × `tp_rank` 逐个 `mp.Process(target=run_scheduler_process_func, args=(server_args, port_args, gpu_id, tp_rank, pp_rank, ...))`
- 等级：A
- 意味着什么：SGLang 没有"一个调度器管整组 TP"这回事；**每个 TP rank 一个 Scheduler 进程、一张 GPU、一个 `TpModelWorker`**，TP 组是这些进程之间的 NCCL 组；PP 时再乘 `pp_size`。DPA 下这个规则不变：`launch_tensor_parallel_group` 仍对每个 `tp_rank` 起一个 scheduler，`dp_rank` 由 `compute_dp_attention_world_info` 推导（`data_parallel_controller.py:711-790`）。`num_dp_ranks = dp_size × attn_dp_size` 是**逻辑 DP rank 数**，不是 scheduler 数；只有 `dp_size == 1`、PP=1、`attn_dp_size == tp_size` 时两者才碰巧相等。与 DeepInfer 相比差异很大：DeepInfer 的 `ParallelConfig` 里根本没有 TP 轴（R08 #4），调度在 C++ 扩展内；SGLang 的 TP 是一等公民，调度器是 Python 进程。
- 疑点：`docs/docs/advanced_features/dp_dpa_smg_guide.mdx:112` 说 DPA "creates N attention data-parallel groups ... one scheduler each"，`129` 又说 scheduler 数 = `dp_size × attn_dp_size`；代码是**每 tp_rank 一个 scheduler**（×PP），且 `attn_dp_size < tp_size` 时同一 `dp_rank` 的多个 scheduler 共用 `worker_ports[dp_rank]`（`server_args.py:843-844`）。这条非典型路径本笔记未实跑验证，书里引用调度器数量时按 `tp_size × pp_size` 写。

### #2 Scheduler 与 TpModelWorker 的分工
- 证据：`python/sglang/srt/managers/scheduler.py:442-446`
  > `class Scheduler(...): """A scheduler that manages a tensor parallel GPU worker."""`
  `python/sglang/srt/managers/tp_worker.py:279-281`
  > `class TpModelWorker(BaseTpWorker): """A tensor parallel model worker."""`
  调度主循环 `scheduler.py:4559-4563` `def run_batch(self, batch, pp_proxy_tensors=None)`；调用点 `scheduler.py:4743` `self.tp_worker.forward_batch_generation(...)`；`tp_worker.py:615-625` 是模型前向 + 采样入口
- 等级：A
- 意味着什么：Scheduler 持有 `tree_cache`、`req_to_token_pool`、`token_to_kv_pool_allocator` 和批调度状态；`TpModelWorker` 持有 `ModelRunner`（模型、attention backend、CUDA graph）。书里画"调度 / 执行"边界应画在这一层。
- 疑点：`Scheduler` 的 docstring 说 "a tensor parallel GPU worker"，实际是**一个 rank 的 worker**，措辞容易误读。

### #3 两种 DP 不要混：`dp_size` 是模型副本，`attn_dp_size` 是 TP 组内的 attention DP
- 证据：`python/sglang/srt/arg_groups/fields/parallel.py:70-90`
  > `dp_size: ... "The number of data-parallel replicas of the model."`
  > `attn_dp_size: ... "the number of data-parallel attention groups inside the TP group, one scheduler each, while the FFN stays tensor parallel."`
  `python/sglang/srt/managers/data_parallel_controller.py:157-158`
  > `class DataParallelController: """A controller that dispatches requests to multiple data parallel workers."""`
  同文件 `188-194` 的 dispatch 表：`ROUND_ROBIN` / `FOLLOW_BOOTSTRAP_ROOM` / `TOTAL_REQUESTS` / `TOTAL_TOKENS`
  `docs/docs/advanced_features/dp_dpa_smg_guide.mdx:129`
  > "The number of DP ranks (one scheduler each) is `dp_size × attn_dp_size`; one of the two is always 1."
- 等级：A（参数定义）/ B（rank 数语义）
- 意味着什么：`--dp-size N` 是 N 个完整模型副本（内置 `DataParallelController`）；`--attn-dp-size N` 是同一 TP 组里切出 N 个 attention DP 组、每组一个 attention-TP 子组，FFN 仍 TP。两者不能同时 >1（见 #15）。
- 疑点：内置 DP 的 `load_balance_method` 只有 round_robin / total_requests / total_tokens / follow_bootstrap_room，没有 cache-aware；生产文档明确不推荐（见 #22）。

### #4 overlap scheduler 默认开启，`--disable-overlap-schedule` 才是开关
- 证据：`python/sglang/srt/arg_groups/fields/schedule.py:193-199`
  > `disable_overlap_schedule: A[bool, Arg(help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.", resolvable=True)] = False`
  `python/sglang/srt/managers/scheduler.py:495-497`
  > `self.enable_overlap = (not get_schedule().disable_overlap_schedule and not use_mlx())`
- 等级：A
- 意味着什么：默认就是 overlap 路径；`event_loop_normal` 只在显式关闭时走。与 DeepInfer 的 ZOS 一样，"异步调度"是默认行为而不是可选优化。
- 疑点：无。

### #5 overlap 的首次发布：v0.4.0（2024-12-04）；更早版本是 opt-in
- 证据：SGLang v0.4 博客（2024-12-04，https://lmsys.org/blog/2024-12-04-sglang-v0-4/）
  > "In this new version, we pushed it to the extreme and achieved a near zero-overhead batch scheduler. This idea is simple and has been proposed in NanoFlow."
  > "The scheduler runs one batch ahead and prepares all the metadata required for the next batch."
  > "Usage: It is turned on by default, so you do not need to change anything!"
  GitHub PR #2067（标题 `Enable overlap by default`，created 2024-11-18，merged 2024-11-20）；v0.4.0 release published 2024-12-04（GitHub API）
- 等级：B
- 意味着什么：准确说法是"**overlap 机制在 v0.4.0 之前已可 opt-in（flag `--enable-overlap-schedule`），v0.4.0 是第一个默认开启、并以 zero-overhead 为卖点的 release**"。不要写成"SGLang v0.4 才发明 overlap"。
- 疑点：PR #2067 的正文说 Triton backend 的性能调优、更多 penalty、mixed chunked prefill 当时还没做完；这些是 v0.4.0 的已知限制。

### #6 overlap 的工作流：三 stream + FutureMap + 延迟一批处理
- 证据：`python/sglang/srt/managers/scheduler.py:2020-2093`
  > `def event_loop_overlap(self): """A scheduler loop that overlaps the CPU processing and GPU computation."""`
  关键步骤：`get_next_batch_to_run` → `run_batch(batch)` → `self.result_queue.append((batch.copy(), batch_result))` → 再 `pop_and_process()` 处理**上一批**的结果（`2076-2079`）；采样在上一批结果处理之后才 launch（`2084-2087`）
  三条流：`scheduler.py:1141-1148` `self.schedule_stream = ... device_module.Stream(priority=0)`、`forward_stream = model_runner.forward_stream`；`init_overlap` 里另有 `copy_stream`（`1690-1698`）
  元数据中转：`python/sglang/srt/managers/overlap_utils.py:281-284`
  > `class FutureMap: """Always-on pool-indexed relay for cross-iter values. Forward writes via publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu."""`
  `scheduler.py:4600-4630`：`with self.forward_stream_ctx:` → `forward_stream.wait_stream(schedule_stream)` → `future_map.resolve_seq_lens_cpu(batch)` → `resolve_forward_inputs(batch, future_map)`
- 等级：A
- 意味着什么：流程是 `GPU: forward(N) → sample(N)` 与 `CPU: schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)` 并行；`FutureMap` 用 `req_pool_idx` 做跨迭代中继，避免在 Python 侧做 D2H 同步。DeepInfer 的 QA 文档把这条流水线逐字写出来了（`doc/inference_deployment_qa.md:274`），并注明参考 SGLang、NanoFlow。
- 疑点：`schedule_stream` / `forward_stream` / `copy_stream` 的实际并发关系依赖 backend；代码里有 WAR barrier（`1971-1982`）防止下一批的共享 buffer 写入盖住当前 forward 的读。

### #7 overlap 不是无条件的：连续 prefill、spec topk、mamba 都会关掉它
- 证据：`scheduler.py:2095-2131` `is_disable_overlap_for_batch`
  > "For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch."
  > "In DP attention mode, use the globally synchronized is_extend_in_batch so all DP ranks make the same overlap decision (avoiding deadlock)."
  `docs/docs/advanced_features/speculative_decoding.mdx:743-750`
  > "The overlap scheduler currently only supports `--speculative-eagle-topk 1`; set `--speculative-eagle-topk 1` explicitly."
  `docs/docs/advanced_features/server_arguments.mdx:2051`（mamba scheduler）：`no_buffer` "does not support overlap scheduler"
- 等级：A（代码）/ B（文档）
- 意味着什么：书里画 overlap 收益时要写清配置前提。DPA 下 overlap 决策要全 DP 组一致，否则会死锁；这条比单 rank 的 overlap 更容易被忽略。
- 疑点：`docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx:93` 注释说 "Pipeline parallelism is incompatible with the overlap scheduler"，但 `scheduler.py` 里 PP 与 overlap 共用 `forward_stream` 且有 `scheduler_pp_mixin`；**PP 与 overlap 是否完全互斥未能确认**。

### #8 RadixAttention 的数据结构：token 前缀树 + 节点级 KV 段 + lock_ref
- 证据：`python/sglang/srt/mem_cache/radix_cache.py:59-100` `RadixKey`（`token_ids`、`extra_key`、`cache_salt`、`is_bigram`、`limit`）
  > `extra_key` "Namespaces the tree and storage; KV events see only its LoRA name."
  > `is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams)...`
  `radix_cache.py:281-330` `class RadixCache(BasePrefixCache)` 初始化 root、`evictable_leaves`、eviction strategy；`364-417` `match_prefix`（按 `page_size` 截断、命中落在节点中间时 split）；`428-551` `insert`；`552-595` `evict`（leaf heap + `free_segment` + `_delete_leaf`）；`582-616` `inc_lock_ref` / `dec_lock_ref`
- 等级：A
- 意味着什么：每个节点是一段连续 token 的 KV；`lock_ref` 把正在跑的请求钉住，evict 只从 `evictable_leaves` 里按策略挑；`extra_key` / `cache_salt` 把 LoRA、租户、采样 salt 隔离成不同命名空间。DeepInfer 开源树的 `BlockManager` 没有 radix/hash/LRU，只能显式 `fork_seq` 共享前缀（R06 #25），这是两者最根本的缓存分层差异。
- 疑点：`RadixCache` 仍是可用的 Python 实现，但默认路径已换到 `UnifiedRadixCache`（见 #10）；书里不要把 `radix_cache.py` 当成唯一实现。

### #9 淘汰策略不止 LRU：LRU/LFU/FIFO/MRU/FILO/Priority/TLRU/SLRU
- 证据：`python/sglang/srt/mem_cache/evict_policy.py:10-49`
  > `class EvictionStrategy(ABC)` / `LRUStrategy` / `LFUStrategy` / `FIFOStrategy` / `MRUStrategy` / `FILOStrategy` / `PriorityStrategy` / `TLRUStrategy` / `SLRUStrategy`
- 等级：A
- 意味着什么：`--radix-eviction-policy` 可切；默认 `lru`（`cache_init_params.py:31`；`radix_cache.py:313-316` 在 `disable` 时也强制 lru）。HiCache 的 host 侧还有自己的 eviction 逻辑（见 #12）。
- 疑点：不同 policy 对 HiCache / PD 的兼容性未逐一核对。

### #10 当前默认是 UnifiedRadixCache，树核心默认 Rust
- 证据：`python/sglang/srt/mem_cache/unified_radix_cache.py:199` `class UnifiedRadixCache(BasePrefixCache)`；`611` `match_prefix`、`659` `insert`、`683` `evict`
  `python/sglang/srt/mem_cache/unified_cache/components/README.md:1-12`
  > "A component-based, pluggable prefix cache framework for SGLang that unifies Full-attention, Sliding-Window-Attention (SWA), and Mamba/SSM caching into a single radix tree."
  `python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py:1-8`
  > "selected by its per-cache override or SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND (default \"rust\")"
  `python/sglang/srt/mem_cache/kv_cache_builder.py:51` 已 `from ...unified_radix_cache import UnifiedRadixCache`
  Release：v0.5.19（2026-09-05）"**Unified radix tree by default.** ... for every model"（#35081）；v0.5.21（2026-10-02）"The prefix cache now runs on a Rust core by default"（#39627），并移除旧 `HiRadixCache`（#40787）
- 等级：A（代码）/ B（release）
- 意味着什么：书里写 "RadixAttention" 应说明它是**设计概念**，当前实现是 UnifiedRadixCache：Full/SWA/Mamba 三种 component 共用一棵树，各自有 lock ref、LRU、evict 驱动；Rust core 不支持的配置（session radix cache、非 Linux、自定义 component）自动回退 Python。
- 疑点：`radix_cache.py` 仍在树里，可能用于仿真/旧 backend；其与 UnifiedRadixCache 的精确调用边界未逐路径核对。

### #11 HiCache 的三层：L1 GPU、L2 host、L3 storage；L2 是实例私有
- 证据：`docs/docs/advanced_features/hicache_design.mdx:10-12`
  > "SGLang introduced RadixAttention ... and **HiCache**, which extends this idea to host memory and distributed storage."
  > "HiCache organizes GPU memory as L1, host memory as L2, and distributed storage as L3."
  `hicache_design.mdx:25-29` 表格：L1 = GPU memory / one instance / 不共享；L2 = host memory / one instance / **No**；L3 = storage backend / 由 backend 决定 / 只有 backend 配对才共享
  `hicache_design.mdx:61`
  > "**L2 is node-local and instance-private.** ... two instances never read each other's L2 ... A KV cache produced by instance 0 becomes visible to instance 1 only after it reaches L3."
- 等级：B
- 意味着什么：HiCache 不能把多台机器的 host memory 拼成一个更大的 L2；`--hicache-ratio` / `--hicache-size` 只扩本实例的私有 L2，跨实例复用必须配 `--hicache-storage-backend`。这与 DeepInfer 的生产三层（L2 只放 SWA state、L3 是 3FS，R06 §5.7）不是同一个切分：SGLang 的 L2 放完整 KV，DeepInfer 的 L2 只放 state。
- 疑点：无。

### #12 HiRadixTree 与 HiCache 工作流：local match → prefetch L3 → write-back
- 证据：`hicache_design.mdx:67-75`
  > "HiRadixTree extends this idea: each node ... records where that KV cache is stored—whether in local GPU memory, CPU memory, L3 storage, or multiple of these tiers."
  > "to reduce overhead, HiRadixTree does not store or continuously synchronize metadata for L3 ... queries the backend in real time"
  > "three key operations: **local match**, **prefetch** and **write-back**."
  `hicache_design.mdx:81-131` 分别讲 local match / prefetch from L3 / data write-back；`132-160` 多 rank 同步与传输优化；`157-179` 与 PD-disaggregation 的集成、统一接口与 L3 backend
- 等级：B
- 意味着什么：命中路径是"先查本地 L1/L2 的树，缺的部分向 L3 prefetch，prefill 完再按 write policy 回写 L2/L3"；L3 元数据不常驻，查一次 L3 就多一次 backend 查询。DeepInfer 的 Entrypoint 是"先把 hash-chain 命中长度算完再路由"，SGLang 是"引擎内树先给本地命中，再异步补 L3"，两者把延迟放在了不同位置。
- 疑点：L3 查询延迟与 `--hicache-storage-prefetch-policy`（best_effort / wait_complete / timeout）的取舍，文档没给统一数字。

### #13 HiCache 的参数与 L3 后端
- 证据：`python/sglang/srt/arg_groups/fields/memory.py:98-176`
  > `enable_hierarchical_cache`（alias `--enable-hicache`）默认 `False`
  > `hicache_write_policy` 默认 `write_through`，可选 `write_back` / `write_through_selective`
  > `hicache_io_backend` 默认 `kernel`，可选 `direct` / `kernel` / `kernel_ascend`
  > `hicache_mem_layout` 默认 `page_first`
  > `hicache_storage_backend` 内置：`file`、`sim`、`mooncake`、`npu_memcache`、`hf3fs`、`nixl`、`aibrix`、`dynamic`、`eic`、`simm`、`mori`、`shm`、`tensorcast`、`seaweedfs`
  `python/sglang/srt/managers/cache_controller.py:295-367` `class HiCacheController`；write policy 校验 `362-367`；`memory_pool_host.py` / `storage/` 分层见 `mem_cache/README.md`
- 等级：A
- 意味着什么：L3 是可插拔插件，DeepSeek 3FS（`hf3fs`）和 Mooncake 都在内置列表里；`write_through` 是默认（每页写 GPU 时同步镜像到 host），`write_back` 推迟到 evict。与 DeepInfer 只把 3FS 当外置 kvcache-store 相比，SGLang 把 L2/L3 都做进了引擎。
- 疑点：`HiCacheController.__init__` 的 `write_policy` 形参默认是 `write_through_selective`（`cache_controller.py:306`），但 server args 的默认是 `write_through`（`memory.py:127`）；实际以 server args 为准，读代码时别被形参默认值误导。

### #14 DP-Attention 的起源：PR #1970，2024-11 合入，v0.4.0 发布
- 证据：GitHub PR #1970 `Support DP MLA`（created 2024-11-09，merged 2024-11-16），正文
  > "Support data parallel on MLA for DeepSeek model to reduce replicated KV cache."
  > `python -m sglang.launch_server ... --tp 8 --dp 8 --enable-dp-attention`
  SGLang v0.4 博客（2024-12-04）
  > "DeepSeek models use MLA and only have one KV head. If we use tensor parallelism on 8 GPUs, it will lead to duplicated KV cache..."
  > "we've implemented data parallelism (DP) for the multi-head latent attention (MLA) mechanism..."
  > "The attention-processed data will be all-gathered among all workers before entering the Mixture-of-Experts"
  > "on 8 x H100 80GB GPUs ... SGLang v0.4 achieved 1.9x decoding throughput compared to SGLang v0.3"
- 等级：B
- 意味着什么：SGLang 在 2024-11 就为 DeepSeek 的 MLA 做了 attention-DP，随 v0.4.0 发布；动机是"MLA 只有一个 KV head，TP 会把 KV 复制 8 份"。DeepInfer 也走 attention DP + EP MoE，但它的 DP 组用 C++ 调度器做 `MIN` all_reduce 同步步型、空闲 rank 跑 1-token placeholder（R08 #1）；SGLang 文档强调的是每个 DP replica 可以独立处于 prefill/decode/idle（`dp_dpa_smg_guide.mdx:46-50`）。这是两种不同的 DPA 协同策略。
- 疑点：PR #1970 早于 v0.4.0 约 3 周，但"首次发布"应以 v0.4.0（2024-12-04）为准；不要把 PR 合并日当 release 日。

### #15 DPA 的当前实现：`attn_dp_size` 在 TP 组内切 attention，FFN 仍 TP/EP
- 证据：`docs/docs/advanced_features/dp_dpa_smg_guide.mdx:110-129`
  > "It creates N attention data-parallel groups inside the TP group, one scheduler each, while the FFN stays tensor parallel."
  > "`--attn-dp-size` must be greater than 1 ... `tp_size % attn_dp_size == 0` ... `--dp-size` > 1 with DPA is not supported."
  > "The number of DP ranks (one scheduler each) is `dp_size × attn_dp_size`; one of the two is always 1."
  `python/sglang/srt/layers/dp_attention.py:409-424` `compute_dp_attention_world_info`；`462-464` `is_dp_attention_enabled`；`625-730` `_dp_gather_via_all_reduce` / `_dp_gather_via_all_gather`；`dp_attention.py:951-1044` `_dp_gather` / `dp_gather_partial` / `dp_gather_replicate` / `dp_scatter`
- 等级：A（代码）/ B（文档）
- 意味着什么：DPA 的通信核心是"attention 各 rank 独立算，进 MoE 前把 token all-gather 成全局 batch，出 MoE 再 scatter 回去"；如果同时开 EP，则改用 all-to-all（DeepEP），不必 all-gather。`--enable-dp-attention` 是旧拼写，等价于 `--dp-size N --enable-dp-attention` → `--attn-dp-size N --dp-size 1`，将在 2026-12-31 后移除（`dp_dpa_smg_guide.mdx:123-125`）。
- 疑点：① DPA + `--dp-size > 1` 明确不支持；elastic EP scale joiner 是例外（`dp_dpa_smg_guide.mdx:127`）。② 文档说每个 attention-DP 组 "one scheduler"，但 `launch_tensor_parallel_group` 按 `tp_rank` 起 scheduler（`data_parallel_controller.py:711-790`），`attn_dp_size < tp_size` 时一组有多个 scheduler；引用调度器数量时要按 `tp_size`（×PP），不要按 `dp_size × attn_dp_size`。

### #16 DPA 已扩展到标准 attention 与 V4.1 的 PD decode
- 证据：`dp_dpa_smg_guide.mdx:93-108`
  > DPA 目标模型：DeepSeek V2/V3/R1、MiniMax、Kimi-K2，以及 Qwen（PR #6121）
  Release v0.5.22 notes：`[DeepSeek-V4.1][PD] Decode node support DP attention: #40177`、`[DeepSeek-V4.1][PD] Support prefill node use cp parallel strategy: #40323`、`[DeepSeek-V4.1] Enable DP attention with MegaMoE: #43228`
- 等级：B
- 意味着什么：V4.1 的 DPA 不是"只能跑在普通 colocate"，PD 的 decode 节点也支持；prefill 节点可以用 CP。官方 cookbook 的静态矩阵把 DPA 从 V4.1 默认配方里拿掉了，但 release notes 说明它在 PD/agentic 组合里是存在的。书里不要把"默认矩阵没有"写成"引擎不支持"。
- 疑点：cookbook 的 `deepseek-v4_1.jsx:11-13` 明确写 "DP-Attention, DeepEP and MegaMoE are absent by design ... DP-Attention runs in the MI355X agentic recipe (cookbook §3.6)"；与 release notes 的 #40177/#43228 有张力，见 §4。

### #17 V4.1-Flash 的官方支持：Day-0 全套特性表
- 证据：SGLang Day-0 博客（2026-09-09，https://www.sglang.io/blog/deepseek-v4.1-day0-support）
  > "SGLang provides Day-0 support for V4.1."
  特性表：Parallelism = TP / EP / **DP attention** / CP（text only）/ MoRI（text only）；Speculative = **Official DSpark implementation**；Caching = Radix prefix cache / chunked prefill / **Mooncake** / **PD disaggregation**；**HiCache = Yes, compatible with DSpark speculative decoding**；Execution = decode CUDA graph / prefill CUDA graph / multi-stream overlap；Official kernels = **DeepGEMM, FlashMLA**；Quantization = FP8 dense（Blackwell 上 MXFP8）/ MXFP4 experts / FP4 indexer cache；Vision = image input / dynamic resolution / multi-image / VL expert routing；API = chat encoder / tool calling / reasoning parser / reasoning effort / structured output / streaming；SWA bounded replay = encoder + decoder；Engram = on by default, host or device
  > "DeepSeek's updated FlashMLA and DeepGEMM kernels for V4.1 are integrated. We are still integrating the new DeepSelect..."
- 等级：B
- 意味着什么：这不是"能跑"级别的适配，而是把 V4.1 的新架构（compressed attention + SWA、mHC、Engram、DSpark、native vision）逐项列了覆盖状态；HiCache 明确声明与 DSpark 兼容，这是 Q5 的核心答案。
- 疑点：表里的 "DP attention / CP / MoRI" 是 feature 级支持，不等于 cookbook 每个硬件矩阵都默认开启。

### #18 V4.1 的发布状态：Day-0 预览镜像 → v0.5.21 正式 release → v0.5.22 性能/AMD
- 证据：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:19`
  > "DeepSeek-V4.1 Flash support has **not shipped in an SGLang release yet**, so a stock `pip install sglang` cannot serve it. Use the preview Docker image below."
  同文件 `27` / `42-45`：NVIDIA 用 `lmsysorg/sglang:dev-dsv41`；AMD 用 `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021`
  GitHub Release v0.5.21（2026-10-02）新模型表：
  > `| DeepSeek-V4.1 Flash | LLM / VLM | link |`
  GitHub Release v0.5.22（2026-10-19）：
  > "DeepSeek-V4.1 gets up to 2.3x more throughput per GPU on H200, and now runs on AMD MI350X / MI355X with up to 1.9x throughput from DSpark"
- 等级：B
- 意味着什么：**Day-0（2026-09-09）是 preview 镜像；正式 release 支持从 v0.5.21（2026-10-02）开始；v0.5.22（2026-10-19）补齐 H200 性能和 MI350X/MI355X。** 但 cookbook 仍保留"not shipped yet"的旧文案，和 release notes 矛盾。
- 疑点：`deepseek-v4_1.jsx:55-56` 的注释又说 "DeepSeek-V4.1 support has not shipped in a CUDA release yet; the ROCm images carry it from v0.5.21." 三种说法（cookbook / jsx / release）不一致；以 release notes 的"新模型表"为准更可靠，但**CUDA 正式镜像是否就是 `v0.5.21` 的默认 tag 未能从本地文档确认**。

### #19 V4.1 的硬件配方：H200 TP8/EP8，Blackwell TP4/EP4，MI355X TP2/EP1
- 证据：`docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx:15-62` 的 `supportedHardware` 与 `dockerImages`；逐个 cell 解析 `flags` 得到（`flash` 变体，全部 `verified: true`）：
  - GB300：TP4 / EP4；Low-Latency 开 DSpark；High-Throughput `--max-running-requests 256`、不开 DSpark
  - B200：TP4 / EP4；Low-Latency 开 DSpark；High-Throughput 256、不开
  - B300：TP4 / EP4；Low-Latency 开 DSpark；High-Throughput 256、不开
  - H200：TP8 / EP8；两个 cell 都**不开** DSpark；High-Throughput 256
  - MI355X：TP2 / EP1；两个 cell 都开 DSpark；High-Throughput `--max-running-requests 128`
  同文件 `91-99`：
  > "DSpark is the model's bundled 3-stage draft. There is no EAGLE/MTP path and no `--speculative-num-steps` knob."
  `deepseek-v4_1.jsx:11-13`：DP-Attention / DeepEP / MegaMoE 不在默认矩阵；MI355X agentic recipe 例外
- 等级：B
- 意味着什么：V4.1 的官方矩阵是"单机 + FP4 + DSpark 低延迟 / 关投机高吞吐"；H200 的显存最紧（140 GiB），两个 cell 都关 DSpark 且要降 memory fraction（jsx 注释 6-9）。MI355X 反而两个 cell 都开 DSpark。
- 疑点：`flash-1017` 变体的 10 个 cell 全部 `verified: false`，不能当已验证数据引用。

### #20 V4.1 的 P/D、HiCache、router 组合与硬约束
- 证据：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:201-221`
  > "Prefill/decode disaggregation is validated token-identical against a single server across 32 greedy prompts, with GSM8K matching through the router."
  > "Both roles run MegaAttention (`--dsv4-attn-backend flashmla_mega`) by default on SM100/SM103 ... Prefill and decode must store KV in the same layout ... agree on `--dsv4-attn-backend` and `SGLANG_DSV4_KV_LAYOUT`."
  > "PD runs with DSpark: ... static verification (`SGLANG_RAGGED_VERIFY_MODE`, `static` by default), and no context parallelism on decode (CP=1, DCP=1); it rejects any other combination at startup."
  > "HiCache ... cold KV pages spill to CPU pinned memory (L2, GPU + CPU) ... The card is not offered on MI355X: hierarchical cache has not been validated on ROCm for this model yet."
  > "Only the L2 tier is exposed here. For the storage (L3) tier ... see the HiCache best-practices recipe."
  `docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx:102-120`：PD 的 router 命令是 `sglang-router launch --pd-disaggregation ...`
- 等级：B
- 意味着什么：V4.1 的"定制方案"是真实存在的，但组合有硬约束：PD + DSpark 只接受 Mooncake/NIXL + `static` verify + decode CP=1；HiCache 在 V4.1 的 Playground 只开 L2，MI355X 不验证 HiCache；prefill/decode 的 attention backend 和 KV layout 必须一致。书里画"全家桶"时要把这些约束写上。
- 疑点：**PD + DSpark + HiCache 三者同时开启**没有在同一段文档里给出验证结论；HiCache 与 DSpark 兼容（Day-0 博客）和 PD 与 DSpark 兼容（cookbook）是两条独立声明。

### #21 `sgl-router`：单模型、请求级、KV-aware 的 slim router
- 证据：`sgl-router/README.md:3-8`
  > "Slim, KV-aware router for SGLang workers. A router serves one model and spreads its requests across that model's SGLang engines. It picks an engine per request from live engine load, global KV-cache prefix matches, or session affinity, supports prefill/decode (P/D) disaggregation, bucketed deployments..."
  同文件 `26-45`：router → buckets → plain 或 P/D engine group；引擎上报 load report 和 KV events；router 用它们 + 自己的 in-flight + session binding 选引擎；P/D 时同时发 prefill/decode，prefill 直接传 KV 给 decode
  同文件 `42-45`：每个请求先 render + tokenize，按长度选 bucket（SLO → capacity → rank），policy 选一个通过 admission 的 engine，失败可换 engine 重试
  同文件 `127-162`：policy = `power_of_two`（默认，reorg 路径）/ `cache_aware`（按最长缓存前缀，至少 `--cache-affinity-min-matched-tokens`，默认 1024）/ `session_aware`（`x-session-id`）；admission 有 `--max-inflight-requests` / `--max-waiting-requests` / `--max-pending-prefill-tokens` / `--max-running-usage` / `--max-kv-usage`
  同文件 `331-335`：`--dp-aware` 在多 rank engine 内部再选 DP rank
- 等级：A（README 是仓库内文档）/ B（行为描述）
- 意味着什么：这是**请求级**负载均衡，不是 token 级；它把"选哪个 engine"和"engine 内部 continuous batching"分开。cache-aware 需要 tokenizer，且只用于 plain/prefill group；decode group 默认 power_of_two。`sgl-router` crate 版本 `0.1.0`（`sgl-router/Cargo.toml:6-7`）。`--dp-aware` 的 rank 选择顺序是：sticky/session key 哈希 → 最长缓存前缀 → in-flight 最少（或 round_robin）（`sgl-router/src/policies/dp_rank.rs:4-12`）。
- 疑点：`--chat-routing reorg` 不传时走 legacy 路径，默认 policy 变 `round_robin`（`sgl-router/README.md:121-123`）；书里引用默认值时要带这个前提。另外 `PolicyKind` 代码里还有 `load_based` / `fused_score` / `score_policy` / `sticky`（`sgl-router/src/config/types.rs:123-148`），README 只文档化了三种。

### #22 SMG（SGLang Model Gateway）：带控制面的完整网关
- 证据：`sgl-model-gateway/README.md:1-30`
  > "High-performance model routing control and data plane ... orchestrates fleets of workers, balances traffic across HTTP and gRPC backends, and exposes OpenAI-compatible APIs..."
  > Control Plane：Worker Manager / Job Queue / health checker / load monitor / K8s discovery
  > Data Plane：SGLang HTTP routers（regular + PD）、gRPC router、OpenAI router、Router Manager（IGW）
  > "Multiple load balancing strategies (`random`, `round_robin`, `cache_aware`, `power_of_two`, `bucket`) with DP-aware scheduling."
  `docs/docs/advanced_features/sgl_model_gateway.mdx:76-90` 同义；`848-890` policy 表：random / round_robin / power_of_two / cache_aware（默认）/ bucket
  `docs/docs/advanced_features/dp_dpa_smg_guide.mdx:155-190`
  > "Starting from September 2024, SGLang Model Gateway, i.e., SMG, formerly named as SGLang DP Router, was built especially as a production-ready DP routing system with Rust."
  > "Native DP is highly not recommended for use right now. It is only used in some ancient/outdated RL frameworks."
  `sgl-model-gateway/bindings/python/pyproject.toml:6-7,46-48`：包名 `sglang-router`，版本 `0.3.2`，console scripts `smg` / `amg` / `sglang-router`
- 等级：A（代码/README）/ B（文档）
- 意味着什么：SMG 是"控制面 + 数据面"的网关，适合多 worker / 多模型 / PD / gRPC / K8s；内置 `DataParallelController` 只是玩具级（文档自己说不推荐）。V4.1 cookbook 里的 `sglang_router` / `sglang-router` 指的就是 SMG 这个 pip 包，不是 `sgl-router/` crate。
- 疑点：代码里还有 `consistent_hashing`（`sgl-model-gateway/src/config/types.rs:308`）和 `manual` policy（`src/policies/manual.rs`），文档 policy 表没列全。

### #23 两个 router 并存，命名容易混
- 证据：`sgl-router/Cargo.toml:6-10`：`name = "sgl-router"`、`version = "0.1.0"`、`description = "Slim KV-aware OpenAI-compatible router for SGLang workers"`
  `sgl-model-gateway/Cargo.toml:6-7`：`name = "sgl-model-gateway"`、`version = "0.3.2"`
  `sgl-model-gateway/bindings/python/pyproject.toml:6`：包名 `sglang-router`
  `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:89` / `345`：`sglang_router` / `sglang-router`
- 等级：A
- 意味着什么：`sgl-router` 是独立的 slim 单模型 router（Rust binary，crate 0.1.0）；`sglang-router` 是 SMG 的 Python 包名（内含 Rust 扩展）。**同名不同物**，书里第一次出现时必须写全 "SGLang Model Gateway (SMG, pip 包 `sglang-router`)" 或 "sgl-router crate"。
- 疑点：两者是否有代码复用/迁移计划，仓库里没有一份文档说明；标 E。

### #24 投机解码算法族：EAGLE / EAGLE3 / MTP / DFLASH / DSPARK / STANDALONE / NGRAM / UNO
- 证据：`python/sglang/srt/speculative/spec_info.py:39-46`
  > `DFLASH = auto()` / `UNO = auto()` / `DSPARK = auto()` / `EAGLE = auto()` / `EAGLE3 = auto()` / `FROZEN_KV_MTP = auto()` / `STANDALONE = auto()` / `NGRAM = auto()`
  `spec_info.py:106-140`：`is_eagle()` 把 `EAGLE`、`EAGLE3`、`FROZEN_KV_MTP` 都算作 EAGLE family；`is_dflash_family()` = DFLASH 或 DSPARK
  `docs/docs/advanced_features/speculative_decoding.mdx` 目录：EAGLE-2 / EAGLE-3 / Block verification / Multi Token Prediction / UNO / DFlash / Standalone / V2 overlap / Ngram
- 等级：A
- 意味着什么：SGLang 的投机解码不是单一 EAGLE，而是"算法 + draft 来源"的组合矩阵；MTP 被当作 EAGLE 的一种（`FROZEN_KV_MTP`）；DFLASH/DSPARK 共享 draft 基础设施。V4.1 是例外：只走 DSPARK（#19）。
- 疑点：`FROZEN_KV_MTP` 的 `FIXME(kpham_sgl): Remove ... once we have established support for it in the scheduler`（`spec_info.py:103-108`）说明它在 scheduler 里的支持还不完整。

### #25 MTP：用 EAGLE 路径跑模型的 nextn 层
- 证据：`docs/docs/advanced_features/speculative_decoding.mdx:434-452`
  > "We support MTP (Multi-Token Prediction) in SGLang by using speculative decoding."
  > 示例：`--speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`
  > "For DeepSeek MTP usage, refer to DeepSeek-V3.2 cookbook §4.2.3"
  > "For DSA-based MTP, draft workers default to the `sgl-kernel` top-k backend."
  `tp_worker.py:314-315` `self.model_runner_list: List[ModelRunner]`（MTP model runners）
- 等级：B
- 意味着什么：DeepSeek 的 MTP 层在 SGLang 里不是单独的 `MTP` 算法名，而是 `EAGLE` + 模型自带的 nextn/MTP 权重 + `num_steps=1`；`FROZEN_KV_MTP` 是另一条 MTP 变体（从命名看是冻结 KV 的版本，本笔记未展开核对）。V4.1 明确没有 EAGLE/MTP 路径（`deepseek-v4_1.jsx:91-92`）。
- 疑点：`FROZEN_KV_MTP` 与普通 EAGLE-MTP 的差异（是否复用 target KV、是否单独 draft KV）需要读 `frozen_kv_mtp_worker_v2.py` 才能下定论；本笔记未展开。

### #26 DSpark：V4.1 自带的 3-stage draft，不是 EAGLE 树
- 证据：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:100`
  > "It also carries **DSpark**, its own three-stage speculative draft, which the Low-Latency recipe turns on."
  同文件 `193-199`
  > "DSpark is DeepSeek-V4.1 Flash's own bundled draft — there is no EAGLE or MTP path for this model, and no `--speculative-num-steps` knob."
  > "A DSpark step costs meaningfully more than a plain decode step ... at large batch the fixed step cost stops paying for itself."
  `python/sglang/srt/models/deepseek_v4_dspark.py:726-792`
  > `self.gamma = int(dspark_config.resolve_gamma(...))`
  > `if dspark_config.target_layer_ids is not None: self.num_stages = len(dspark_config.target_layer_ids)`
  > `self.stages = nn.ModuleList([DSparkV4Stage(...) for stage_id in range(self.num_stages)])`
  `python/sglang/srt/speculative/dspark_components/dspark_config.py:61-76` `DSparkDraftConfig`（`num_hidden_layers` / `num_target_layers` / `gamma` / `target_layer_ids` / `mask_token` / `markov_rank` / `markov_head_type`）
  `srt/arg_groups/fields/spec.py:78-81`
  > "Draft block size gamma (number of proposed draft tokens). The verify window is gamma + 1"
- 等级：A（代码）/ B（cookbook）
- 意味着什么："3-stage" 指 draft 有 3 个 `target_layer_ids` 对应的 stage（`num_stages = len(target_layer_ids)`），不是 3 个 draft token；`--speculative-dspark-block-size 5` 表示 gamma=5、verify window=6。DSpark 还有 markov head + confidence head（`dspark.py:66-355`），confidence head 是 ragged verify 的前提（#27）。
- 疑点：checkpoint 的 `target_layer_ids` 具体是哪三层，本地没有权重文件，未确认。

### #27 变长验证：`RaggedVerifyLayout` 只对 DSPARK 开放
- 证据：`python/sglang/srt/speculative/ragged_verify.py:9-13`
  > `class RaggedVerifyMode(str, Enum): STATIC = "static"; CAP_ACCEPT = "cap-accept"; COMPACT = "compact"`
  `ragged_verify.py:44-56` `class RaggedVerifyLayout`：`verify_lens`、`graph_num_tokens`、`extend_start_loc`、`qo_indptr_device`、`total_verify_tokens`、`cap`
  `python/sglang/srt/speculative/spec_info.py:176-181`
  > `def supports_ragged_verify(self) -> bool: """Whether this algorithm's verify step may carry a RaggedVerifyLayout (per-request verify lengths)...""" return self.is_dspark()`
  `python/sglang/srt/layers/attention/deepseek_v4_backend.py:1193` `supports_ragged_verify_graph: bool = True`
  `python/sglang/srt/speculative/dspark_components/dspark_planner.py:130-196`
  > `if self._ragged_verify_mode is not RaggedVerifyMode.STATIC:` → 需要 confidence head；`HostConfidenceBudgetPlanner`；`_dynamic_graph_tier` / `_dp_tier_gather_enabled`
  `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx:209`：PD + DSpark 只接受 `static`，decode CP=1
- 等级：A
- 意味着什么：**SGLang 支持"每请求不同 verify 长度"的变长验证，但只限 DSPARK**；EAGLE/EAGLE3/MTP/NGRAM 仍是固定 `num_draft_tokens` 的树/链验证。三种模式：`static`（全量验证，默认，`environ.py:1532`）、`cap-accept`、`compact`（后两者由 confidence head 给每请求预算；`compact` 还使用紧凑的验证行布局，并消费 SPS 成本表）。DeepInfer 的 DSpark 则把每请求长度放到 GPU TileLang kernel 里算（`doc/inference_deployment_qa.md:264`：cumulative product、global sort top-k、fused remap scatter、写回 pinned buffer），SGLang 是 host planner 算预算、GPU 按 token-bucket 图执行；这是同一目标的两条实现路径。
- 疑点：`compact` 在 DP attention + TP MoE 下有一串额外条件（`dspark_planner.py:167-177`），不是所有 V4.1 配方都能开。

### #28 Block verification 不是"变长验证"
- 证据：`docs/docs/advanced_features/speculative_decoding.mdx:409-429`
  > "For stochastic EAGLE, EAGLE3, or NEXTN decoding ... add `--speculative-use-block-verification` with `--speculative-eagle-topk 1`."
  > "Block verification checks every draft prefix and keeps the longest accepted prefix, including when an earlier prefix was rejected."
  > "Use `--speculative-use-rejection-sampling` alone for token-wise verification, which stops at the first rejected token."
- 等级：B
- 意味着什么：block verification 是"验证算法"（保留最长接受前缀），仍然在一个固定 draft 预算内；ragged verify 是"验证形状"（每请求长度不同）。书里不要把两者混为一谈。
- 疑点：DSpark 也支持 block verification（`416-422`），但它和 ragged verify 的组合关系未在文档里展开。

### #29 V4.1 的模型与内核落点：dsv4 backend + flashmla_mega
- 证据：`python/sglang/srt/configs/deepseek_v41.py:1-45`
  > `"""Translate DeepSeek V4.1 HF configs to the runtime's flat config schema."""`
  > `model_type = "deepseek_v41"`；vision 字段映射 `vision_n_layers` / `vision_dim` / ...
  `python/sglang/srt/arg_groups/fields/exec_.py:329-344`
  > "DeepSeek V4 attention backend. 'auto' (default) resolves to 'flashmla_mega' where it is supported and 'flashmla' otherwise."
  > "'flashmla_mega' (DeepSeek-V4.1 on SM100/SM103) runs the fused Q-RoPE + sparse attention + inverse-RoPE + MXFP8 FlashMLA kernel."
  `python/sglang/srt/layers/attention/attention_registry.py:177` `@register_attention_backend("dsv4")`
  `python/sglang/kernels/ops/attention/dsv4/`：`indexer.py` / `fp4_indexer.py` / `compress.py` / `candidate_blocks.py` / `decode_attention_sm100.py` / `attn_metadata_kernels.py` 等；`dsv4/attn.py:43` `assert layout is KVLayout.V4, "only the FlashMLA cache has V4.1 layouts"`
- 等级：A
- 意味着什么：V4.1 走独立的 `dsv4` attention backend，不是通用 flashinfer/flashmla 路径；decode 默认 `flashmla_mega` 把 Q-RoPE、sparse attention、inverse-RoPE、MXFP8 FlashMLA 融成一个 kernel；`trtllm` 是 opt-in（SM100/SM103 + FP8 KV）。这解释了为什么 V4.1 的 cookbook 说"不要覆盖 backend"。
- 疑点：H200 配方显式 `--attention-backend dsv4`，但没有显式 `--dsv4-attn-backend`；`auto` 在 Hopper 上落到 `flashmla` 还是 `flashmla_mega` 未确认（cookbook 只说 `flashmla_mega` 用于 SM100/SM103）。

### #30 硬件与 DeepSeek 内核集成：DeepGEMM + FlashMLA 已进，DeepSelect 未完成
- 证据：Day-0 博客（2026-09-09）
  > "Official Kernels | Yes | DeepGEMM, FlashMLA"
  > "DeepSeek's updated FlashMLA and DeepGEMM kernels for V4.1 are integrated. We are still integrating the new DeepSelect..."
  Release v0.5.21 notes：CUDA 13.4 image 带 `sgl-kernel 0.4.7`、`sgl-deep-gemm 0.2.0`、`sgl-deep-ep 0.1.2`
  `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:16-24`：`import deep_gemm` + `get_mn_major_tma_aligned_tensor`
  `python/sglang/srt/layers/attention/flashmla_backend.py:10`：`from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata`
  `python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py:17`
  > "512 fp8 values + 4 fp32 scales; FlashMLA's V4.1 format has the same size."
  `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:54`：`from deep_ep import Buffer, Config`
  `docker/sgl-deep-gemm.Dockerfile` / `docker/sgl-deep-ep.Dockerfile` / `docker/sgl-flash-mla.Dockerfile` 分别构建这三个组件
- 等级：A（代码）/ B（博客、release）
- 意味着什么：**DeepSeek 官方内核里，SGLang 已集成 DeepGEMM（含 SM90 native block32，#42604）和 FlashMLA（含 V4.1 KV layout），DeepEP 作为 MoE all-to-all 后端可用，DeepSelect 仍在集成中。** 此外 V4.1 还有大量 SGLang 自己在 `kernels/ops/attention/dsv4/` 里写的 kernel（FP4 indexer、compress、top-k、mHC、Engram），不能都算成"DeepSeek 内核"。
- 疑点：DeepGEMM 在 V4.1 的默认 MoE runner 是不是 `flashinfer_mxfp4` 而不是 DeepGEMM？`deepseek-v4_1.jsx:76-77` 说 `flashinfer_mxfp4` 是"唯一 MoE runner this model has run on"；DeepGEMM 更多用在 dense GEMM / SM90 block32。两者不要混。

### #31 性能数字（必须带条件）
- 证据：
  - overlap：SGLang v0.4 博客，"1.1x speedup against its previous version and a 1.3x speedup against other state-of-the-art baselines"，Triton attention backend 的 Nsight profile（5 个连续 decode batch，GPU 无 idle）
  - DPA：v0.4 博客，8×H100 80GB，DeepSeek 模型，1.9× decoding throughput vs SGLang v0.3
  - V4.1 H200：v0.5.22 release，"up to 2.3x more throughput per GPU on H200"（8×H200，PR #41251 标题写 "2.1x to 2.3x output throughput per GPU"）
  - V4.1 AMD：v0.5.22 release，"DSpark up to 1.9x the throughput of plain decode on 4x MI350X; 128K-input TTFT 860 to 576 ms"；另一条 "102.6K tok/s/GPU on AgentX at concurrency 32, 2x MI355X"
  - Day-0 博客：4×GB300，random 4K in / 1K out，DSpark，EP4，bs=1 时 1,130 tok/s per user，221 tok/s per GPU
- 等级：B
- 意味着什么：这些数字的基线各不相同（v0.3、plain decode、不同 GPU 数、不同 batch），不能直接横比；书里引用必须带"谁测的、硬件、配置、负载、和谁比"。
- 疑点：Day-0 的 1,130 tok/s/user 是 bs=1、EP1 的条件（图注写 "bs=1 on EP1"），不要当成通用吞吐。

## 3. 必答问题逐条回答

### Q1 引擎架构与进程模型（Scheduler、TP/DP Worker）
- **主进程**：HTTP server + Engine + TokenizerManager；负责 tokenize、请求接入、结果返回（`engine.py:247-255`）。
- **Scheduler 子进程**：每个 TP rank 一个（PP 时再乘 PP），进程内是 `Scheduler` + 一个 `TpModelWorker`（一张 GPU）+ `ModelRunner`；调度循环是 `ingest_requests → get_next_batch_to_run → run_batch → process_batch_result`（`scheduler.py:4559` 起）。
- **Detokenizer 子进程**：把 output token 还原成文本再回 TokenizerManager（`engine.py:247-255`）。
- **DP**：`dp_size > 1` 时多一个 `DataParallelController` 子进程，按 `round_robin` / `total_requests` / `total_tokens` / `follow_bootstrap_room` 把请求分给各 DP 组；DPA 时每个 TP rank 仍各有一个 scheduler，`dp_rank` 由 `compute_dp_attention_world_info` 推导，同一 `dp_rank` 的 rank 组成一个 attention-TP 组（`data_parallel_controller.py:711-790`）。`num_dp_ranks = dp_size × attn_dp_size` 是逻辑 DP rank 数，不等于 scheduler 数；scheduler 数是 `tp_size × pp_size`（单机 PP=1 且 `tp_size == attn_dp_size` 时才与 DP rank 数相等）。
- 与 DeepInfer：DeepInfer 没有 TP 轴，调度器在 C++ 扩展里，rank 按 `(replica, dp, cp)` 组织（R08 #1/#4）；SGLang 的 TP 是一等公民，调度器是 Python 进程，靠多 stream 隐藏 CPU 开销。**相同点**是都把调度与 GPU 执行解耦；**不同点**是 SGLang 的并行轴更多（TP/PP/DP/DPA/CP/EP），DeepInfer 更偏 DP+EP+CP。
- 引用：#1、#2、#3、#15。

### Q2 零开销重叠调度器：设计、工作流、首次发布时间
- **设计**：scheduler 永远比 GPU 早一批；`schedule(N+1)` 在 CPU 上跑，`forward(N)` 在 GPU 上跑，`process_batch_result(N−1)` 在 CPU 上跑；三者用 `schedule_stream` / `forward_stream` / `copy_stream` 分离（#6）。
- **工作流**（按代码顺序）：
  1. `ingest_requests()` 收请求；
  2. `get_next_batch_to_run()` 取下一批；
  3. `run_batch(batch)` 在 `forward_stream` 上 launch，`FutureMap` 解析上一批留下的 seq_lens / input_ids；
  4. 把 `(batch.copy(), result)` 放进 `result_queue`；
  5. `pop_and_process()` 处理上一批结果（采样、语法、KV 释放）；
  6. 当前批的 sample 在上一批结果处理之后 launch（因为 grammar FSM 依赖上一批的 token）；
  7. `last_batch = batch`，进入下一轮。
- **首次发布**：overlap 机制在 v0.4.0 之前已可 opt-in；**v0.4.0（2024-12-04）是第一个默认开启、以 zero-overhead 为卖点的 release**，PR #2067（2024-11-20）把默认打开。SGLang 自己说 idea 来自 NanoFlow（v0.4 博客）。
- **DeepSeek 文档的引用**：`DeepInfer/doc/inference_deployment_qa.md:274` 原文 "Following existing implementations (SGLang, NanoFlow), the pipeline is overlapped by scheduling asynchronously between GPU and CPU..."；DeepInfer 的流水线是 `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`，与 SGLang 的 event loop 同构。
- **与 DeepInfer 的差异**：DeepInfer 把 DSpark 的每请求 verify 长度放到 GPU TileLang kernel 里算（`doc/inference_deployment_qa.md:264`），SGLang 用 host 侧 `HostConfidenceBudgetPlanner` + `RaggedVerifyLayout`（#27）；DeepInfer 的调度器是 C++，SGLang 是 Python + 三 stream。
- 引用：#4、#5、#6、#7、#27。

### Q3 RadixAttention 与跨层级 HiCache
- **RadixAttention**：以 token 序列为 key 的 radix tree，每个节点是一段连续 token 的 KV；`match_prefix` 找最长前缀（按 `page_size` 对齐），`insert` 在命中终点 split 节点，`evict` 从 `evictable_leaves` 按策略淘汰，`lock_ref` 保护在飞请求（#8）。`extra_key`/`cache_salt` 隔离 LoRA/租户/采样 salt。
- **当前实现**：`UnifiedRadixCache`（Full/SWA/Mamba 组件化）自 v0.5.19 起是所有模型的默认；Rust TreeCore 自 v0.5.21 起默认，不支持的配置回退 Python；旧 `HiRadixCache` 已移除（#10）。
- **HiCache**：L1 GPU / L2 host / L3 storage；L2 是**节点本地、实例私有**，不能跨实例拼池；L3 才可能跨实例共享（`file` 是节点本地，`mooncake`/`hf3fs`/`nixl`/`aibrix` 等才是分布式）（#11、#13）。工作流是 local match → 从 L3 prefetch → prefill 后 write-back；HiRadixTree 只常驻 L1/L2 元数据，L3 元数据按需查 backend（#12）。
- **与 DeepInfer 的异同**：DeepInfer 开源树的 `BlockManager` 没有 hash/radix/LRU，前缀共享只有显式 `fork_seq`（R06 #25）；生产形态把前缀匹配放在 Entrypoint 做 metadata hash-chain，KV 由 3FS 外部存取（R06 Q5）。SGLang 把前缀树和 L2/L3 都放进引擎：**DeepInfer 是"引擎轻、缓存外置"，SGLang 是"引擎重、缓存内建"**。两者都面对"前缀亲和 vs 负载均衡"的冲突，但一个在 Entrypoint 路由层解决，一个在引擎内 + router 层解决。
- **限制**：HiCache 的 L2 不是跨机共享；L3 查询有额外延迟；V4.1 的 Playground 只开放 L2，MI355X 不验证 HiCache（#20）。
- 引用：#8、#9、#10、#11、#12、#13、#20。

### Q4 DP-Attention 的起源与实现
- **起源**：PR #1970 `Support DP MLA`，2024-11-09 创建、2024-11-16 合并；随 v0.4.0（2024-12-04）发布；动机是 DeepSeek MLA 只有一个 KV head，TP 会复制 8 份 KV；8×H100 上 decode 吞吐 1.9× vs v0.3（#14）。
- **实现**：`--attn-dp-size N`（旧名 `--enable-dp-attention`）在 TP 组内切出 N 个 attention DP 组，每组一个 scheduler、各自 KV；FFN 仍 TP；进 MoE 前 all-gather token，出 MoE 后 scatter；开 EP 时改用 all-to-all（DeepEP）；`tp_size % attn_dp_size == 0`（#15）。
- **当前状态**：DPA 已支持标准 attention（Qwen，PR #6121）和 V4.1 的 PD decode（#40177）、prefill CP（#40323）、MegaMoE + DPA（#43228）；`--enable-dp-attention` 将在 2026-12-31 后移除（#15、#16）。
- **与 DeepInfer 的异同**：两者都是 attention DP + EP MoE；DeepInfer 的 DP 组用 C++ 调度器做 `MIN` all_reduce 同步步型、空闲 rank 跑 1-token placeholder（R08 #1），SGLang 文档强调每个 DP replica 可以独立处于 prefill/decode/idle（`dp_dpa_smg_guide.mdx:46-50`），靠 all-gather 的变长 token 数对齐。前者用"同步步型"换简单，后者用"变长 gather"换灵活。
- 引用：#14、#15、#16。

### Q5 SGLang 对 DeepSeek-V4.1-Flash 的官方支持方案
- **有，而且是 Day-0 全套方案**：官方 DSpark 实现、P/D 分离、HiCache（声明与 DSpark 兼容）、`sglang-router`、CUDA graph、DeepGEMM/FlashMLA、FP8/MXFP4/FP4 indexer、native vision、Engram、SWA bounded replay（#17）。
- **发布路径**：2026-09-09 Day-0 预览镜像 `lmsysorg/sglang:dev-dsv41`；v0.5.21（2026-10-02）进入正式 release（新模型表含 V4.1 Flash）；v0.5.22（2026-10-19）补 H200 2.3× 与 MI350X/MI355X DSpark 1.9×；ROCm 镜像 `v0.5.22-rocm10-mi35x-20261021`（#18）。
- **定制方案**：cookbook 给出单机 FP4 配方（H200 TP8/EP8、Blackwell TP4/EP4、MI355X TP2/EP1），低延迟开 DSpark、高吞吐关 DSpark；PD 用 Mooncake/NIXL + `static` verify + decode CP=1；HiCache 在 V4.1 只开 L2，MI355X 不验证；router 用 `sglang_router --policy cache_aware --dp-aware`（#19、#20）。AgentX 还给出 HiCache + Mooncake external linker + Dynamo / sglang_router 的 agentic 长上下文命令（`docs/src/snippets/agentx/deepseek-ai/deepseek-v4_1.jsx`）。
- **限制**：V4.1 没有 EAGLE/MTP 路径，只有 DSpark；DSpark 在高吞吐/大 batch 下不划算；PD + DSpark 对 KV layout / backend / CP 有硬约束；PD + DSpark + HiCache 三者同时开启未在同一段文档里验证（#20、#26）。
- **与 DeepInfer 的关系**：DeepInfer 是 V4.1 的"原厂"引擎（DSpark、mHC、Engram 都在其代码里）；SGLang 是独立的 Day-0 实现，架构理解一致但工程路径不同（DSpark ragged verify 的 host planner vs DeepInfer 的 GPU TileLang）。
- 引用：#17、#18、#19、#20、#26、#27。

### Q6 路由器（sgl-router / model gateway）与请求级负载均衡
- **sgl-router**：单模型 slim router；每请求 tokenize → 按长度选 bucket → policy 选 engine → admission → dispatch；用引擎上报的 load report + KV events + 自己的 in-flight + session binding 决策；P/D 时同时发 prefill/decode；`--dp-aware` 再选 DP rank；无状态，可水平扩（#21）。
- **SMG**：`sgl-model-gateway`，pip 包 `sglang-router`；控制面（Worker Manager / Job Queue / Health / Load Monitor / K8s discovery）+ 数据面（HTTP regular/PD、gRPC、OpenAI proxy、multi-model IGW）；policy 有 random / round_robin / power_of_two / cache_aware（默认）/ bucket，代码里还有 consistent_hashing / manual；2024-09 起从 "SGLang DP Router" 演化而来，Rust 实现（#22）。
- **请求级 LB 的关键**：router 只决定"请求去哪个 engine/DP rank"，engine 内部仍是 continuous batching；cache-aware 需要 tokenizer，按最长缓存前缀选 engine；decode group 默认 power_of_two；admission 有 in-flight / waiting / pending-prefill-tokens / running-usage / KV-usage 五个 cap（#21）。
- **命名坑**：`sgl-router`（crate 0.1.0，slim）≠ `sglang-router`（SMG 的 pip 包 0.3.2）；V4.1 cookbook 里的 `sglang_router` 指后者（#23）。
- **与 DeepInfer 的异同**：DeepInfer 生产形态在 Entrypoint 做 hash-chain 前缀匹配后再路由（R06 Q5），SGLang 的 router 用引擎 KV events 维护近似 radix tree 做 cache-aware；两者都把"前缀亲和"放在引擎外，但 SGLang 引擎内还有一层 RadixAttention 本地命中。
- 引用：#21、#22、#23。

### Q7 投机解码支持（EAGLE / MTP / 变长验证）
- **算法族**：EAGLE、EAGLE3、FROZEN_KV_MTP、DFLASH、DSPARK、STANDALONE、NGRAM、UNO（#24）。
- **EAGLE 系列**：EAGLE-2 / EAGLE-3 是主要推荐路径；V2 speculative worker 默认跑 overlap scheduler，要求 `--speculative-eagle-topk 1`（#7、#24）。
- **MTP**：通过 `--speculative-algorithm EAGLE` + 模型自带 nextn/MTP 权重 + `--speculative-num-steps 1` 实现；DeepSeek 的 MTP 用法见 V3.2 cookbook；`FROZEN_KV_MTP` 是另一条变体（#25）。
- **变长验证**：**只有 DSPARK 支持** `RaggedVerifyLayout`（每请求 verify_lens），模式 `static`（默认）/ `cap-accept` / `compact`；后两者需要 confidence head，PD 场景只允许 `static`；EAGLE/MTP/NGRAM 仍是固定 draft 预算（#27）。`--speculative-use-block-verification` 是"保留最长接受前缀"的 block 验证，不是变长（#28）。
- **V4.1**：只走 DSPARK，没有 EAGLE/MTP；`--speculative-dspark-block-size 5` = gamma 5、verify 6；高吞吐配方关投机（#19、#26）。
- **与 DeepInfer 的异同**：DeepInfer 的 DSpark 用 GPU TileLang kernel 算每请求 verify 长度（`doc/inference_deployment_qa.md:264`），SGLang 用 host confidence planner + `RaggedVerifyLayout`；目标相同（按置信度分配验证预算），实现位置相反。
- 引用：#24、#25、#26、#27、#28。

### Q8 硬件与内核支持（集成了哪些 DeepSeek 内核）
- **硬件**：NVIDIA H200 / B200 / B300 / GB300；AMD MI350X / MI355X（V4.1）；更广义的 SGLang 还支持 MI300X/MI325X、Ascend NPU、Intel XPU、TPU、CPU、Apple Metal、Moore Threads（`docs/docs/hardware-platforms/overview.mdx`）。V4.1 的 Day-0 硬件是 Blackwell + H200 + MI350X，MI355X 在 v0.5.22 补齐（#17、#18）。
- **DeepSeek 内核**：DeepGEMM（含 SM90 native block32，#42604）、FlashMLA（含 V4.1 KV layout，`sgl_kernel.flash_mla`）、DeepEP（MoE all-to-all，`deep_ep.Buffer`）已集成；**DeepSelect 仍在集成中**（Day-0 博客原话）。SGLang 还有自己的 `kernels/ops/attention/dsv4/`（FP4 indexer、compress、candidate blocks、SM100 decode attention）和 DSpark kernels，不能都归到 DeepSeek 名下（#29、#30）。
- **V4.1 默认 kernel 路径**：`--dsv4-attn-backend auto` → `flashmla_mega`（SM100/SM103，fused Q-RoPE + sparse attention + inverse-RoPE + MXFP8 FlashMLA），否则 `flashmla`；`trtllm` 是 opt-in；MoE 默认 `flashinfer_mxfp4`（V4.1 配方注释说这是唯一跑过的 MoE runner）；ROCm 用 AITER / monokernel / Opus sparse prefill（#29、#30）。
- **与 DeepInfer 的异同**：DeepInfer vendor 了 DeepGEMM、FlashMLA、DeepSelect 并自己做 TileLang kernel；SGLang 把 DeepGEMM/FlashMLA 作为上游依赖，DeepSelect 还没接完，但自己补了 dsv4/DSpark 的 in-tree kernel。
- 引用：#17、#18、#29、#30、#31。

## 4. 新发现的问题

1. **V4.1 的"是否已发布"有三种说法**：cookbook 说 "not shipped in a release yet"（`DeepSeek-V4_1.mdx:19`），jsx 注释说 "not shipped in a CUDA release yet; ROCm from v0.5.21"（`deepseek-v4_1.jsx:55-56`），v0.5.21 release notes 却把 V4.1 Flash 列进新模型表。**需要向 SGLang 团队确认**：CUDA 正式镜像到底是 `v0.5.21` 还是仍要 `dev-dsv41`？书里建议写"v0.5.21 起进入 release，cookbook 的 preview 文案滞后"，并加注。
2. **DPA / DeepEP / MegaMoE 在 V4.1 的默认矩阵里被拿掉，但 release notes 说已支持**：`deepseek-v4_1.jsx:11-13` vs #40177 / #40323 / #43228。需要确认这是"cookbook 保守"还是"功能只在特定硬件/版本可用"。
3. **两个 router 的命名冲突**：`sgl-router` crate（0.1.0）与 `sglang-router` pip 包（SMG，0.3.2）同名不同物；仓库没有文档说明两者的关系/迁移计划。书里必须统一命名。
4. **PD + DSpark + HiCache 三者同时开启**没有验证结论；现有文档只分别证明 "DSpark + HiCache" 和 "PD + DSpark"。
5. **V4.1 的 deterministic inference 被拒绝**：cookbook `110` 说 "output is not bitwise stable across batch composition today ... `--enable-deterministic-inference` is refused on this backend." 这对 RL / 训练-推理一致性场景是硬限制，值得单列。
6. **PP 与 overlap 是否互斥**：代码有 PP + overlap 的 stream 路径，但某个配置注释说 PP 与 overlap 不兼容；未确认。
7. **HiCache 的 L3 元数据查询开销**：文档明确说 L3 元数据不常驻、按需查 backend，但没有给出 prefetch policy 的延迟/命中率数据。
8. **Rust TreeCore 的回退边界**：session radix cache、非 Linux、自定义 component 会回退 Python（`tree_core_registry.py:44-70`）；这些配置在书里若画成"Rust 默认"会不准确。
9. **DSpark 的 "3-stage" 语义**：是 `len(target_layer_ids)` 个 draft stage，不是 3 个 draft token；书里需要明确，否则读者会把它和 `gamma`/`block_size` 混淆。
10. **`FROZEN_KV_MTP` 的 scheduler 支持不完整**：代码里有 `FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we have established support for it in the scheduler`（`spec_info.py:103-108`）；它在文档里被当作 MTP 的一种，但实际可用性要打折。

## 5. 可用于图表的材料

### 5.1 overlap 调度器时间线
| 时间 | 事件 | 来源 |
|---|---|---|
| 2024-11-18 → 2024-11-20 | PR #2067 `Enable overlap by default` | GitHub PR |
| 2024-12-04 | v0.4.0 发布，zero-overhead scheduler 默认开启，1.1× vs 上一版、1.3× vs 基线 | v0.4 博客 |
| 2025-04-07 | v0.4.5，HiCache 相关修复开始出现（experimental） | GitHub Release |
| 2025-09-12 | v0.5.2，HiCache 官方博客后的正式 release | GitHub Release |
| 2026-09-05 | v0.5.19，Unified radix tree 对所有模型默认 | GitHub Release #35081 |
| 2026-10-02 | v0.5.21，Rust TreeCore 默认；移除 HiRadixCache；V4.1 进新模型表 | GitHub Release #39627/#40787 |
| 2026-10-19 | v0.5.22，V4.1 H200 2.3×、MI350X/MI355X DSpark 1.9× | GitHub Release |

### 5.2 SGLang 与 DeepInfer 的缓存分层对比
| 维度 | SGLang | DeepInfer（开源树 / 生产形态） |
|---|---|---|
| 前缀缓存位置 | 引擎内 RadixAttention / UnifiedRadixCache | 开源树无自动前缀缓存，只有显式 `fork_seq`（R06 #25）；生产在 Entrypoint 做 hash-chain（R06 Q5） |
| L1 | GPU HBM，实例私有 | HBM 分页 block + state |
| L2 | host DRAM，**实例私有**，放完整 KV | host 只放 SWA state（`state_size_to_disk`），单副本（R06 §5.7） |
| L3 | 可插拔 storage（file/mooncake/hf3fs/nixl/aibrix/...） | 专用 3FS 集群，kvcache-store 管理 |
| 跨实例共享 | 只有 L3 可能共享 | 3FS 全局共享 |
| 淘汰 | LRU/LFU/FIFO/... + `lock_ref` | FIFO free list + `ref_count`，无 LRU（R06 #25） |

### 5.3 V4.1 硬件配方（`flash` 变体，verified）
| 硬件 | TP | EP | Low-Latency DSpark | High-Throughput | 备注 |
|---|---|---|---|---|---|
| GB300 | 4 | 4 | 是 | 256 req，无 DSpark | 单机 |
| B200 | 4 | 4 | 是 | 256 req，无 DSpark | 单机 |
| B300 | 4 | 4 | 是 | 256 req，无 DSpark | 单机 |
| H200 | 8 | 8 | 否 | 256 req，无 DSpark | 140 GiB，显存最紧 |
| MI355X | 2 | 1 | 是 | 128 req，DSpark | ROCm，gfx950 |

### 5.4 路由器策略对比
| 策略 | sgl-router | SMG | 说明 |
|---|---|---|---|
| power_of_two | ✅ 默认（reorg） | ✅ | 采样两个 engine 取负载低者 |
| cache_aware | ✅（plain/prefill） | ✅ 默认 | 按最长缓存前缀选 engine，需 tokenizer |
| session_aware | ✅ | ✅（session header） | 同一会话粘在同一 engine |
| round_robin | ✅（legacy 默认） | ✅ | 轮询 |
| random | ✅ | ✅ | 随机 |
| bucket | ✅（bucket 配置） | ✅ | 按长度/负载分桶 |
| consistent_hashing | ❌（未见） | ✅（代码里有） | 文档未列 |
| DP-rank 选择 | ✅ `--dp-aware` | ✅ DP-aware scheduling | 多 rank engine 内部再选 rank |

注：`sgl-router` 的 `PolicyKind` 代码里还有 `load_based` / `fused_score` / `score_policy` / `sticky`（`sgl-router/src/config/types.rs:123-148`），README 只文档化了 power_of_two / cache_aware / session_aware；SMG 代码里另有 `consistent_hashing` / `manual`，文档表也没列全。

### 5.5 投机解码算法矩阵
| 算法 | draft 来源 | 验证形状 | V4.1 可用 | 备注 |
|---|---|---|---|---|
| EAGLE / EAGLE3 | 独立 draft model | 固定 tree / topk | 否 | 主流推荐 |
| FROZEN_KV_MTP | 模型 nextn / 冻结 KV | 固定 | 否 | scheduler 支持待完善 |
| DSPARK | V4.1 自带 3-stage draft + markov/confidence head | `RaggedVerifyLayout`（static/cap-accept/compact） | 是 | 唯一支持变长验证 |
| DFLASH | 独立 DFlash draft | 固定 block | 否 | DFlash family |
| STANDALONE | 小 draft model | 固定 | 否 | 通用 |
| NGRAM | 无 draft model，host ngram 表 | 固定 | 否 | 无 draft KV |
| UNO | target 模型 + UNO LoRA | 固定 | 否 | 复用 target 权重 |

### 5.6 内核栈
| 层 | 组件 | 来源 | 备注 |
|---|---|---|---|
| Attention（V4.1） | `dsv4` backend / `flashmla_mega` | SGLang in-tree + FlashMLA | fused Q-RoPE + sparse attention + inverse-RoPE + MXFP8 |
| MLA（通用） | FlashMLA | DeepSeek 上游，`sgl_kernel.flash_mla` | V4.1 KV layout |
| MoE GEMM | DeepGEMM / FlashInfer MXFP4 / CUTLASS | DeepSeek + FlashInfer | V4.1 默认 `flashinfer_mxfp4` |
| MoE 通信 | DeepEP / Mori / NIXL | DeepSeek + 第三方 | V4.1 默认矩阵未开 DeepEP |
| Indexer / compress | SGLang `kernels/ops/attention/dsv4/` | SGLang in-tree | FP4 indexer、top-k、compress |
| AMP/量化 | AITER（ROCm）、MXFP8、FP4 | AMD / SGLang | MI355X 用 AITER |
| 缺失 | DeepSelect | DeepSeek | Day-0 尚未集成 |
