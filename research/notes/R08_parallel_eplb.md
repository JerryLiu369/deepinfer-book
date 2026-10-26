---
# R08 DP-Attention + EP-MoE 数据流、Wide-EP、EPLB 与平衡通信

## 0. 元信息
- 代码版本：`deepseek-ai/DeepInfer`，本地 `research/sources/DeepInfer`，HEAD `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21）
- 行号口径：全部基于上述 commit；路径一律写仓库相对路径（`路径:起始行-结束行`）
- 查阅的主要来源：
  - `deepinfer/model_executor/parallel_utils/{parallel_state,communication_op,cp,userbuffer,nccl_memory}.py`
  - `deepinfer/model_executor/eplb/{__init__,policy,shuffle}.py`
  - `deepinfer/model_executor/models/v4_modules/{moe,mlp,attn,linear}.py`、`deepinfer/model_executor/models/deepseek.py`、`deepinfer/worker/{worker,model_runner,cuda_graph_runner,base}.py`
  - `deepinfer/model_executor/input_metadata/input_metadata.py`、`deepinfer/model_executor/layers/tilelang_ops/{prepare_cp_mirror,cuda/prepare_cp_mirror}.py`
  - `csrc/engine/scheduler.hpp`、`csrc/engine/pybind_scheduler.cpp`
  - `deepinfer/config.py`、`deepinfer/environ.py`、`deepinfer/engine/{llm_engine,stats_logger}.py`
  - `vendor/DeepEP/deep_ep/buffers/ep.py`（DeepEP 2.5.0，随仓库 vendor）
  - `tests/distributed/test_expert_shuffle.py`、`tests/models/test_eplb_policy.py`、`examples/{offline_inference,_config}.py`
  - `doc/inference_deployment_qa.md`（官方部署 QA，作为 B 级来源）
- 注意：本笔记**没有运行任何代码**（无 GPU 环境）。执行顺序均为读码推断。

## 1. 三个最重要的发现
1. **Attention DP 与 EP-MoE 之间不靠"同步 batch 大小"来对齐，只靠"同步步型"。** C++ 调度器在 DP 组内做一次 `MIN` all_reduce 投票，决定整组这一步做 prefill 还是 decode；没活干的 rank 跑一个 1-token 的 placeholder 步陪跑，从而保证 EP 的 `dispatch/combine` 集合通信两边数量对齐。各 rank 的 token 数可以完全不同。
2. **MoE 一层的真实流水是 6 段：gate → DeepEP dispatch（expand 布局 + 延迟 epilogue）→ 两个 grouped GEMM → 共享专家在"扣除通信 SM"后并行算 → combine 时把共享专家输出当 `bias` 一起加回 → 等事件。** 共享专家没有独立的 all-reduce 路径，它被折进了 combine 的归约里。
3. **EPLB 的"平衡"是两级贪心，优化目标写死在 `policy.py`：先按窗口内"每副本负载"贪心选热点 expert 复制冗余副本，再以 `min Σ_step max_rank(load)` 把物理槽位装箱到各 GPU。** 权重搬运在同一 NVLink 域内用 `symm_mem` + CUDA IPC 直读（无 P2P 通信），跨机才退回 NCCL 分桶 `batch_isend_irecv`。

## 2. 事实条目

### A. 并行轴与进程组（对应 Q1、Q7）

### #1 全局 rank 按 `(replica, dp, cp)` 三级拆分，DP 组跨 cp 取、CP 组连续取
- 证据：`deepinfer/model_executor/parallel_utils/parallel_state.py:80-99`
  > `ranks = torch.arange(max(enp_size, dp_size * cp_size))`
  > `dp_ranks = rearrange(ranks, "(replica dp cp) -> (replica cp) dp", dp=dp_size, cp=cp_size)`
  > `cp_ranks = rearrange(ranks, "(dp cp) -> dp cp", cp=cp_size)`
- 等级：A
- 意味着什么：全局 rank 编号是 `((replica*dp + r)*cp + c)`。同一个 CP 组的成员在编号上相邻（通常落在同一台机器的同一组 GPU 上，见 #6），而 DP 组是"每隔 cp 个 rank 取一个"。书中画拓扑图必须按这个编号顺序画，否则解释不了 `CUDA_VISIBLE_DEVICES` 的切法。
- 疑点：`replica` 维在部署中是否恒为 1？`ParallelConfig.world_size = max(cp*dp, engram)`（`deepinfer/config.py:173`）会引入额外 rank，需在 engram 章节确认。

### #2 EP 是独立切分轴；`ep_size == world` 时 EP 组直接复用 world 组
- 证据：`deepinfer/model_executor/parallel_utils/parallel_state.py:101-106`
  > `if ep_size == ranks.numel():`
  > `    self.ep = self.world`
  > `else:`
  > `    ep_ranks = rearrange(ranks, "(edp ep) -> edp ep", ep=ep_size)`
- 等级：A
- 意味着什么：EP 组是全局 rank 的**连续段**。当 DP=EP（#3 的默认关系）时，EP 就等于整个世界，`dispatch/combine` 的通信域 = 全部 GPU，这正是文档所说"一个实例的所有 GPU 构成单一通信域"（`doc/inference_deployment_qa.md:211`）的代码来源。
- 疑点：无。

### #3 `dp_size` 的默认值是 `ep_size // cp_size`，即"用 CP 从 DP 里切走一部分 rank"
- 证据：`deepinfer/config.py:161-163`；`examples/offline_inference.py:27-29`
  > `if data_parallel_size == 0:`
  > `    data_parallel_size = max(1, expert_parallel_size // context_parallel_size)`
  > `dp_size = max(ep_size // cp_size, 1) if args.dp is None else args.dp`
  > `world_size = dp_size * cp_size` / `assert world_size % ep_size == 0`
- 等级：A
- 意味着什么：`world = dp × cp`，EP 通常等于 world。开 CP 的代价是**直接减少 attention 的 DP 路数**（EP8+CP2 → DP4），不是"额外加卡"。这解释了为什么 CP 只在对长序列 prefill 有收益时才值得开。
- 疑点：`assert world % ep_size == 0` 允许 world > ep（多实例/多 EP 组），但 examples 只用 world==ep 的配置。

### #4 `ParallelConfig` 里没有 `tensor_parallel_size`，"model parallel" 就等于 CP
- 证据：`deepinfer/config.py:150-174`
  > `def __init__(self, data_parallel_size, context_parallel_size, expert_parallel_size, engram_parallel_size, engine_rank, engine_group_rank, ...)`
  > `self.model_parallel_size = self.context_parallel_size`
- 等级：A
- 意味着什么：TP 不是"默认关闭"，而是**这个代码库的配置对象里根本没有这个维度**。任何"DeepInfer 用 TP=x"的说法都与源码不符。
- 疑点：无。

### #5 前向路径里没有任何 TP 型 all-reduce；集合通信只有四类
- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:61,450,943`（CP all_gather）、`deepinfer/model_executor/models/v4_modules/mlp.py:470,505`（EP dispatch/combine）、`deepinfer/model_executor/models/v4_modules/engram.py:100`（engram all_gather）；`grep all_reduce` 在 `deepinfer/model_executor/` 下只命中 `eplb`、`nccl_memory`、`deepinfer/model_executor/parallel_utils/parallel_state.py:187`（建组 warmup）；worker 侧的 all_reduce 只用于启动 warmup/计时与 KV block 数（`deepinfer/worker/model_runner.py:597,607,621`、`deepinfer/worker/base.py:265`）
- 等级：A
- 意味着什么：Transformer 层内没有"切 hidden 维再 all-reduce"的模式。`ReplicatedLinear`（`deepinfer/model_executor/models/v4_modules/linear.py:8-20`）在每张卡上都持有完整权重；被"切"的只有专家维（`ParallelRoutedLinear`，`deepinfer/model_executor/models/v4_modules/linear.py:23-34`，形状 `h_out × (h_in × n_local_experts)`）。
- 疑点：`VocabParallelEmbedding`（`deepinfer/model_executor/layers/vocab_parallel_embedding.py:10-24`）只是**沿用了 vLLM 的类名**，实现是 `gather(self.weight, input_)` 的整表索引，没有词表切分——书里不要望文生义。

### #6 设备可见性按 CP 组整块切，CP 组成员必然同机
- 证据：`examples/_config.py:55-61`
  > `local_engine_rank = engine_rank * cp_size % device_count // cp_size`
  > `str(device_offset + i) for i in range(local_engine_rank * cp_size, (local_engine_rank + 1) * cp_size)`
- 等级：A
- 意味着什么：CP 组被安排在连续的 `cp_size` 张卡内（单机 8 卡时即整机），这是 #27 的"CP all_gather 走 DeepEP BucketBuffer"能成立的前提。
- 疑点：无。

### #7 所有 NCCL/HCCL 子组是 lazy 的，必须靠一次 dummy all_reduce 真正建链
- 证据：`deepinfer/model_executor/parallel_utils/parallel_state.py:173-187`
  > `def init_device_groups(self) -> None:`
  > `    """... 全部 lazy：必须先跑一次 collective 才会真正建 comm。`
  > `    dist.all_reduce(torch.zeros(1).cuda(), group=value)`
- 等级：A
- 意味着什么：EPLB 首次重排、DeepEP buffer 分配都依赖 communicator 已存在；如果省略这一步，会在很晚的地方以段错误形式炸。属于"看懂初始化顺序"的关键细节。
- 疑点：无。

### #8 `ep_in_dp` 组被创建但前向里没有任何使用者
- 证据：`deepinfer/model_executor/parallel_utils/parallel_state.py:111-116`（创建）；`grep -rn "ep_in_dp"` 在 `deepinfer/` 下除 `parallel_state.py` 外无命中
- 等级：A
- 意味着什么：这是从"EP 可小于 DP"时代留下的备用组，当前 EP=DP 部署下无用。书里若画"每个 DP 组内部还有 EP 子组"的图会与现状不符。
- 疑点：`enp`（engram 组）则真的在用（`deepinfer/model_executor/models/v4_modules/engram.py:79`、`deepinfer/model_executor/models/v4_modules/engram_parameters.py:19`）。

### B. MoE 一层 forward 的 dispatch/combine（对应 Q2）

### #9 gate 完全在本地算，不通信：`sqrtsoftplus` 打分 + `noaux_tc` top-k + 两个 FP32 校正 bias
- 证据：`deepinfer/model_executor/models/v4_modules/moe.py:52-67`
  > `self.scoring_func = moe_cfg.get("scoring_func", "sqrtsoftplus")` / `assert self.scoring_func == "sqrtsoftplus"`
  > `self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts, dtype=torch.float32))`
  > `self.e_score_correction_bias_vl = nn.Parameter(...)`  # 图像 token 单独一套 bias
  > `self.Wgate = ReplicatedLinear(cfg.hidden_size, self.n_routed_experts, bias=False, use_fp8_gemm=False)`
- 等级：A
- 意味着什么：gate 权重在每张卡上是**全量复制**的（`n_routed_experts` 是逻辑专家数，不是本地数），且是唯一的 bf16 非量化 GEMM。视觉/文本双 bias 是 V4.1 的新东西，书里值得单独提。
- 疑点：`noaux_tc` 的 bias 更新（训练侧）不在推理仓库内。

### #10 dispatch 用的是 DeepEP `EPBuffer.dispatch`，开了 expand 布局、延迟 epilogue、按 MK 对齐
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:468-492`
  > `event_overlap = self.ep_buffer.dispatch(`
  > `    x=hidden_states, num_sms=envs.DEEPINFER_USERBUFFER_COMM_SMS, topk_idx=topk_idx, topk_weights=topk_weights,`
  > `    cumulative_local_expert_recv_stats=self._eplb_layer().expert_count,`
  > `    num_experts=self.num_physical_experts,`
  > `    num_max_tokens_per_rank=self.max_num_prefilling_tokens if input_metadata.is_prompt else self.max_num_decoding_tokens,`
  > `    expert_alignment=deep_gemm.get_mk_alignment_for_contiguous_layout(),`
  > `    do_expand=True, use_tma_aligned_col_major_sf=True, defer_epilogue=True)`
- 等级：A
- 意味着什么：`do_expand=True` 让每个 (token, expert) 对占一个槽位，MoE 因此变成"按专家分组的 GEMM"，不需要再排序；`expert_alignment` 把每个专家收到的 token 数补齐到 DeepGEMM 的 MK 对齐，这是 grouped GEMM 能用 contiguous 布局的前提。`num_max_tokens_per_rank` 传入的是**上界**，说明缓冲区按最大 batch 预留、实际 token 数可变（这是 #19 "不同 rank 不同 batch" 能成立的原因）。
- 疑点：`num_sms=envs.DEEPINFER_USERBUFFER_COMM_SMS` 未设置时默认 0，即 DeepEP 自己算理论 SM 数（`vendor/DeepEP/deep_ep/buffers/ep.py:605`）。

### #11 `defer_epilogue` 意味着 dispatch 返回的是"事件"而不是张量，取数据必须 `current_stream_wait()`
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:578-585`
  > `event_overlap = self.dispatch(hidden_states, topk_idx, topk_weights, input_metadata, None)`
  > `moe_in_hidden, _dispatched_topk_idx, dispatched_topk_weights, handle = event_overlap.current_stream_wait()`
- 等级：A（DeepEP 语义见 `vendor/DeepEP/deep_ep/buffers/ep.py:562-640`，docstring 写明 defer 时返回 `EventOverlap`，"Call `event.current_stream_wait()` to run the copy epilogue"）
- 意味着什么：dispatch 的 kernel 与后面 gate/其他计算可以重叠，真正的"收 token"发生在 wait 处。书里讲"通信-计算重叠"时，这是最具体的一处证据。
- 疑点：`_dispatched_topk_idx` 未被使用（expand 模式下 topk 权重已展开为 1D），命名上略显误导。

### #12 两个 grouped GEMM 用 handle 里的"每专家 token 数前缀和"做分组边界
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:593-598` 与 `406-430`
  > `moe_out_hidden = self.routed_experts_forward(moe_in_hidden, dispatched_topk_weights, handle.psum_num_recv_tokens_per_expert, expected_m=None)`
  > `_grouped_gemm(hidden_states, self._weight_and_sf(self.gate_up_proj_weight), gate_up_proj_out, psum_num_recv_tokens_per_expert, expected_m, self.use_fp8_act, self.use_fp4_weight)`
- 等级：A
- 意味着什么：MoE 的"分组"边界是通信的副产物——DeepEP 在 handle 里直接返回每组前缀和，GEMM 不需要自己算 histogram。这是 EP-MoE 与"先 all-to-all 再 scatter"的传统实现的关键差别。
- 疑点：`expected_m=None` 时 `_grouped_gemm` 是否走 TMA multicast 的差异，未在本次范围细读。

### #13 combine 把共享专家输出当 `bias` 传进去，共享专家因此没有独立归约
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:600-610`
  > `combine_pre_evt = self.ep_buffer.capture()`
  > `with limit_num_sms(envs.DEEPINFER_USERBUFFER_COMM_SMS, enabled=PLATFORM.is_gpu):`
  > `    shared_out = self.shared_forward(hidden_states)`
  > `event_overlap = self.combine(moe_out_hidden, shared_out, handle, combine_pre_evt)`
  > `moe_combined_hidden, _combined_topk_weights = event_overlap.current_stream_wait()`
- 等级：A（`combine(..., bias=...)` 语义见 `vendor/DeepEP/deep_ep/buffers/ep.py:770-797`："bias: 0, 1 or 2 `[num_combined_tokens, hidden]` … final bias to the output"）
- 意味着什么：这是"共享专家成本被藏进通信"的实现方式：combine 在归约 routed 专家结果的同时，把 shared 分支的结果加进同一个 epilogue。共享专家不是"另一路 all-reduce + add"。
- 疑点：`limit_num_sms` 扣掉通信 SM 这一事实强烈暗示 shared GEMM 与 combine **同时在跑**（共用 SM 池），但本次没有 profiler 数据证明两者真的并发；书里若要画"SM 分区"图应标注为推断（E）。

### #14 NVL 域内走 mega MoE：dispatch+GEMM+combine 融进一个 kernel，不再用 DeepEP per-layer buffer
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:184-188` 与 `528-559`
  > `self.use_mega_moe = ((PLATFORM.is_gpu or self.mma_type == "fp8xfp4") and self.mma_type in (...) and PG.is_within_nvl_domain(PG.expert_parallel_group()))`
  > `self.mega_moe_func(out, self._weight_and_sf(self.gate_up_proj_weight), self._weight_and_sf(self.down_proj_weight), sym_buffer, ..., cumulative_local_expert_recv_stats=self._eplb_layer().expert_count, ...)`
- 等级：A（`is_within_nvl_domain` 定义见 `deepinfer/model_executor/parallel_utils/parallel_state.py:130-134`：`group.size() <= deep_ep.get_physical_domain_size(group)[1]`）
- 意味着什么：Wide-EP 的"域"概念在代码里是**物理 NVLink 域大小**。EP64 落在同一个 NVL 域（如 72 卡域）时走 mega MoE（单 kernel、无 per-layer dispatch/combine launch 开销）；跨机则退回 #10 的 DeepEP 路径。书里讲"EP 多大算 Wide-EP"必须带上这个域边界，否则会误以为 EP 数只影响 kernel 选择。
- 疑点：`mma_type == "fp8xfp4"` 在非 GPU 平台也允许 mega MoE，NPU 侧的对应实现在 `vendor/DeepGEMM-Ascend`，未读。

### #15 两条路径都往 EPLB 喂 `cumulative_local_expert_recv_stats`
- 证据：`deepinfer/model_executor/models/v4_modules/mlp.py:474`（dispatch 路径）、`555`（mega MoE 路径）
- 等级：A
- 意味着什么：EPLB 的统计源是**通信 kernel 顺带累加的本地专家计数**，不是另跑一遍 gate 统计；所以 EPLB 的采样开销≈0，但也意味着"没被 dispatch 的 token 不计入"（如 pad 行）。
- 疑点：计数是 cumulative（DeepEP docstring：`[num_local_experts]` int，"a cumulative expert count tensor"），由 `ExpertLoadBalancer.step()` 每步清零（`deepinfer/model_executor/eplb/__init__.py:138`），语义上等价于 per-step 计数。

### C. Attention DP 下不同 batch 大小如何对齐（对应 Q3）

### #16 步型由 DP 组内的 `MIN` all_reduce 投票决定，分 ALL 点和 ANY 点两种触发
- 证据：`csrc/engine/scheduler.hpp:295-318`
  > `// NORMAL 引擎：决定这一步全组做 prompt 还是 decode。dp 内必须同型，靠投票统一。`
  > `bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);`
  > `bool is_all_point = (decode_step % prefill_attempt_interval == 0);`
  > `do_prefill = (_dp_allreduce(want ? 0 : 1, c10d::ReduceOp::MIN) == 0); // ANY：任一 rank 想做即触发`
  > `do_prefill = (_dp_allreduce(want ? 1 : 0, c10d::ReduceOp::MIN) == 1); // ALL：全员想做才触发`
  > （上面两行分别来自 ANY 分支与 ALL 分支，不是顺序执行）
- 等级：A（`_dp_allreduce` 实现在 `csrc/engine/scheduler.hpp:168-182`，gloo/NCCL 两种后端按组选）
- 意味着什么：**这正是"Attention DP 各 rank batch 不一样大"的答案：不同步大小，只同步"这一步是 prefill 还是 decode"。** ALL 点保证不会为一个 rank 的 prefill 让全组陪跑；ANY 点给孤立 rank 兜底（最多等 `prefill_any_attempt_interval` 个 decode 步）。注意 `prompt step` 会把 `decode_step` 清零（同文件 305-307 注释），所以连续 prefill 也要走 ALL 点。
- 疑点：`prefill_attempt_interval` / `prefill_any_attempt_interval` 的默认值在 `SchedulerConfig` 里（`deepinfer/config.py`），本次未逐值核对。

### #17 第二轮投票：防止"全员空手"的空 prompt step 造成死循环
- 证据：`csrc/engine/scheduler.hpp:326-341`
  > `bool any_scheduled = (_dp_allreduce(outputs ? 0 : 1, c10d::ReduceOp::MIN) == 0);`
  > `if (!any_scheduled) { return _schedule_decode(acquire); }`
  > `// 组内确有 rank 在做 prompt：本 rank 陪跑空 prompt step 保持步型一致`
  > `return _make_outputs({}, true, 0, {}, {}, {}, 0);`
- 等级：A（注释是中文原文）
- 意味着什么：投票只看"waiting 非空"，但真正调度可能空手（slot 满、KV 块不够）。没有这轮兜底，会出现"空 prompt 步清零 decode_step → 下一步又投票 → 又空手"的活锁。这是一个很值得写进书的工程细节。
- 疑点：无。

### #18 "陪跑"的 rank 跑的是 1-token 的 placeholder 请求
- 证据：`deepinfer/worker/worker.py:107,125-127`；`deepinfer/worker/model_runner.py:386-403`；`deepinfer/engine/llm_engine.py:972`
  > `is_placeholder=scheduler_outputs.is_empty()`
  > `seq_metadata_list = [self.model_runner._placeholder_seq_metadata(is_prompt)]`
  > `return SequenceMetadata(request_id=PLACEHOLDER_REQUEST_ID, seq_id=0, is_prompt=is_prompt, seq_data=SequenceData([0]), ..., seqlen_q=1, seqlen_k=1, ...)`
- 等级：A
- 意味着什么：**EP 的 dispatch/combine 是集合通信，每个 rank 每步必须进入相同次数的 kernel。** placeholder 步就是这个"最小合法参与"：1 个 token、block_table=[0]、采样温度 0，结果被丢弃。书里画时序图时，空 rank 那一列不是空的，而是一个 1-token 的假 batch。
- 疑点：placeholder 步是否也走 cuda graph（`is_prompt=True` 时 `run_cuda_graph=False`，故 prefill 陪跑走 eager），未逐一验证。

### #19 各 rank 的 batch 大小不对齐：decode 各选各的 cuda graph 桶，桶内用 padding seq 补齐
- 证据：`deepinfer/worker/model_runner.py:182-191`；`deepinfer/worker/cuda_graph_runner.py:41-51`
  > `graph_size, cuda_graph_used = self.cuda_graph_runner.select(total_q)`
  > `self._prepare_input_tensors(seq_metadata_list + self.cuda_graph_runner.padding_seq_metadata_list[: graph_size - total_q], ...)`
  > `def build_graph_batch_sizes(max_size, interval): ... 2 的幂 ... 再按 interval 等距 ... 最后 max_size`
- 等级：A
- 意味着什么：cuda graph 的 batch 桶是"2 的幂 + 等距"的稀疏集合，每个 DP rank 按自己的 `total_q` 选桶、自己 pad。**没有任何跨 rank 的 batch 对齐逻辑**，因为 DeepEP 的 dispatch 是变长 all-to-all（token 数由 handle 里的 metadata 携带）。
- 疑点：桶大小不同时 DeepEP 内部是否需要等长 buffer，取决于 `num_max_tokens_per_rank`（#10 传的是上界），本次未在 DeepEP 侧验证。

### #20 唯一的"跨 DP 数值对齐"是 KV block 数：按 world 组取平均（向下取偶）
- 证据：`deepinfer/worker/base.py:263-265`
  > `num_gpu_blocks = torch.tensor(num_gpu_blocks, device="cuda", dtype=torch.int64)`
  > `dist.all_reduce(num_gpu_blocks, op=dist.ReduceOp.SUM, group=PG.world_group())`
  > `num_gpu_blocks = num_gpu_blocks.item() // PG.world_group().size() // 2 * 2`
- 等级：A
- 意味着什么：显存不同的卡会被拉平到同一个（偏保守的）块数，避免某个 rank 先 OOM。这是启动期一次性对齐，不是每步对齐。
- 疑点：用平均而不是最小值，意味着显存小的卡仍可能比平均值更早吃紧；推测配合 `gpu_memory_utilization` 使用（E）。

### #21 结束条件用 `MAX` all_reduce，保证没有 rank 提前退出
- 证据：`deepinfer/engine/llm_engine.py:665-679`
  > `if dp_group.size() > 1:`
  > `    unfinished = torch.full((1,), unfinished, dtype=torch.int64, device=device)`
  > `    dist.all_reduce(unfinished, op=dist.ReduceOp.MAX, group=dp_group)`
- 等级：A
- 意味着什么：请求分配不均时，先做完的 rank 不能退出循环，否则其它 rank 的 EP 集合通信会挂死。这条与 #16/#18 一起构成"DP 不均衡下的三条保命规则"。
- 疑点：`DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 打开时改用 gloo 的 `cpu_data_parallel_group`（同函数 667-672），说明投票/统计走 CPU 组是为了不与 GPU 流同步。

### #22 文档对"DP 内还要压预算"的描述与代码一致（B 级旁证）
- 证据：`doc/inference_deployment_qa.md:115`
  > "The budget itself is further reduced by an all-reduce MIN within the DP group."
- 等级：B
- 意味着什么：除了步型投票，prefill 的时间预算也在 DP 组内取 MIN，进一步说明"DP 之间靠归约对齐的是**决策**，不是 batch"。这条与 #16 相互印证。
- 疑点：`compute_quota_limit` 的实现在 C++ 调度器内，本次未读到对应 all_reduce 调用点。

### D. "balanced CP" / mirror CP（对应 Q4）

### #23 代码里没有 "balanced CP" 这个词；对应实现叫 **mirror CP**，块分配是 zigzag
- 证据：`deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py:143-152`（纯 Python 参考实现，注释"directly expands the zigzag definition"）
  > `ps = ps_all[i]`（其值见同文件 `135`：`(sq + 2*cp_size - 1) // (2*cp_size)`）
  > `for rank in range(cp_size):`
  > `    for half, block in enumerate((rank, 2 * cp_size - 1 - rank)):`
  > `        local_idx = ps * block + j`
  > `            reverse_index[cu_q[i] + local_idx] = rank * total_per_rank + cu_elems[i] + half * ps + j`
- 等级：A（`grep -ri "balanced"` 在 `deepinfer/`、`doc/` 下无 "balanced CP" 命中；只有 `deepinfer/model_executor/parallel_utils/userbuffer.py:218` 的 `get_balanced_sm_locality_domains`，那是 SM 局部性，与 CP 无关）
- 意味着什么：**每个序列被切成 `2*cp` 个块，rank r 拿第 r 块和第 `2cp-1-r` 块。** "mirror" 指这个对称性；"balanced" 平衡的是 **causal attention 的计算量**（以及每 rank 的 token 数）。
- 疑点：无。

### #24 "balanced" 平衡的到底是什么：配对块的因果 key 长度之和与 rank 无关
- 证据：`deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py:153-156`
  > `delta = sk - sq + local_idx + 1 if local_idx < sq else 0`
  > `ke[ri].append(cu_k_cn[ri][i] + delta // ratio)`
  > `ks[ri].append(cu_k_cn[ri][i])`
- 等级：E（由上面三行 + #23 的块编号推出，未跑数值验证）
- 推理过程：块 b 的第 j 个 token 的"可见 key 数" `delta+1 = sk-sq + ps*b + j + 1`；rank r 的两块是 b=r 与 b=2cp-1-r，`Σ_j (ps*b + j)` 在 b 取这两个值时，`ps²*b` 项相加为 `ps²(2cp-1)`（与 r 无关），`Σ_j j` 项相同。故每 rank 的 `Σ delta` 恒定 → **attention 的 (query,key) 对总数在 cp 个 rank 间相等**（`sq` 能被 `2cp` 整除时严格成立；否则末块不满，差 1 块以内）。若按连续块切（rank r 只拿块 r），rank 0 只需看很少的 key、rank cp-1 要看几乎全部，最慢 rank 决定步时，CP 的收益会被吃掉大半。
- 意味着什么：这是"为什么必须 zigzag 而不能顺序切"的定量解释，是书里 CP 一节的论证核心。
- 疑点：`ps = ceil(sq/(2cp))` 的向上取整会让各 rank 差 1 个 token；尾巴块可能越界（`local_idx < sq` 判断，否则 `mirror_cp_index=-1`），实际不均衡度需实测。

### #25 mirror CP 只在 prefill 生效，函数里直接 `assert self.is_prompt`
- 证据：`deepinfer/model_executor/input_metadata/input_metadata.py:169-180`
  > `@cpu_timer.record("prepare_for_mirror_cp")`
  > `def prepare_for_mirror_cp(self) -> None:`
  > `    """Prepare each sequence's CP-mirror layout (CP > 1, prefill only)."""`
  > `    assert self.is_prompt`
- 等级：A
- 意味着什么：decode 阶段没有 mirror 布局（attn 里 `use_cp = mirror_cp_reverse_index is not None`，decode 恒为 False，`deepinfer/model_executor/models/v4_modules/attn.py:934-936`）。所以 CP 在本仓库是**长序列 prefill 加速器**，decode 时 cp 个 rank 是纯复制关系。
- 疑点：decode 下 cp>1 的 KV cache 是否每 rank 各存一份完整副本（意味着 CP 在 decode 只浪费显存不省时间），需在 cache 章节确认。

### #26 模型入口先把 token 重排成 mirror 布局，出口再反向还原
- 证据：`deepinfer/model_executor/models/deepseek.py:169-170` 与 `242-244`
  > `input_metadata.prepare_for_mirror_cp()`
  > `hidden_states = gather(hidden_states, input_metadata.mirror_cp_index)`
  > `with userbuffer.cp_session():`
  > `    gathered_hidden_states = userbuffer.cp.all_gather(hidden_states)()`
  > `    hidden_states = gather(gathered_hidden_states, input_metadata.mirror_cp_reverse_index)`
- 等级：A
- 意味着什么：**CP 对模型的其余部分完全透明**：层内看到的永远是"本 rank 的那 2 个块"，出口处 all_gather + 反向索引恢复原始 varlen 顺序，sampler 完全不知道 CP 存在（`deepinfer/worker/model_runner.py:238-240` 注释："sampler 在 CP 之后，CP 对 sampler 透明"）。
- 疑点：`gather` 是 TileLang 的索引 kernel（`deepinfer/model_executor/layers/tilelang_ops/gather.py`），不是通信，成本与 `total_q` 成正比。

### #27 CP 的 all_gather 不是 NCCL，而是 DeepEP 的 `BucketBuffer`，且返回的是"延迟 getter"
- 证据：`deepinfer/model_executor/parallel_utils/cp.py:9-51`
  > `class CPManager: """Manage CP communication storage, sessions, and all-gather operations.`
  > `Wait callbacks must be consumed within the session. Gathered tensors alias the buffer storage, which is reused by the next session."""`
  > `handle = self._buffer.all_gather(x)`
  > `return lambda: handle.wait().view(gathered_shape).flatten(0, 1)`
- 等级：A
- 意味着什么：`all_gather` 立即返回一个 lambda，真正的 `wait()` 由调用方在需要时触发（`deepinfer/model_executor/models/v4_modules/attn.py:450` 的 `wait_kv_swa()`、`943` 的 `wait_main_k()`），从而把 gather 与 Q 投影重叠。**别名语义是坑**：gathered 张量指向 buffer 存储，下一个 session 会被覆盖，所以必须在 session 内消费完（docstring 明说）。
- 疑点：`BucketBuffer` 的 `num_bytes` 由 `allocate_cp` 预算（#44），分配不足时的行为未读。

### #28 CP 下 KV 压缩在"完整序列"上做，Q 侧切分而 KV 侧不切
- 证据：`deepinfer/model_executor/input_metadata/input_metadata.py:230`；`deepinfer/model_executor/models/v4_modules/attn.py:782-785`
  > `# With CP, compression runs on the original full sequence, so no reordering is needed.`（positions_c 的 docstring）
  > `# CP: launch the all_gather of the compressor input first, overlapping it with the Q projection below.`
  > `wait_projected_kv = _make_cp_gather_getter(projected_kv, use_cp) ...`
- 等级：A
- 意味着什么：CP 省下的是 **Q 侧 attention 计算**，KV 的投影/压缩是每 rank 对全序列各算一遍（然后 all_gather 结果）。所以 CP 的收益随 `sq` 增长但**不是线性**，且 KV cache 显存不随 cp 减少。书里不能写成"CP 把 KV 也切了"。
- 疑点：`kv_swa` 的 gather 目标 buffer 是 `kv[...]`（`deepinfer/model_executor/models/v4_modules/attn.py:825-830`），说明 SWA 的 K/V 也是全量物化后再切片使用。

### #29 CP 组内的 metadata 广播：driver 算，follower 收，非张量走共享内存
- 证据：`deepinfer/worker/model_runner.py:200-212`；`deepinfer/model_executor/parallel_utils/communication_op.py:51-82`
  > `if PG.context_parallel_group().size() > 1: broadcast_tensor_dict(input_metadata_dict, src=0)`
  > `"""Broadcast a dict across the CP group: cuda tensors via NCCL, everything else via shm object broadcast."""`
- 等级：A
- 意味着什么：非 driver 的 CP rank 不跑调度器、不建 metadata，只接收。`num_reqs` 的注释（`deepinfer/model_executor/input_metadata/input_metadata.py:52-53`）直说"Needed by non-driver CP ranks"。这是 CP 与 DP 的结构性差别：DP rank 各有一套完整引擎，CP rank 是"半个 worker"。
- 疑点：`to_flatten_dict` 的字段清单会随模型演进，属于易变接口。

### #30 官方文档明确写"TP 和 CP 目前不用"，与代码里 CP 完备实现并存
- 证据：`doc/inference_deployment_qa.md:53`
  > "- TP and CP are not used for now."
- 等级：B
- 意味着什么：代码有 CP（长上下文 prefill 可用，`examples/offline_inference.py:15` 暴露 `--cp`），但**生产部署不用**。书里应写成"代码具备、生产未启用"，并给出触发条件（超长 prefill）而非暗示生产在用。
- 疑点：文档日期与该 commit 的对应关系未确认；`--cp` 在 examples 里的默认值是 1。

### E. EPLB：策略、搬运、周期（对应 Q5）

### #31 冗余专家数是"不小于下限、且让 physical 能被 ep_size 整除"的最小值
- 证据：`deepinfer/model_executor/eplb/__init__.py:21-28`
  > `padding = -(num_routed_experts + min_num_extra_experts) % ep_size`
  > `return min_num_extra_experts + padding`
- 等级：A
- 意味着什么：文档说"从每卡 4 个（prefill）/ 1 个（decode）冗余专家起步，向上取整到能整除"（`doc/inference_deployment_qa.md:99,103`），代码就是这个公式。**用户配的是下限，实际值可能被 pad 大很多**：当 `n_routed_experts` 本身是 `ep_size` 的整数倍时，`-(N+1) % 64 = 63`，EP64 下 1 个下限会被 pad 成 64 个。
- 疑点：文档说 EP64 时"四舍五入后每卡 7 个专家"，按本公式反推 `n_routed_experts = 384`（`(384+1)%64=1 → extra=64 → 448/64=7`）。仓库内没有 HF config，**未能确认 V4.1 的逻辑专家数**；若实际是 256，则每卡应为 5 个。这条数字差异需要用模型 config.json 复核。

### #32 两级贪心之一：先给热点 logical expert 加冗余副本（逐个副本贪心）
- 证据：`deepinfer/model_executor/eplb/policy.py:99-123`
  > `# 贪心分配冗余副本：每轮选一个 expert 复制。`
  > `score = tokens_per_expert_all_diff / logical_count`
  > `score_added = tokens_per_expert / (logical_count + 1)`
  > `values, indices = score.max(-1, keepdim=True)`
  > `redundancy_indices = values.sum(0).argmin(-1)  # [num_moe_layers]，本轮被复制的 logical 编号`
- 等级：A
- 意味着什么：目标是"把窗口内**每副本峰值负载**压下去"：给 expert j 加一个副本后，j 的每副本负载从 `T/c` 变成 `T/(c+1)`；选"加上去之后全局最大每副本负载最小"的那个 j。`tokens_per_expert_all_diff` 加的 `arange*1e-4` 是为了打破并列（注释："Values in score must be different from each other"）。
- 疑点：这是**逐层独立**的决策（`sum(0)` 是在 step 维求和，不是层维），所以不同层可以有不同副本布局；书里若画"全局统一的冗余专家表"会画错。

### #33 两级贪心之二：把物理槽位装箱到各 rank，目标是 `min Σ_step max_rank(load)`
- 证据：`deepinfer/model_executor/eplb/policy.py:43-70`
  > `# 优化目标是逐 layer 独立的 ... 使得 sum_{step i} max_{rank j} (sum_{physical expert e} [e 被分到 rank j] e 在 step i 的负载) 尽量小`
  > `torch.add(loads, vector.unsqueeze(1), out=candidates)`
  > `torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)`
  > `torch.sum(candidates, dim=-1, out=scores)`
  > `scores.masked_fill_(occupancy == capacity, float("inf"))` / `rank = scores.argmin(dim=-1)`
- 等级：A（注释为中文原文）
- 意味着什么：**不是**"按平均负载贪心"，而是"按历史每一步的最慢 rank 之和贪心"。这解释了为什么它是 `O(num_physical)` 次 `argmin`、为什么保留 `num_steps` 维（不同 step 的负载互补性会被奖励）。`occupancy == capacity` 的 mask + `argmin` 同分取最小 rank，保证确定性（#36）。
- 疑点：复杂度是 `O(L · P · R · S)`（层×物理槽×rank×step），EPLB 周期短的部署里这段 CPU 时间是否可忽略，需实测（文档没给）。

### #34 统计窗口与重分配周期全部来自环境变量，不在模型 config 里
- 证据：`deepinfer/environ.py:113-121`；`deepinfer/model_executor/eplb/__init__.py:102-111`
  > `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE = EnvVar.int(default=100)`
  > `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL = EnvVar.int()`（默认 0 = 关闭）
  > `DEEPINFER_MOE_LB_STAT_INTERVAL = EnvVar.int(default=128)`
  > `if self._rearrangement_interval != 0 and not (0 < expert_count_window_size <= self._rearrangement_interval): raise ValueError(...)`
- 等级：A
- 意味着什么：**窗口必须 ≤ 周期**，否则启动即报错。默认"窗口 100、周期 0（关）"，所以 EPLB 默认不开，必须显式设 `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`。examples 里为了跑通设成 30（`examples/_config.py:82`），**这不是生产值**——文档给的是 prefill 1000 / decode 10000（#40）。
- 疑点：`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` 在 examples 里设 32（`examples/_config.py:79`），同样是示例值。

### #35 周期计数只数"非 prefill 步"，并支持 prefill 后立刻重排
- 证据：`deepinfer/model_executor/eplb/__init__.py:83-84,146-165`
  > `self._schedule_rearrangement_after_prefill = is_normal_engine and envs.DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE`
  > `self._skip_rearrangement_on_prefill = is_normal_engine and not self._schedule_rearrangement_after_prefill`
  > `if is_prompt and self._schedule_rearrangement_after_prefill: self._rearrangement_step = self._rearrangement_interval - 1`
  > `self._rearrangement_step += 1`
- 等级：A
- 意味着什么：NORMAL 引擎里 prefill 步默认**不参与计数**（`_skip_...`），所以 "每 1000 步" 是 1000 个 decode 步；开了 `REARRANGE_AFTER_PREFILL` 则一个 prefill 步会把计数器顶到"下一步就重排"。这解释了为什么 prefill 与 decode 需要不同周期（#40）。
- 疑点：PROMPT/COMPLETION 引擎（`is_normal_engine=False`）下所有步都计数，与 NORMAL 语义不同。

### #36 首次重排提前到 1/4 周期，且窗口是环形缓冲
- 证据：`deepinfer/model_executor/eplb/__init__.py:113-123,152-154`
  > `self._expert_count_window = torch.zeros(expert_count_window_size, num_moe_layers, num_local_physical_experts, dtype=torch.int)`
  > `initial_rearrangement_delay = max(1, self._rearrangement_interval // 4)`
  > `self._rearrangement_step = max(0, self._rearrangement_interval - initial_rearrangement_delay)`
  > `self._expert_count_window[window_index].copy_(self.expert_count)` / `self._expert_count_window_index = (window_index + 1) % ...`
- 等级：A
- 意味着什么：冷启动时不均衡最严重，所以第一次提前；窗口是**定长 step 维环形缓冲**，不是时间窗。窗口里的"step"是"有 token 的步"，空步（placeholder）也会写一行零。
- 疑点：窗口按步不按时间，导致长 prefill 步与短 decode 步在窗口里权重相同（每步一票），这可能是有意的（避免长步主导），但代码没写理由。

### #37 重排时先按本地 physical→logical 映射聚合成 logical 计数，再 EP 组 all_reduce
- 证据：`deepinfer/model_executor/eplb/__init__.py:238-260`
  > `local_physical_to_logical_map = self.physical_to_logical_map[:, rank*n_local : (rank+1)*n_local]`
  > `logical_expert_token_counts.scatter_add_(-1,`
  > `    local_physical_to_logical_map.to(torch.int64).expand_as(self._expert_count_window), self._expert_count_window)`
  > `dist.all_reduce(logical_expert_token_counts, group=ep_group)`
  > `next_physical_to_logical_map, next_logical_to_physical_map, next_logical_count = rebalance_experts_global(logical_expert_token_counts, num_physical_experts, num_local_physical_experts)`
- 等级：A
- 意味着什么：**策略是全局的、结果也是全局一致的**（所有 rank 算同一份 map），所以不需要广播布局，只需要 all_reduce 计数。副本被分到哪个 rank 由算法统一决定。
- 疑点：all_reduce 是 SUM，若某 rank 的 `expert_count` 有重复统计（同一 token 被多个副本计数）会重复计入；按 #15 的语义不会，但需 DeepEP 侧确认 expand 模式下计数是按物理槽位。

### #38 权重重排：同 NVL 域走 `symm_mem` + CUDA IPC 直读，跨域走 NCCL 分桶 P2P
- 证据：`deepinfer/model_executor/eplb/__init__.py:195-220`；`deepinfer/model_executor/eplb/shuffle.py:150-176`
  > `self._intranode_shuffle = PG.is_within_nvl_domain(ep_group) and PLATFORM.is_gpu`
  > `self._symmetric_weight_buffers = [symm_mem.empty(weight.shape, ...) for weight in self._expert_weights[0]]`
  > `self._symmetric_memory_handles = [symm_mem.rendezvous(buffer, ep_group) for buffer in ...]`
  > `"""同 expert_model_parallel_shuffle_inplace，但限单机：权重经 IPC 镜像直接读对端，无 P2P 通信。"""`
- 等级：A（文档侧旁证 `doc/inference_deployment_qa.md:97`："expert weights are shuffled within the group over NVLink `symm_mem`"）
- 意味着什么：**同域重排是"读对端显存"而不是"发消息"**：每卡把自己一层的权重拷进对称缓冲，两次 `barrier` 后，缺的专家直接从配对 rank 的镜像 `copy_` 过来。跨机才退回 `dist.batch_isend_irecv`，并按 `(对端 rank 差 mod DEEPINFER_REARRANGE_P2P_GROUPS)` 分桶（`deepinfer/model_executor/eplb/shuffle.py:113-114`），避免同时开太多 P2P。
- 疑点：`symm_mem` 需要 NVLink + 同机；NPU 不支持（`deepinfer/model_executor/eplb/__init__.py:199` 的 `TODO(NPU)`）。镜像只放**一层**权重（注释"镜像只需容纳一层权重，shuffle 时逐层刷新复用"），所以重排耗时 ≈ `num_layers × 一层权重/NVLink 带宽`。

### #39 重排的一致性靠两次 barrier；期间不能有前向在跑
- 证据：`deepinfer/model_executor/eplb/shuffle.py:191-195`
  > `ep.barrier().wait()  # 等所有 rank 读完上一层的镜像`
  > `for m, weight in zip(mirror, weights, strict=True): m.copy_(weight)`
  > `ep.barrier().wait()  # 等所有 rank 的镜像就绪`
- 等级：A
- 意味着什么：EPLB 是**全组停顿**的操作（文档没有给出重排耗时数字，`doc/inference_deployment_qa.md` 全文只描述机制、无耗时数据）。周期设得越大越省，但冷启动/负载漂移时更被动——这正是 prefill 1000 / decode 10000 的原因（decode 步短、停顿相对更贵）。
- 疑点：`allow_cuda_sync()` 装饰器（`deepinfer/model_executor/eplb/__init__.py:222`）说明这是少数允许同步的地方，但没说明是否会与 ZOS 的异步调度冲突。

### #40 文档给的生产参数与实测 `moe_lb`
- 证据：`doc/inference_deployment_qa.md:99` 与 `103`
  > "Prefill settings: start from 4 redundant experts per GPU ..., a 100-step statistics window, and a re-assignment every 1,000 steps. Measured `moe_lb` (slowest rank / mean) = 1.074 (EP4DP4) / 1.045 (EP2DP2)."
  > "Settings: start from 1 redundant expert per GPU ... at EP64 ... 7 experts per GPU after rounding ... a 1,000-step statistics window and a re-assignment every 10,000 steps. Measured `moe_lb` = 1.090."
- 等级：B
- 意味着什么：**prefill 与 decode 的窗口/周期差 10 倍，冗余专家数也不同**，这是"EPLB 在 prefill 和 decode 下配置不同"的官方口径。`moe_lb` 越小越均衡，1.09 意味着最慢 rank 比平均多扛 9% 的 token。
- 疑点：`moe_lb` 的口径在代码里有**两套**（见 #41），文档没说用的是哪套；且没给硬件/负载条件，不能与其它来源的数字直接比。

### #41 `moe_lb` 的代码口径是"窗口内 Σmax / Σmean"，且另有一个反过来的 `rank_balance`
- 证据：`deepinfer/engine/stats_logger.py:238-241`；`deepinfer/model_executor/eplb/__init__.py:190-193,303-311`
  > `def _moe_lb_ratio(self) -> float: ... return self._moe_lb.sum(lambda p: p[1]) / max(1, self._moe_lb.sum(lambda p: p[0]))`
  > `self.moe_load_balance_ratio = (tokens_per_rank.mean(dim=0).sum().to("cpu"), tokens_per_rank.max(dim=0).values.sum().to("cpu"))`
  > `rank_balance = mean_rank_load / (mean_peak_rank_load + 1e-6)`
  > `"[EPLB] rearranged: ... rank_balance=%.4f (mean=%.1f, peak_mean=%.1f) ..."`
- 等级：A
- 意味着什么：日志里的 `moe_lb = Σ_step max_rank(avg) / Σ_step mean_rank`，是**按 token 加权**的比值；而 EPLB 重排日志里的 `rank_balance` 是"先算每步最慢 rank，再对步取平均"的倒数。两者分母不同、方向相反，**不能混用**。已知"知乎与 QA 的 moe_lb 数字有差异"，口径不同（哪套指标、哪个窗口、EP/DP 配置）就是首要嫌疑。
- 疑点：`_moe_lb` 是**按 step 年龄裁剪的滑动窗口**，裁剪上限取 `max(日志间隔, DEEPINFER_MOE_LB_STAT_INTERVAL)`（`deepinfer/engine/stats_logger.py:455-457`），因此**同一负载在不同日志配置下会得到不同 moe_lb**。这一点文档没写，属于书里可以点出的"数字陷阱"。

### #42 EPLB 的单元测试锁的是"结构自洽 + 确定性 + tie-break"，不是最优性
- 证据：`tests/models/test_eplb_policy.py:33-42,100-106`
  > `"""Structural self-consistency: phy2log and log2phy are inverses, log_count matches the valid slots, and the total replica count equals the physical expert count."""`
  > `assert log_count.sum(-1).equal(torch.full((num_layers,), num_physical_experts, dtype=torch.int))`
  > `# when placing expert 1 both GPUs score the same, and rank 0 is still preferred even though its accumulated load is higher.`
- 等级：A
- 意味着什么：测试用例覆盖 8 种 `(steps, layers, logical, physical, local)` 组合，包括"冗余数 > 逻辑数"的极端情形（`(2,1,4,16,1)`）与全零历史；tie-break 规则（同分取最小 rank、同负载保留原槽位顺序）被显式断言。这说明**布局稳定性是设计要求**——否则每次重排都会抖动权重。
- 疑点：没有性能/最优性断言；也没有"重排后 moe_lb 一定下降"的回归测试。

### #43 权重重排的分布式测试用 8 卡跑 200 步随机布局，比对两种实现
- 证据：`tests/distributed/test_expert_shuffle.py:20-22,86-89,133-144`
  > `ep_size = 8` / `num_local_experts = 2` / `num_experts = 5`
  > `current_experts_weight_ipc_handle = [CudaIPCWrapper(tensor) for tensor in current_experts_weight]`
  > `def test_expert_shuffle(): """NCCL batched P2P shuffle ..."""` / `def test_expert_shuffle_intranode(): """Intranode (CUDA IPC) shuffle ..."""`
- 等级：A
- 意味着什么：这里同时是**两种搬运实现的规格说明**：`p2p` 走 NCCL 组，`intranode` 走 `CudaIPCWrapper` 手工建 IPC 镜像（注意：是测试代码自己建的，生产走 `symm_mem.rendezvous`，#38）。5 个专家铺到 16 个槽位，保证"同一专家多副本"的分支被覆盖。
- 疑点：测试用 `dist.broadcast` 由 rank0 生成布局，没有覆盖"各 rank 独立算出不同布局"的失效模式（生产中靠 #37 的全局 all_reduce 保证一致）。

### F. userbuffer（对应 Q6）

### #44 `userbuffer` 是"通信缓冲区 + session 管理"的单例，装四类缓冲：EP / CP / Engram / 对称缓冲
- 证据：`deepinfer/model_executor/parallel_utils/userbuffer.py:54-127,183-225`；`deepinfer/model_executor/models/v4_modules/mlp.py:303-324`；`deepinfer/model_executor/models/deepseek.py:412-419`
  > `def allocate_ep(...)-> EPBuffer`（DeepEP dispatch/combine 用）
  > `def allocate_cp(...)` → `self.cp = CPManager(cp_group, num_bytes)`（#27 的 BucketBuffer）
  > `def allocate_sym_buffer(self, ep_group, num_physical_experts, max_num_tokens, ..., mma_type, activation)` → `deep_gemm.SymmBuffer`（mega MoE 用）
  > `max_ag_bytes_per_token = self.hidden_size * 2  # hidden_state in sample`
- 等级：A
- 意味着什么：**"userbuffer" 不是一种通信算法，而是"这些通信库需要预分配显存"的统一入口**。它解决的是：DeepEP/DeepGEMM 的 buffer 必须在 cuda graph capture 之前分配好，且大小要按 `max_num_prefilling_tokens / max_num_decoding_tokens` 的上界算（`deepinfer/model_executor/model_loader.py:119-120`）。
- 疑点：`allocate_engram`（127-181）属 engram 章节范围，未细读。

### #45 `limit_num_sms` 用"扣减 SM"的方式给通信让路，是重叠的执行层保证
- 证据：`deepinfer/model_executor/parallel_utils/userbuffer.py:40-51`
  > `def limit_num_sms(num_sms: int, enabled: bool = True) -> Generator[None]:`
  > `tile_kernels.set_num_sms(prev_sms - num_sms)`
  > `deep_gemm.set_num_sms(prev_sms - num_sms)`
  > `finally: tile_kernels.set_num_sms(prev_sms); deep_gemm.set_num_sms(prev_sms)`
- 等级：A
- 意味着什么：这是全局状态（进程级 SM 配额），不是 kernel 参数。所以 #13 里 shared expert 的 GEMM 只拿到 `总SM - 通信SM`，与 combine 真正并行。书里讲"计算通信重叠"时，这是最容易被忽略的一层（OS/驱动层没有隔离，靠 SM 配额手工分）。
- 疑点：`DEEPINFER_NCCL_NUM_SMS`（默认 32，`deepinfer/environ.py:108`）管的是 NCCL 的 CTA 数，与 `DEEPINFER_USERBUFFER_COMM_SMS`（默认 0=auto）是两套，别混。

### #46 `nccl_memory.py` 只是监控，不参与通信
- 证据：`deepinfer/model_executor/parallel_utils/nccl_memory.py:13-14,37-58`
  > `class NCCLMemoryStatsLogger: """Sample driver-local communicators and log snapshots when they change."""`
  > `total = backend.memory_stats()["total"]`
- 等级：A
- 意味着什么：按 `DEEPINFER_NCCL_MEMORY_SAMPLE_INTERVAL_STEPS` 采样每个 communicator 的显存占用并打日志（`total=... GiB, communicator_count=...`）。这是排查"Wide-EP 下 NCCL buffer 吃显存"的工具，不是数据通路。
- 疑点：无。

### G. Wide-EP（对应 Q2/Q5 的规模维度）

### #47 每卡专家数 = `physical_experts / ep_size`，因此每步 HBM 读随 EP 线性下降
- 证据：`deepinfer/model_executor/models/v4_modules/moe.py:91-95`；`deepinfer/model_executor/models/v4_modules/linear.py:23-34`
  > `self.n_local_routed_experts = self.n_routed_experts // self.ep_size`
  > `assert self.n_routed_experts % self.ep_size == 0`
  > `super().__init__(h_out, h_in * n_local_experts, bias=False)`  # 每卡只存本地专家的权重
- 等级：A（"每步 HBM 读 ∝ 1/EP"是 E，推理如下）
- 推理过程：一层 MoE 的权重字节数 = `n_local_experts × 3 × hidden × moe_inter`（gate_up 2 份 + down 1 份）× 精度字节。一步 decode 里每个 rank 必须把它**本地**所有专家的权重读一遍（token 少，算术强度低，是纯带宽受限）。EP 翻倍 → 本地权重减半 → 每步权重读减半。这就是文档"Wide-EP substantially reduces per-step memory traffic in the MoE expert part"（`doc/inference_deployment_qa.md:57`）的机制。
- 意味着什么：Wide-EP 的收益来自**用通信换带宽**：token 要跨更多 rank 走 all-to-all，但每卡要读的权重更少。存在一个最优 EP（通信增量 = 带宽节省），所以文档说 decode 用"能拿到的最大的 EP"、prefill 用"能装下 HBM 的最小的 EP"（`doc/inference_deployment_qa.md:47-49`）。
- 疑点：没有实测曲线（`doc/images/wide_ep_decode_pareto.png` 是唯一数据点来源，B 级）；本笔记无法给出 EP8 vs EP64 的具体倍数。

### #48 文档的 EP 建议与"域"约束
- 证据：`doc/inference_deployment_qa.md:47-49`
  > "Prefill: EP2–8 / DP2–8; use the smallest EP/DP that HBM allows."
  > "Decode: EP64–128 / DP64–128 (Wide-EP); use the largest EP/DP that the available GPUs allow."
- 等级：B
- 意味着什么：prefill 是 compute-bound（EP 大了通信占比上升、收益小），decode 是 memory-bound（EP 大了省带宽、收益大）。这两个数字与 #14 的 NVL 域判断共同决定走 mega MoE 还是 DeepEP：EP64 在 NVL72 域内可以走 mega MoE，EP128 跨域则走 DeepEP。
- 疑点：文档没说 EP64/128 对应的物理域大小；EP128 必然跨域（当前单域最多 72/64），此时 mega MoE 关闭，性能特征会变，书中应提示这一点。

### #49 单实例单一通信域 → 单卡故障整实例不可用
- 证据：`doc/inference_deployment_qa.md:211`
  > "In a large-EP deployment, all GPUs of an instance form a single communication domain, so a single failed GPU makes the whole instance unusable. On restart ... the platform re-determines the EP size from all currently healthy GPUs"
- 等级：B
- 意味着什么：EP 组的"连续段 + 全实例一个域"（#2）在可靠性上的代价。重启后 EP size 变小 → `physical % ep_size` 的整除约束（#31）会被重新满足，但**冗余专家数与布局会变**，这也是 EPLB 必须支持任意 ep_size 的原因之一。
- 疑点：代码里是否有"运行中改 ep_size"的路径，未确认（推测没有，需重启，E）。

## 3. 必答问题逐条回答

**Q1 进程组如何划分（DP、EP、CP）？**
- 全局 rank = `((replica*dp + r)*cp + c)`；world 组 = 全部 rank（#1）。
- **DP 组**：固定 `(replica, cp)` 坐标、取遍 dp 下标 → 编号上"隔 cp 个取一个"（#1）。
- **CP 组**：固定 `(dp)` 下标、取连续 cp 个 rank；CP 成员被安排在同机连续 GPU（#6）。
- **EP 组**：全局 rank 的连续 `ep_size` 段；`ep_size == world` 时直接用 world 组（#2）。EP 与 DP/CP 是**正交切分**：EP 切的是专家维，DP 切的是请求维，CP 切的是序列维。
- 另有 engram 组 `enp`（在用，`deepinfer/model_executor/models/v4_modules/engram.py:79`）与 `ep_in_dp`（建了没用，#8）；每个 NCCL 组配一个 gloo 影子组（`cpu_dp_group` / `cpu_cp_group`），供 ZOS 下的 CPU 侧投票与统计使用（#21）。
- 见 #1–#7。

**Q2 MoE 层一次 forward 的 dispatch/combine 流程，DeepEP 在哪一步？**
1. `gate_forward`：本地全量 gate 权重算 top-k（`deepinfer/model_executor/models/v4_modules/mlp.py:574`，#9）。
2. **DeepEP dispatch**（`deepinfer/model_executor/models/v4_modules/mlp.py:578-585` → `468-492`）：`EPBuffer.dispatch(x, topk_idx, topk_weights, do_expand=True, defer_epilogue=True, expert_alignment=MK对齐)`。DeepEP 就在这里，做的是"按专家做 all-to-all 并把每个 (token,expert) 展开成一个槽位"；返回 `EventOverlap`，`current_stream_wait()` 才拿到 `moe_in_hidden` 与 `handle`（#10、#11）。
3. **两个 grouped GEMM**（`deepinfer/model_executor/models/v4_modules/mlp.py:593-598`、`406-459`）：用 `handle.psum_num_recv_tokens_per_expert` 作分组前缀和，contiguous 布局，gate_up → SwiGLU → down（#12）。
4. **共享专家并行**（`deepinfer/model_executor/models/v4_modules/mlp.py:600-603`）：`ep_buffer.capture()` 记事件后，在 `limit_num_sms(通信SM)` 限制下算 shared 分支（#13、#45）。
5. **DeepEP combine**（`deepinfer/model_executor/models/v4_modules/mlp.py:604-610` → `493-511`）：`combine(moe_out_hidden, handle, bias=shared_out)`，归约 routed 结果的同时把 shared 输出加进 epilogue；再 `current_stream_wait()`。
6. 若在 NVL 域内且精度组合允许，第 2–5 步被 **mega MoE** 单 kernel 取代（`deepinfer/model_executor/models/v4_modules/mlp.py:571-572`、`528-559`，DeepGEMM `SymmBuffer`），此时不再有 per-layer 的 DeepEP buffer 调用（#14）。
- 统计旁路：两条路径都把 `expert_count` 传给通信 kernel，EPLB 因此零成本采样（#15）。

**Q3 Attention DP 下各 DP rank 的 batch 不一样大，怎么同步？**
- **不同步 batch 大小，只同步三件事：**
  1. **步型**：DP 组内 `MIN` all_reduce 投票决定"整组做 prefill 还是 decode"；分 ALL 点（全员有活才 prefill）与 ANY 点（任一有活就 prefill），并用第二轮投票避免空 prompt 步活锁（#16、#17）。
  2. **参与集合通信**：没活的 rank 跑 1-token placeholder 步，保证每步每 rank 都进入相同次数的 EP dispatch/combine（#18）。
  3. **退出条件**：`MAX` all_reduce 保证所有 rank 同时退出（#21）。
- **允许不同的部分**：每 rank 自己调度、自己定 batch；decode 各选各的 cuda graph 桶并用 padding seq 补齐（#19）；DeepEP dispatch 是变长 all-to-all，token 数由 handle 携带，缓冲区按 `num_max_tokens_per_rank` 上界预分配（#10）。
- 启动期还有一次数值对齐：KV block 数按 world 取平均（#20）；prefill 的时间预算在 DP 组内取 MIN（#22，文档口径）。
- 见 #16–#22。

**Q4 balanced CP 是什么？"balanced" 平衡的是什么？`prepare_cp_mirror` 做什么？**
- 代码里**没有** "balanced CP" 这个词，对应实现叫 **mirror CP**（`prepare_for_mirror_cp` / `mirror_cp_index` / `CPManager`）（#23）。
- 做法：每个序列切成 `2*cp` 块，rank r 取第 r 块与第 `2cp-1-r` 块（zigzag/mirror）（#23）。
- "balanced" 平衡的是 **causal attention 的计算量**：配对块的位置和恒为 `2cp-1`，因此每 rank 的 `Σ(可见 key 数)` 与 rank 无关（#24，E 级推导）。顺带每 rank 的 token 数也相等（`2*ceil(sq/2cp)`）。
- `prepare_cp_mirror`（`deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py:11-76`，kernel 在 `cuda/prepare_cp_mirror.py`）输出四个张量：`mirror_cp_index`（本 rank 的 token 在原始 varlen 中的下标，-1 为 padding）、`mirror_cp_reverse_index`（逆映射，长度 `total_q`）、以及每个压缩比下的 `ks/ke`（每个 query 的 key 区间）（#23、#28）。它在 prefill 开始时调一次（`deepinfer/model_executor/input_metadata/input_metadata.py:169-208`，`assert is_prompt`），把 `positions`、`ngram_token_ids`、`gate_force_random` 等按索引重排，并生成 `valid_token_mask`（#25、#26）。
- 边界：只在 prefill 生效；CP 下 Q 侧切分但 KV 压缩/投影对全序列各算一遍，KV 显存不随 cp 减少（#28）；所有 CP all_gather 走 DeepEP `BucketBuffer` 的延迟 getter，返回值是 buffer 别名，必须当步消费（#27）；生产文档明确说 CP 目前不用（#30）。

**Q5 EPLB 策略算法、权重搬运（是否用 symm_mem）、统计窗口与重分配周期参数在哪配置？**
- 算法（`policy.py`，两层贪心，逐层独立）：
  1. `rebalance_experts_global`（`deepinfer/model_executor/eplb/policy.py:73-140`）：初始布局 = 前 `num_logical` 个槽位放逻辑专家、其余槽位放冗余副本（`102-103`）；然后循环 `num_redundant` 次，每次选"加一个副本后全局最大每副本负载最小"的逻辑专家（`109-123`），维护 `physical_to_logical_map` / `logical_to_physical_map` / `logical_count` 三张表。
  2. `_batch_greedy_indices`（`deepinfer/model_executor/eplb/policy.py:12-70`）：把物理槽位按"每副本历史负载降序"排序后逐个装箱到 rank，目标 `min Σ_step max_rank(load)`，跳过已满 rank、同分取最小 rank（#32、#33）。
- 权重搬运（`shuffle.py`，**是，用了 symm_mem**）：`ExpertLoadBalancer._initialize_shuffle`（`deepinfer/model_executor/eplb/__init__.py:195-220`）在 `PG.is_within_nvl_domain(ep_group)` 且 GPU 平台时，用 `torch.distributed._symmetric_memory.empty/rendezvous` 建**一层权重**的镜像，`deepinfer/model_executor/eplb/shuffle.py:150-218` 直接从对端 IPC 缓冲 `copy_`，无 P2P；跨域则用 `expert_model_parallel_shuffle_inplace`（`deepinfer/model_executor/eplb/shuffle.py:65-147`）的 `dist.P2POp` + `batch_isend_irecv`，按 rank 差分桶（#38、#39）。
- 参数配置位置（**全部是环境变量，不是模型 config**）：
  - 统计窗口：`DEEPINFER_EXPERT_COUNT_WINDOW_SIZE`（默认 100）（`deepinfer/environ.py:115`）
  - 重分配周期：`DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`（默认 0 = 关闭）（`deepinfer/environ.py:116`）
  - 冗余专家下限：`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`（`deepinfer/environ.py:114`），实际值按 #31 向上取整
  - prefill 后立即重排：`DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE`（`deepinfer/environ.py:117`、`deepinfer/model_executor/eplb/__init__.py:83-84`）
  - 负载统计 flush 间隔：`DEEPINFER_MOE_LB_STAT_INTERVAL`（默认 128 步）（`deepinfer/environ.py:118`）
  - P2P 分桶数：`DEEPINFER_REARRANGE_P2P_GROUPS`（默认 1）（`deepinfer/environ.py:111`、`deepinfer/model_executor/eplb/shuffle.py:113`）
  - 约束：`0 < window <= interval`，否则构造时报错（`deepinfer/model_executor/eplb/__init__.py:106-111`）；首次重排提前 1/4 周期（`121-123`）。
- 生产值（文档，B 级）：prefill 4 个冗余/卡、窗口 100、周期 1000、`moe_lb` 1.074/1.045；decode 1 个冗余/卡、窗口 1000、周期 10000、`moe_lb` 1.090（#40）。examples 里是 32 个冗余、窗口 3、周期 30（`examples/_config.py:79-83`），属示例值（#34）。
- 指标口径：`moe_lb = Σmax/Σmean`（`deepinfer/engine/stats_logger.py:238-241`），另有反向的 `rank_balance`（#41）。

**Q6 userbuffer 是什么，用于什么通信？**
- 是 `UserBufferManager` 单例（`deepinfer/model_executor/parallel_utils/userbuffer.py:54`），统一管理**四类预分配通信缓冲**：`EPBuffer`（DeepEP dispatch/combine，`allocate_ep:63-95`）、`CPManager`/`BucketBuffer`（CP all_gather，`allocate_cp:97-117`）、Engram buffer（`allocate_engram:127-181`）、`deep_gemm.SymmBuffer`（mega MoE，`allocate_sym_buffer:183-225`）（#44）。
- 它解决的问题是"通信库需要提前占显存、且必须在 cuda graph capture 之前分配"，尺寸按 `max_num_prefilling_tokens` / `max_num_decoding_tokens` 上界算（`deepinfer/model_executor/model_loader.py:119-120`；`deepinfer/model_executor/models/deepseek.py:412-419` 里 `max_ag_bytes_per_token = hidden_size*2`）。
- 附带两个能力：`cp_session()`（CP 的 session 生命周期与别名语义，#27）和 `limit_num_sms()`（按 SM 数量给通信让路，#45）。
- 见 #27、#44、#45、#46。

**Q7 有没有张量并行（TP）代码？为什么生产不使用 TP？**
- **没有 TP**：`ParallelConfig` 无 `tensor_parallel_size`，`model_parallel_size = context_parallel_size`（#4）；前向无任何 TP 型 all-reduce，线性层是 `ReplicatedLinear`（全量复制）与 `ParallelRoutedLinear`（按专家切）（#5）；`VocabParallelEmbedding` 只是沿用了 vLLM 的类名，实为整表 gather（#5 的疑点）。
- 官方文档："TP and CP are not used for now"（#30，B 级）。
- **"为什么"代码里没有写理由**，只能给出与代码一致的推断（E）：①MoE 已经把参数分片做在 EP 上（#47），attention/共享专家的权重在 EP 维度下已经"每卡全量"，再叠 TP 只会增加每层的 all-reduce；②decode 是带宽受限（#47 的推理），TP 的 all-reduce 增加的是延迟敏感的同步，而 EP 的 all-to-all 可以与计算重叠（#11、#13）；③CP 已经承担了"序列维切分"的角色且只用于长 prefill（#25）。
- 未能确认：DeepSeek 官方是否在其它场合解释过"不用 TP"的原因；本次只查了本仓库与 `doc/`。

## 4. 新发现的问题
1. **EPLB 的 `moe_lb` 有两套口径**（#41）。若书里要引用 1.074/1.090 这类数字，必须同时写明"哪套指标 + 窗口长度 + EP/DP 配置 + 负载"，否则无法与知乎/其它来源对齐。建议主编统一采用 `stats_logger` 的口径并在脚注说明。
2. **`n_routed_experts` 到底是多少？** 文档"EP64 每卡 7 个专家"反推 384（#31 疑点），但仓库里没有 HF config，无法确认。这直接影响"每卡几个专家/多少显存"的所有推算，建议向 R02 系列笔记核实 V4.1 的 MoE 配置。
3. **`prefill_attempt_interval` / `prefill_any_attempt_interval` / `compute_quota_limit` 的默认值**没在本次核对（#16、#22 疑点），它们决定 DP 组"多久投一次票"，是调度章节（R05）的输入。
4. **mega MoE 的关闭条件**是 `group.size() <= 物理 NVL 域大小`（#14）。EP128 必然跨域 → mega MoE 关闭，此时 Wide-EP 的性能特征会与 EP64 不同。文档把 EP64–128 并列推荐，但没区分这一点，书里应补一句。
5. **CP 在 decode 下是否纯浪费**（#25 疑点）：如果 cp 个 rank 各存一份完整 KV cache 且只由 driver 出 token，则 CP 的 decode 成本是 cp 倍显存。这一点对"为什么生产不用 CP"（#30）是更强的解释，值得在 attention/KV 章节（R02a）交叉确认。
6. **`ep_in_dp` 组是死代码**（#8）。若后续版本恢复"EP < DP"，这里的语义（每个 DP 组内部的 EP 子组）会成为关键，值得在版本对比时留意。
7. **placeholder 步是否计入 EPLB 窗口**（#36 疑点）：placeholder 步会写一行零进窗口，等于用"零负载步"稀释统计。周期 1000/窗口 1000 时影响可能可忽略，但 prefill 密集场景值得验证。

## 5. 可用于图表的材料

### 5.1 MoE 一层 forward 的 6 段时间线（可直接画时序图）
| # | 动作 | 代码位置 | 通信 | 备注 |
|---|---|---|---|---|
| 1 | gate：sqrtsoftplus + noaux_tc top-k | `deepinfer/model_executor/models/v4_modules/mlp.py:574` | 无 | gate 权重全量复制 |
| 2 | DeepEP dispatch（expand，defer epilogue） | `deepinfer/model_executor/models/v4_modules/mlp.py:578-585` | EP all-to-all | 返回 EventOverlap |
| 3 | wait → `moe_in_hidden` + `handle` | `deepinfer/model_executor/models/v4_modules/mlp.py:585` | — | 此处才真正收 token |
| 4 | grouped GEMM ×2（gate_up → SwiGLU → down） | `deepinfer/model_executor/models/v4_modules/mlp.py:593-598` | 无 | 用 `psum_num_recv_tokens_per_expert` 分组 |
| 5 | shared expert（受限 SM，另一 stream） | `deepinfer/model_executor/models/v4_modules/mlp.py:600-603` | 无 | `limit_num_sms(通信SM)` |
| 6 | DeepEP combine（`bias=shared_out`） | `deepinfer/model_executor/models/v4_modules/mlp.py:604-610` | EP all-to-all | shared 折进 epilogue |

### 5.2 DP 组内一步的决策流程（可直接画流程图）
```
decode_step++
├─ is_any_point = (decode_step % prefill_any_interval == 0)   # 任一 rank 有活即可
├─ is_all_point = (decode_step % prefill_interval == 0)       # 全员有活才可
├─ 都不满足 → decode
└─ 满足 → want = (KV 使用率 ≤ 水位) and waiting 非空
     ├─ ANY 点：MIN(want?0:1) == 0 → prefill
     ├─ ALL 点：MIN(want?1:0) == 1 → prefill
     ├─ 否 → decode
     └─ 是 → _schedule_prefill
          ├─ 全员空手（any_scheduled == false）→ 强制 decode（防活锁）
          └─ 有 rank 排到：本 rank 若无活 → 发 1-token placeholder 步
```
（依据 `csrc/engine/scheduler.hpp:295-341`）

### 5.3 并行轴对照表
| 轴 | 切什么 | 组构成 | 通信原语 | 生产是否用 |
|---|---|---|---|---|
| DP | 请求（batch） | 固定 (replica, cp)，取遍 dp | 仅投票/统计（MIN/MAX all_reduce）、KV block 数对齐 | 是，DP=EP |
| EP | 专家（MoE 权重） | 连续 ep_size 段，常等于 world | DeepEP dispatch/combine 或 mega MoE | 是，prefill 2–8 / decode 64–128 |
| CP | 序列（Q 侧） | 连续 cp 个 rank，同机 | BucketBuffer all_gather + 索引重排 | 否（代码具备，长 prefill 可用） |
| TP | — | — | 不存在 | 否（无代码） |

### 5.4 EPLB 参数对照表（示例值 vs 文档生产值）
| 参数 | env 名 | 默认 | examples | 文档 prefill | 文档 decode |
|---|---|---|---|---|---|
| 统计窗口 | `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 100 | 3 | 100 | 1000 |
| 重排周期 | `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | 0（关） | 30 | 1000 | 10000 |
| 冗余专家下限 | `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` | 未设 | 32 | 4/卡 | 1/卡 |
| prefill 后立即重排 | `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE` | false | 1 | 未提 | 未提 |
| LB 统计 flush | `DEEPINFER_MOE_LB_STAT_INTERVAL` | 128 | 未设（=默认） | 未提 | 未提 |
| 实测 moe_lb | — | — | — | 1.074 (EP4DP4) / 1.045 (EP2DP2) | 1.090 |

### 5.5 mirror CP 的块分配示意（cp=4，单序列 8 块）
```
块号:      0    1    2    3  |  4    5    6    7
rank:      0    1    2    3  |  3    2    1    0
           └── 前半 ────────┘  └── 后半（镜像）──┘
rank r 拿 {r, 2cp-1-r} → 每 rank 的 Σ(可见 key 数) 相等
```
（依据 `deepinfer/model_executor/layers/tilelang_ops/prepare_cp_mirror.py:146-152`；`ps = ceil(sq/(2cp))`）

### 5.6 数据点清单（带条件）
- Wide-EP prefill vs decode 的 EP 建议：EP2–8 / EP64–128（B，`doc/inference_deployment_qa.md:47-49`）
- `moe_lb`：1.074（EP4DP4 prefill）/ 1.045（EP2DP2 prefill）/ 1.090（decode EP64），条件=文档口径，硬件与负载未给（B，`doc/inference_deployment_qa.md:99,103`）
- EP8 vs EP64 decode 吞吐对比图：`doc/images/wide_ep_decode_pareto.png`（B，未读数值）
- 重排耗时：代码只打日志（`[EPLB] rearranged: ... time=%.3fs`，`deepinfer/model_executor/eplb/__init__.py:312-325`），**仓库内无实测数字**（需自行测）
