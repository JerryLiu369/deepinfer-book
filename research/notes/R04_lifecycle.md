# R04 请求生命周期：构造、进程模型、SHM 广播、CUDA Graph

## 0. 元信息
- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（本地 `research/sources/DeepInfer`）
- 查阅的主要来源（全部 A 级一手源码）：
  - `examples/offline_inference.py`、`examples/_launch.py`、`examples/_config.py`
  - `deepinfer/entrypoints/llm.py`、`deepinfer/engine/llm_engine.py`、`deepinfer/engine/arg_utils.py`
  - `deepinfer/worker/base.py`、`worker.py`、`model_runner.py`、`cuda_graph_runner.py`、`dflash_cuda_graph_runner.py`、`zos_status.py`
  - `deepinfer/model_executor/models/deepseek.py`、`layers/sampler.py`、`model_metadata_utils.py`
  - `deepinfer/model_executor/input_metadata/*.py`、`parallel_utils/parallel_state.py`、`parallel_utils/communication_op.py`
  - `deepinfer/distributed/shm_broadcast.py`、`deepinfer/shared_memory.py`、`deepinfer/utils.py`
  - `csrc/engine/{sequence.hpp, scheduler.hpp, block_manager.hpp, zos.cpp, prepare_sample.cpp, pybind_*.cpp, outputs.hpp}`
  - `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`
  - `deep_cluster/topology/bind.py`、`deep_cluster/hardware/{device,cpu,nvidia}.py`
- 另用 B 级：`doc/inference_deployment_qa.md`（异步调度一节，:266-274）

## 1. 三个最重要的发现
1. **DeepInfer 的"多卡"不是 TP**：一个 engine 进程只吃 `cp_size` 张卡（driver + `cp_size-1` 个 spawn 子进程），DP 副本之间靠一次 all-reduce 投票保证"这一步全组做 prefill 还是 decode"；World 里没有 tensor-parallel group，只有 DP/CP/EP/ENP。
2. **driver→worker 的 RPC 只广播方法名和布尔参数，不回传任何结果**：`MessageQueue`（无名 SHM 环 + ZMQ XPUB/SUB）传 `(method, kwargs)`；`seq_metadata_list` 留在 driver，CP follower 用 `broadcast_tensor_dict`（非 tensor 走同一条 SHM 队列，tensor 走 NCCL）自己重建 `InputMetadata`；采样结果只在 driver 产生。
3. **ZOS（异步调度）在代码里不是"线程"，而是"错一步的执行顺序 + 乐观账本"**：`execute_model_step` 固定按 Forward(N) → post-process(N-1) → Sample(N) 排列；`Sequence.cache.num_tokens` 在 schedule 时按 `seqlen_q` **乐观推进**，在 `process_outputs` 里用实际接受数**回填**，中间差额记在 `schedule.total_inflight_tokens`。

## 2. 事实条目

### #1 一个 engine 进程 = 1 个 driver Worker + (`cp_size`−1) 个 spawn 子进程，每 GPU 恰好一个 Worker 进程
- 证据：`deepinfer/engine/llm_engine.py:378-395`
  > `for cp_rank in range(1, cp_size):` … `cp_rank % ngpus,  # local_rank` … `SpawnProcessWithCrashHeader(target=_spawn_worker, name=f"worker-cp{cp_rank}", args=args, daemon=True)`
- 证据：`llm_engine.py:402-412` driver 是**当前进程内**构造的 `Worker(..., local_rank=0, cp_rank=0, is_driver_worker=True, engine_callbacks=...)`
- 证据：`deepinfer/worker/base.py:186-187` 每个 Worker 只 `torch.cuda.set_device(f"cuda:{local_rank}")`
- 证据：`deepinfer/config.py:171` `self.model_parallel_size = self.context_parallel_size`——"模型并行"就是 CP，没有独立的 TP 维度
- 等级：A
- 意味着什么：书的进程模型图可以画成"每卡一个进程"，但**只有 dp 副本的 0 号卡那个进程里有 Scheduler**，其余进程是纯执行器；`LLMEngine` 从不跨进程迁移。
- 疑点：`cp_rank % ngpus` 允许 `cp_size > ngpus` 时两个 cp_rank 落在同一张卡上（NVLink 内多进程共享 GPU），本仓库示例没有这种配置。

### #2 控制面 RPC 是 SHM 环 + ZMQ，载荷是 `(方法名, kwargs)`，没有返回值
- 证据：`deepinfer/worker/base.py:129-133`
  > `def start_slave_worker_loop(self):` … `while True:` `method, kwargs = cp_broadcast_object(None)` `getattr(self, method)(**kwargs)`
- 证据：`base.py:296-302` driver 侧 `run_workers()`：`cp_broadcast_object((method, kwargs))` 后 `return getattr(self, method)(**(driver_kwargs or {}), **kwargs)`
- 证据：`deepinfer/model_executor/parallel_utils/parallel_state.py:17-23` `cp_broadcast_object` 只有 `mq.enqueue` / `mq.dequeue` 两个分支
- 等级：A
- 意味着什么：这是一个**单向、fire-and-forget 的集体调用**，不是通用 RPC；worker 的返回值被丢弃，所以任何"结果回传"必须在 driver 侧产生。
- 疑点：无。

### #3 `seq_metadata_list` 不广播；follower 用 `broadcast_tensor_dict` 重建输入
- 证据：`llm_engine.py:1017-1025` + `961-975`：`_run_workers("execute_model_step", driver_kwargs=dict(seq_metadata_list=..., scheduler_outputs=...), is_prompt=..., is_placeholder=...)`——`driver_kwargs` 不进广播
- 证据：`deepinfer/worker/model_runner.py:200-212` driver 侧只广播 `{is_prompt, type, run_cuda_graph, total_q, num_reqs}` + `input_metadata.to_flatten_dict()`
- 证据：`model_runner.py:216-230` follower 侧 `input_metadata_dict = broadcast_tensor_dict(src=0)`；`input_metadata_type.from_flatten_dict(...)`
- 证据：`parallel_utils/communication_op.py:55-82` `broadcast_tensor_dict`：CUDA tensor 走 `cp_broadcast_multi_tensor`（NCCL），其余走 `cp_broadcast_object`（SHM 队列）
- 等级：A
- 意味着什么：CP>1 时 CPU 侧的 `Sequence`/`SequenceMetadata` 只存在于 driver；follower 是"无状态执行器"，它看到的只是张量化的 `InputMetadata`。
- 疑点：follower 因此无法自己判断 `is_prompt`（靠 driver 广播的 key），也无法计算 sampling。

### #4 采样只在 driver 发生；follower 的 Sampler 直接返回空 `SamplerOutput`
- 证据：`deepinfer/model_executor/layers/sampler.py:136-161`
  > `driver_inputs = sampler_inputs.driver` … `if driver_inputs is None:` … `return SamplerOutput(post_funcs=[])`
- 证据：`model_runner.py:348-365` `sampler_inputs = self.prepare_sampler_inputs(...)`，只在 driver 上填 `driver` 子结构
- 等级：A
- 意味着什么：`SamplerInputs.driver` 是否为 None 就是"我在不在 driver 上"的隐式标志；书里讲采样并行度时要说明采样是 DP 内串行、CP 内只做集合通信（`compute_prompt_logprobs` 注释 `sampler.py:147`）。
- 疑点：无。

### #5 一个"step"的固定顺序是 Forward(N) → post-process(N−1) → Sample(N) → 条件性 post-process(N)
- 证据：`deepinfer/worker/worker.py:96-104`
  > `1. Forward current step (main model)` / `2. Post-process previous step` / `3. Sample current step` / `4. Post-process current step: in place for prompt / non-ZOS steps; ZOS decode steps defer it to the next step to overlap with forward`
- 证据：`worker.py:121-124`（forward）、`143-144`（post-process prev）、`154-159`（sample）、`182-187`（`if is_prompt or not ZOS: 当场 post_process else prev_step_state = curr_step_state`）
- 等级：A
- 意味着什么：这是全书讲"一个解码步"的最好骨架：**采样完不立刻等结果**，等待被推迟到下一步 forward 之后，正好用 forward 的 GPU 时间去掩盖 D2H 延迟。
- 疑点：无。

### #6 `RequestOutput` 是 C++ 池对象，Python 拿到的是一串 `uint64` 指针
- 证据：`csrc/engine/outputs.hpp:306-307`
  > `py::gil_scoped_acquire acquire;` `return py::array_t<uint64_t>(request_outputs.size(), request_outputs.data());`
- 证据：`outputs.hpp:176` `static std::vector<uint64_t> request_outputs(RequestOutputPool::max_size);`（**static 借用**，无 capsule）
- 证据：`deepinfer/entrypoints/llm.py:179-182` `for raw_output in step_outputs:` `output = RequestOutput.clone(raw_output)`
- 证据：`deepinfer/_C/engine.pyi:180-183` `RequestOutput.clone(ptr)` 是 staticmethod，参数是整数指针
- 等级：A
- 意味着什么：输出对象的所有权在 C++ 池里，`step()` 的返回值只在本步内有效；`LLM` 必须 clone 才能留住——这解释了 `llm_engine.py:866-870` 的警告注释。
- 疑点：`build_request_outputs` 返回的 numpy array 没有 owner，若 Python 侧跨步持有会读到被复用的 static vector。

### #7 调度主体在 C++：`Scheduler::schedule` 一次调用返回 `(seq_metadata_list, SchedulerOutputs)`
- 证据：`csrc/engine/scheduler.hpp:667-736`
  > `py::tuple schedule(SequenceMetadataPool *sequence_metadata_pool) {` `auto p_scheduler_outputs = _schedule();` … `seq_metadata_list[i] = std::move(p_metadata);` … `return py::make_tuple(std::move(seq_metadata_list), std::move(p_scheduler_outputs));`
- 证据：`scheduler.hpp:688-707` 每个被调度的 seq 在同一个循环里 `sequence_metadata_pool->allocate(seq, prompt_run, seqlen_q, seqlen_k, golden_token)`
- 证据：`deepinfer/engine/llm_engine.py:885-887` Python 侧只有一行 `self.scheduler.schedule(self.sequence_metadata_pool)`
- 等级：A
- 意味着什么：`SequenceMetadata` 不是 Python dataclass，而是**调度器当场生成的快照**（同一循环保证 `scheduled_seqs` 与 `seq_metadata_list` 严格同序，见 `llm_engine.py:911` 注释）。
- 疑点：无。

### #8 NORMAL 引擎的步型由 DP 组投票决定，ZOS 下投票走 gloo/CPU
- 证据：`scheduler.hpp:295-341`（ALL 点/ANY 点两轮投票，`_dp_allreduce(want?1:0, MIN)`）
- 证据：`scheduler.hpp:170-181` `_dp_allreduce` 用 `use_cpu_dp_group ? torch::kCPU : get_default_device()`；`scheduler.hpp:109` `use_cpu_dp_group = dp_group->getBackendName() == "gloo"`
- 证据：`llm_engine.py:313` `PG.cpu_data_parallel_group() if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else PG.data_parallel_group()`
- 等级：A
- 意味着什么：DP 副本之间**必须同步**（否则一个 rank 做 prefill、其余做 decode 会让 EP all-to-all 死锁）；ZOS 之所以能把投票放到 CPU 上，是因为它不再需要等 GPU 结果。
- 疑点：投票是阻塞的 CPU 集合通信，`_schedule` 全程持 GIL 释放（`scheduler.hpp:276`），但仍是每步一个同步点。

### #9 ZOS 的账本是"乐观推进 + 事后回填"
- 证据：`csrc/engine/sequence.hpp:216-221`
  > `void advance_optimistically(int token_count) { previous_num_tokens = num_tokens; num_tokens += token_count; }` / `void commit_processed_tokens(int token_count) { num_tokens = previous_num_tokens + token_count; }`
- 证据：`scheduler.hpp:927`（schedule 内 `seq.cache.advance_optimistically(scheduler_outputs.seqlens_q[i]);`）
- 证据：`scheduler.hpp:999`（process_outputs 内 decode 分支 `seq.cache.commit_processed_tokens(token_cnt);`，注释"未被接受的 draft 不占 KV"）
- 证据：`sequence.hpp:244-255` `start_inflight_step/finish_inflight_step` 维护 `total_inflight_tokens`
- 等级：A
- 意味着什么：调度器在还不知道上一步接受了几个 draft token 的情况下就要为下一步算 block 需求，所以它先按"全接受"占位，等真实结果回来再结算；`get_num_blocks()`（`sequence.hpp:291-293`）里显式加了 `total_inflight_tokens + nextn`。
- 疑点：若某步实际接受数少于乐观值，`block_table` 里会短暂多占块，回收时机取决于 `commit_processed_tokens` 之后的 free 路径。

### #10 ZOS 下 CPU 侧 seqlens 滞后一步，靠 GPU 重算 + 回写 pinned buffer 修正
- 证据：`deepinfer/worker/worker.py:191-245` `_correct_seqlens_qk(...)`，docstring：`CPU 侧的 seqlens_q/k 在 ZOS 下滞后一步：与上一步 batch 有交集时，需按上一步采样结果在 GPU 上修正`
- 证据：`worker.py:147-152` `scheduler_outputs.update_metadata_for_sampling_zos(seq_metadata_list, correction.seqlens_q_cpu, correction.metadata_zos.remap_cpu, self.nextn)`
- 证据：`csrc/engine/zos.cpp:279-307` `update_metadata_for_sampling_zos(...)`
- 证据：`deepinfer/worker/zos_status.py:27-30` `remap_cpu` / `remap`：`remap[i] = j if curr_batch[i] matches prev_batch[j], else -1`
- 等级：A
- 意味着什么：这是 ZOS 最反直觉的代价——**KV 长度、query 长度在采样那一刻是"猜"的**，必须在 GPU 上用上一步真实接受数覆写；书里若只写"异步调度省掉了气泡"会漏掉这条。
- 疑点：`doc/inference_deployment_qa.md:264` 说该修正由一个 TileLang kernel 完成（cumulative product + top-k sort + fused remap scatter），代码里 `_correct_seqlens_qk` 具体 kernel 未逐行核对。

### #11 CUDA Graph 只覆盖 decode 主模型 forward；prefill 与 sampler 都走 eager
- 证据：`deepinfer/worker/model_runner.py:73` `self.use_cuda_graph = not envs.DEEPINFER_ENFORCE_EAGER and self.parallel_config.engine_type != EngineType.PROMPT`
- 证据：`model_runner.py:182` `run_cuda_graph = not is_prompt and self.use_cuda_graph and bool(self.cuda_graph_runner.graphs)`
- 证据：`model_runner.py:326-329` `if run_cuda_graph: model_output = self.cuda_graph_runner.replay(cuda_graph_used) else: model_output = self.model(...)`
- 证据：`model_runner.py:337-367` `execute_sample` 内无 graph 分支
- 等级：A
- 意味着什么：书上画"decode 一步"时，只有中间那段 transformer 是 replay，采样、MTP 验证、post-process 都是普通 kernel 发射（DFlash 例外，见 #12）。
- 疑点：无。

### #12 DFlash（MTP draft）有独立的 graph runner，与主图共用 memory pool / stream / 最大 batch 的 device metadata
- 证据：`model_runner.py:543-550` `self.dflash_cuda_graph_runner.capture(dflash_model, kv_caches, main_input_metadata=main_graph_runner.graphs[max(...)].input_metadata_device, graph_memory_pool=main_graph_runner.graph_memory_pool, graph_stream=main_graph_runner.graph_stream)`；随后 `dflash_model.decode_graph_runner = self.dflash_cuda_graph_runner`
- 证据：`deepinfer/model_executor/models/mtp/dflash.py:383-387` `decode_forward(...)`：`if self.decode_graph_runner is not None: return self.decode_graph_runner.replay(decode_input)`
- 证据：`deepinfer/worker/dflash_cuda_graph_runner.py:52-75` 按 `max_num_seqs` 捕获，`do_sample` 只影响是否拷 output
- 等级：A
- 意味着什么：投机解码的 draft 网络也被图捕获了，否则每个 decode 步会多出几十次 kernel launch；书里讲 MTP 性能时必须提到这条。
- 疑点：`do_sample=self.is_driver_worker`（`model_runner.py:87-90`）→ follower 的 DFlash 图不含 sampling，两条图的节点数不同。

### #13 图按 token 数选，不是按请求数；batch size 集合 = 2 的幂 + 固定步长
- 证据：`deepinfer/worker/cuda_graph_runner.py:41-51`
  > `while size < interval: batch_sizes.append(size); size *= 2` … `batch_sizes.extend(range(interval, max_size, interval))`
- 证据：`model_runner.py:76-79` `CudaGraphRunner(max_batch_size=self.scheduler_config.max_num_seqs * (1 + self.nextn), input_on_device=envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING)`
- 证据：`cuda_graph_runner.py:143-148` `select(total_q)` 用 `bisect_left` 向上取整
- 证据：`deepinfer/environ.py:155` `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL = EnvVar.int(default=8)`
- 等级：A
- 意味着什么：示例配置 `max_num_seqs=4, nextn=5` → 最大图 batch = 24 token，即"4 请求 × (1+5) 个投机位"；请求数不变但投机长度变化时，选到的图会变。
- 疑点：图尺寸是 token 数，而 padding 是按 `padding_seq_metadata_list[: graph_size - total_q]` 补**序列**（`model_runner.py:187`）——要求 pad 序列的 seqlen_q 恰为 1，`_placeholder_seq_metadata` 里 `seqlen_q=1` 满足。

### #14 输入元数据由 C++ 生成，且刻意"只算 offset/len，把大 index 展开推迟到 forward"
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:568-569`（`prepare_decode_v4` 先 `prepare_decode_base` 再 `update_common_v4`）
- 证据：`prepare_tensor_input_v4.cpp:608-612` 注释
  > `这里只计算出 offset 和 len, 避免过大的 CPU 开销。后续 CUDA 计算方式：pos = offset[i_q] + arange(len[i_q])`
- 证据：`deepinfer/model_executor/input_metadata/input_metadata.py:241-244` `kv_ca_index`: `computed in forward from kv_ca_index_offset_len`
- 证据：`model_metadata_utils.py:143` `prepare_decode = prepare_decode_v4_zos if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else prepare_decode_v4`
- 等级：A
- 意味着什么：`InputMetadata` 里的 `*_offset_len` 字段是"CPU 只写 2 个 int、GPU 展开成 [total_q, max_model_len/N] 的 index"，这是 CPU 侧 prepare 成本被压下去的关键设计。
- 疑点：无。

### #15 C++ 化的目的有三条硬证据：放 GIL、对象池、复用静态缓冲
- 证据：`scheduler.hpp:276` `py::gil_scoped_release release; // otherwise other python threads would be blocked`
- 证据：`csrc/engine/outputs.hpp:104` `py::gil_scoped_release release;`（build_request_outputs 全程无 GIL，`:306` 才 `gil_scoped_acquire` 造 numpy）
- 证据：`csrc/engine/prepare_sample.cpp:42`、`pybind_scheduler.cpp:67-71`、`pybind_sequence.cpp:52-55` 均 `gil_scoped_release`
- 证据：`sequence.hpp:325` `struct SequenceMetadata : public deepinfer::EnableObjectPool`；`llm_engine.py:271` `SequenceMetadataPool()`
- 等级：A
- 意味着什么：C++ 化不是为了"跑得快"，而是为了**在 GPU 还在跑上一步时，CPU 能在不持有 GIL 的情况下改调度账本**；对象池 + static vector 是为了消除每步的分配与 Python 对象开销。
- 疑点：无。

### #16 SHM 广播的参数是 1 MiB × 10 块，超过 1 MiB 的载荷退回 ZMQ
- 证据：`parallel_state.py:122-124` `MessageQueue.create_from_process_group(self.cpu_cp_group, 1 << 20, 10)`
- 证据：`deepinfer/distributed/shm_broadcast.py:396-404`
  > `overflow = len(serialized_obj) >= self.buffer.max_chunk_bytes` … `buf[1 : len(serialized_obj) + 1] = serialized_obj` … `if overflow: self.local_socket.send(serialized_obj)`
- 证据：`shm_broadcast.py:442-444` 注释 `assume the group does not cross nodes, so every reader is a local reader`
- 等级：A
- 意味着什么：CP 组被假定单机（也确实由 `CUDA_VISIBLE_DEVICES` 切片保证），所以可以纯 SHM；跨机路径（`n_remote_reader>0`）代码存在但本仓库的 CP 用法不会走到。
- 疑点：`dequeue` 的 overflow 分支依赖 `buf[0]==1` 标志与 ZMQ 消息**一一对应**；若同一 reader 并发 dequeue 会错配（当前只有单线程 slave loop，安全）。

### #17 `deepinfer/shared_memory.py` 与广播通道无关，是多模态数据池
- 证据：`deepinfer/shared_memory.py:80-88` `class SharedTorchMemoryPool` docstring：`多进程共享内存池`，只支持 CPU Tensor
- 证据：`shared_memory.py:113-119` `self._buffer_shm = shared_memory.SharedMemory(create=True, size=self.total_bytes)` + `PLATFORM.pin_host_memory(self.buffer)`
- 证据：唯一使用者 `deepinfer/ve/multimodal_processor.py:22`、`deepinfer/ve/schema.py:7`（`get_multi_modal_pool`）
- 等级：A
- 意味着什么：读代码时容易把两个"shared memory"混为一谈：一个是**控制面**（pickle 消息环），一个是**数据面**（pinned 图像/视频块池，用 `BlockHandle` 分配）。
- 疑点：无。

### #18 拓扑绑定只做 GPU→NUMA→CPU，IB/NIC 的绑定函数没有调用者
- 证据：`deep_cluster/topology/bind.py:42-61` `which_numa(i_gpu=...)` → `physical_device_id` → `device_numa_node`
- 证据：`bind.py:79-93` `bind_numa(node)`：`numa_sched_setaffinity(pid, cpumask)` + `set_mempolicy(MPOL_BIND, ...)`
- 证据：`deep_cluster/hardware/nvidia.py:55-67` NUMA 读自 `/sys/bus/pci/devices/<bus_id>/numa_node`
- 证据：`deep_cluster/topology/bind.py:65` `numa_path = _root / f"sys/class/infiniband/mlx5_{i_ib}/device/numa_node"`——全仓库 grep 无 `which_numa(i_ib=` 调用
- 证据：`deepinfer/utils.py:143-151` `bind_cpu(local_rank)`：`num_cpus_per_gpu = phy_cpus // ngpus`，再 `set_cpu_affinity`
- 等级：A（"IB 分支无人调用"是 A 级 grep 结论；"所以 NIC 未绑定"是 E 级推断）
- 意味着什么：书里讲"CPU/NIC/GPU 拓扑绑定"时必须限定：**代码只做 NUMA 内存策略 + CPU 亲和 + 可见 GPU 切片**，NIC 侧只有遗留的 helper。
- 疑点：`bind_cpu(self.gpu_id)`（`base.py:191`）用**物理 GPU id** 当 `local_rank` 去切片 CPU（`utils.py:147`），而 `_bind_numa(self.local_rank)`（`base.py:185`）用的是**可见序号**；`DEVICE_OFFSET != 0` 时两者会指向不同 CPU 段（E）。

### #19 示例脚本的启动：`launch_dp` spawn 出 `dp_size` 个进程，每个只看 `cp_size` 张卡
- 证据：`examples/_launch.py:26-49` `ctx = get_context("spawn")` … `procs.append(ctx.Process(target=main_fn, args=(dp_size, dp_rank), daemon=False))`
- 证据：`examples/_config.py:55-63` `setup_cuda_visible_devices(engine_rank, cp_size, device_count)`：`CUDA_VISIBLE_DEVICES = str(offset + i) for i in range(local_engine_rank*cp_size, (local_engine_rank+1)*cp_size)`
- 证据：`examples/offline_inference.py:27-29` `dp_size = max(ep_size // cp_size, 1)`；`world_size = dp_size * cp_size`；`assert world_size % ep_size == 0`
- 等级：A
- 意味着什么：默认 `--ep 8 --cp 1` → 8 个进程、每个 1 张卡、EP 组 = world；`--cp 2` → 4 个 engine 进程 ×2 卡。
- 疑点：无。

### #20 初始化必须经过 `profile_run` → 全 world 取最小块数 → 捕获图；图捕获在 `warm_up_model` 里
- 证据：`llm_engine.py:416-440` `_init_cache`：`_run_workers("profile_num_available_blocks")` → `init_cache_engine` → `warm_up_model`
- 证据：`base.py:264-267` `dist.all_reduce(num_gpu_blocks, SUM, group=PG.world_group())`；`num_gpu_blocks = ... // world_size // 2 * 2`
- 证据：`base.py:274-288` `warm_up_model`：`capture_model` →（MTP varlen 时）`profile_decode_perf` → `capture_dflash_model`
- 证据：`scheduler.hpp:82-85` `Scheduler` 构造参数含 `flops_estimator`、`lookahead_tokens`、`cp_size`、`log_stats`
- 等级：A
- 意味着什么：`num_gpu_blocks` 是所有 rank 的最小值再取偶，因此 DP 副本间的 KV 容量严格一致（这是 DP 投票能对齐的前提）。
- 疑点：`// 2 * 2` 的取偶原因（对齐 block 对？）未在注释里说明。

### #21 `SamplingParams` 是纯 Python 对象，但被 C++ 在构造时"抓取"成 `SamplingScalars`
- 证据：`deepinfer/sampling_params.py:47-83`（temperature/top_p/top_k/ignore_eos/max_tokens/seq_seed/so_args/golden_token_ids）
- 证据：`sampling_params.py:76-79` `temperature < SAMPLING_EPS` 时强制 `top_p=1, top_k=-1`（贪心）
- 证据：`csrc/engine/sequence.hpp:92-116` `struct SamplingScalars { ... static SamplingScalars capture(const py::object &sampling_params) }`
- 证据：`sequence.hpp:283-285` `Sequence` 构造时 `generation.sampling = SamplingScalars::capture(generation.params);`
- 等级：A
- 意味着什么：`Sequence` 保留 `py::object params`（给 Python 看）和一份 C++ 标量快照（给 sampler 用），两者可能在 Python 侧改 `params` 后不同步。
- 疑点：`add_request` 里 `sampling_params.eos_id = self.eos_id`（`llm_engine.py:527`）发生在 `Sequence(...)` 之前，所以快照是最新的；但 `_placeholder_seq_metadata` 另有独立 `SamplingParams(temperature=0.0)`。

## 3. 必答问题逐条回答

### Q1 从 `LLM(...)` 构造到 `generate()` 返回，按顺序每一步

| # | 文件:行 | 函数 | 做了什么 |
|---|---|---|---|
| 1 | `examples/offline_inference.py:34,113` | `main` / `launch_dp(main, dp_size)` | 解析参数，按 `dp_size` spawn 子进程；`:45-57` 调 `create_llm(...)` |
| 2 | `examples/_config.py:86-124` | `create_llm` | `os.environ.update(env)` → `resolve_model_path` → `setup_cuda_visible_devices(engine_rank, cp_size, ...)` → 才 `import torch` / `from deepinfer.entrypoints.llm import LLM`（注释：CUPIT 会提前初始化 CUDA） |
| 3 | `deepinfer/entrypoints/llm.py:54-85` | `LLM.__init__` | `kwargs.setdefault("disable_log_stats", True)`、`setdefault("stop_preempted_request", False)`（:67-68）→ `EngineArgs(...)`（:70-81）→ `LLMEngine.from_engine_args(engine_args)`（:84） |
| 4 | `deepinfer/engine/arg_utils.py:62-114` | `EngineArgs.create_engine_configs` | 造 `ModelConfig`/`CacheConfig`/`ParallelConfig`/`SchedulerConfig`；block size 由 `ModelMetadata.resolved_tokens_per_block` 推出（:83-86），engram 模型额外设 `use_compressed_tokenizer`（:89-91） |
| 5 | `llm_engine.py:461-469` | `LLMEngine.from_engine_args` | `cls(*engine_configs, log_stats=not disable_log_stats, engine_type=..., offline_inference=...)` |
| 6 | `llm_engine.py:155-290` | `LLMEngine.__init__` | 见下方展开 |
| 7 | `llm_engine.py:368-412` | `_init_workers` | `cp_rank=1..cp_size-1` 各 spawn 一个 `Worker` 子进程；driver `Worker` 在本进程内构造（`local_rank=0, cp_rank=0, is_driver_worker=True`）并注入 `EngineCallbacks(post_process, pre_sampler_sync)`（:398-400） |
| 8 | `worker/base.py:58-123` | `WorkerBase.__init__` | 造 `ModelRunner`（:90-96）→ `_init_torch`（:108）→ `init_model`（:114）→ `init_dist`（:117）；非 driver 进程在 :120-123 进 `start_slave_worker_loop()` 并**永不返回** |
| 9 | `base.py:182-194` | `init_model` | `configure_worker_environment` → `_bind_numa(local_rank)` → `torch.cuda.set_device` → `configure_worker_kernels` → `bind_cpu(gpu_id)` → `set_random_seed_with_rank` |
| 10 | `base.py:315-344` | `_init_distributed_environment` | 默认 PG 用 **gloo**（:334-340），`rank = engine_rank*cp_size + cp_rank`；`init_parallel_groups(dp, ep, cp, enp)` 建 NCCL/HCCL 子组（`parallel_state.py:86-126`） |
| 11 | `llm_engine.py:241` | `_run_workers("load_model")` | 广播方法名，所有 rank 各自加载分片权重（`Worker.load_model` → `ModelRunner.load_model`，`base.py:203-209`） |
| 12 | `llm_engine.py:416-440` | `_init_cache` | `profile_num_available_blocks` → 校验 `runtime_max_model_len` → `init_cache_engine` → `warm_up_model`（capture CUDA graph） |
| 13 | `llm_engine.py:265-274` | — | `BlockManager(num_gpu_blocks)`、`state_manager = BlockManager(num_state_seqs)`、`_create_scheduler()`、`scheduler.decode_perf_curve = driver_worker.decode_perf_curve`、`SequenceMetadataPool`、`DeferredRelease` |
| 14 | `llm.py:90-142` | `LLM.generate` | 逐个 `_add_request`（:144-166：`copy.deepcopy(sampling_params)`、`request_id = str(next(counter))`、`tokenizer.encode` → `np.int32`）→ `_run_engine(use_tqdm)` |
| 15 | `llm_engine.py:472-628` | `add_request` | 校验 token 范围（:581-590）→ `Sequence(...)`（C++ 构造，`:594-599`）→ 可选 SO 编译 → `scheduler.add_sequence(seq)`（:620-628） |
| 16 | `llm.py:168-195` | `_run_engine` | `while True:` `has_unfinished_requests()` → `llm_engine.step()` 或 `flush_inflight_reqs()`（:177-178）→ 对每个 uint64 指针 `RequestOutput.clone`（:181）→ 按 `request_id` 归并（:187-188） |
| 17 | `llm_engine.py:860-959` | `step` | `schedule`(:881-887) → clone/zero blocks(:889-905) → `post_schedule`(:907) → kvcache 落盘(:930-937) → 多模态 fetch(:939-945) → `_execute_model_async_step`(:952) |
| 18 | `llm_engine.py:961-975` + `1017-1025` | `_execute_model_async_step` / `_run_workers` | 广播 `execute_model_step(is_prompt, is_placeholder)`，driver 侧额外传 `seq_metadata_list`/`scheduler_outputs` |
| 19 | `worker/worker.py:89-189` | `Worker.execute_model_step` | Forward(N) → post-process(N−1) → Sample(N) → 条件性 post-process(N)（详见 Q3） |
| 20 | `worker.py:248-272` → `llm_engine.py:984-1015` → `llm_engine.py:687-798` | `_post_process_step` / `post_process` / `_process_model_outputs` | `sample_end_evt.synchronize()` → `model_runner.post_process`（跑 `post_funcs`）→ 回调 driver 的 `post_process` → `process_MTPInfo` + `scheduler.process_outputs` + `build_request_outputs` |
| 21 | `deepinfer/entrypoints/llm.py:196-198`（调用点 `examples/offline_inference.py:109`） | `LLM.close` | `_run_workers("stop")`（`base.py:135-151`：barrier → `userbuffer.close` → `destroy_cuda_graphs` → `cleanup_shm_broadcast` → `destroy_process_group`，非 driver `os._exit(0)`） |

`LLMEngine.__init__` 展开（`llm_engine.py:188-290`）：
1. `driver_gpu_id = int(CUDA_VISIBLE_DEVICES.split(",")[0])`（:188-189）
2. `get_hf_tokenizer`（:200-205）
3. `_get_model_architecture` 判多模态 → `_bind_numa(0)` + `MultimodalProcessor`（:209-221）
4. `RequestOutputPool()`（:230）
5. `_init_workers()`（:233）
6. `_run_workers("load_model")`（:241）
7. `_init_cache()`（:244）
8. `Scheduler` + 两个 `BlockManager` + `SequenceMetadataPool` + `DeferredRelease`（:265-274）
9. `StatsLogger`（:298-…）

### Q2 有几个进程？每 GPU 一个 worker 吗？driver 和 worker 怎么通信？

**进程数**（A）：`world_size = dp_size * cp_size`（`examples/offline_inference.py:28`）。GPU 进程数恰为 `world_size`：每个 engine 是 1 个 driver 进程 + `cp_size-1` 个 spawn 子进程（`llm_engine.py:378-395`），每个进程 `torch.cuda.set_device(local_rank)` 且 `local_rank = cp_rank % ngpus`、`CUDA_VISIBLE_DEVICES` 只含 `cp_size` 张卡。另有：`dp_size>1` 时 `launch_dp` 的父进程（`_launch.py:36-48`，只 join 不干活）、可选的多模态预处理子进程（`ve/multimodal_processor.py:94`）、SO 编译线程池（`deepinfer/structure_output/so_matcher_builder.py:27-43`，非进程）。

**每 GPU 一个 worker**（A）：是。但"worker"不等于"engine"：只有 `cp_rank==0` 的进程有 `Scheduler`/`BlockManager`/`RequestOutputPool`/tokenizer；其余进程构造完 `Worker` 后直接进 `start_slave_worker_loop`，连 `LLMEngine` 都不存在（`llm_engine.py:235` 注释："slave worker 不构造 LLMEngine，不会重复占端口"）。

**通信**（A）：
1. **控制面**：`PG.mq` = `MessageQueue`（`parallel_state.py:124`，`1<<20` 字节 × 10 块）。writer（cp_rank 0）`enqueue((method, kwargs))`，reader 从无名 SHM 环读，溢出（≥1 MiB）走 ZMQ XPUB/SUB；句柄由 `dist.broadcast_object_list` 在 gloo PG 上交换（`shm_broadcast.py:446-460`）。**无返回值**。
2. **数据面（CP>1）**：`broadcast_tensor_dict`（`communication_op.py:51-82`）：CUDA tensor → `cp_broadcast_multi_tensor`（NCCL）；非 tensor（`TensorMetadata`、`type`、`is_prompt`…）→ 同一条 SHM 队列。`InputMetadata.broadcast()`（`input_metadata.py:120-…`）在 CUDA graph 捕获/回放时用同一机制**原地**写 follower 的预分配张量。
3. **模型内集合通信**：EP all-to-all 走 deep_ep，CP/DP 走 NCCL 子组（`parallel_state.py:86-126`），默认 PG 是 gloo（`base.py:334`），NCCL 组在 `init_device_groups` 里各做一次 `all_reduce` 强制建链（`parallel_state.py:173-187`，注释解释"单 rank 组也不能跳过，否则 engram 的 deep_ep 拿空指针段错误"）。
4. **DP 级同步**：`_dp_allreduce`（`scheduler.hpp:170-181`）做 prefill/decode 投票；`has_unfinished_requests`（`llm_engine.py:665-678`）每步一次 `MAX` all-reduce；ZOS 时两者都用 `cpu_data_parallel_group`（gloo）。

### Q3 一个解码步（step）里，调度 / 准备输入 / forward / 采样 / 后处理分别在哪里

| 阶段 | 位置 | 做了什么 |
|---|---|---|
| 调度 | `llm_engine.py:885` → `csrc/engine/scheduler.hpp:667` `schedule()` → `:275` `_schedule()` → `:559` `_schedule_decode()` | 持 `BlockManagerMutexAcquire`（:277）；`free_finished_seqs` → 清 `waiting` 中已终结的 → 从 `prompted` 队列 admit（:582-600）→ 排序、`_mtp_varlen.schedule`（:648-652）→ 逐 seq `start_inflight_step(max_seqlen_q)`（:726）→ 生成 `SequenceMetadata`（:692-729）；GIL 在 :276 释放 |
| 准备输入 | `worker.py:121-124` → `model_runner.py:308-319` `execute_main_model` → `:149` `_prepare_input` → `:171` `_prepare_input_driver` → `:120` `_prepare_input_tensors` → `model_metadata_utils.py:146` → C++ `prepare_prompt_v4`/`prepare_decode_v4`（`prepare_tensor_input_v4.cpp:391`/`:562`） | driver 用 `seq_metadata_list` 建 pinned 张量 → `.recursive_fn(lambda x: x.cuda(non_blocking=True))`（:198）；CUDA graph 路径则写进 `input_metadata_source.to_dict()` 的复用缓冲（:186-190）；CP>1 时 `broadcast_tensor_dict`（:200-212） |
| forward | `model_runner.py:321-329` | `replay(cuda_graph_used)`（decode 且图可用）或 `self.model(kv_caches=..., input_metadata=...)`；外层 `forbid_cuda_sync()` + `cpu_timer/cuda_timer.record("launch_forward")`；`execute_main_model` docstring 明令"must NOT contain any cuda synchronize"（:317） |
| 采样 | `model_runner.py:337-367` `execute_sample` → `deepseek.py:383-410` `sample` → `sampler.py:114` `Sampler.forward` | `prepare_sampler_inputs`（:233-279）→ `get_logits`（:142）→ prefill 走 `_prefill_golden_gen`（:190），decode 走 `_verification`（:411）+ `mtp_module.decode_forward`（:237，DFlash graph replay） |
| 后处理（GPU→CPU 同步点） | `worker.py:143-144` → `:248-263` `_post_process_step` | `step_state.sample_end_evt.synchronize()`（:263）——**每步必经的同步点**（ZOS 下前面还有 `correction.d2h_evt.synchronize()`，`worker.py:149`；开 kvcache 落盘时 `pre_sampler_sync` 里还有 `copy_state_event.synchronize()`，`llm_engine.py:836`）；随后 `cuda_graph_timer.flush_out()`、`model_runner.post_process`（跑 sampler 注册的 `post_funcs`） |
| 后处理（账本/输出） | `worker.py:270-271` → `llm_engine.py:984-1015` → `llm_engine.py:687-798` | driver 上 `process_MTPInfo`（:722-728）→ `scheduler.process_outputs`（`scheduler.hpp:872`，append token / `_check_stop` / `commit_processed_tokens`）→ `get_finished_seqs_info`(:741) → `build_request_outputs`（`outputs.hpp:94`，返回 uint64 指针数组） |
| 输出交付 | `llm.py:179-186` | `RequestOutput.clone(ptr)` 复制一份，按 `request_id` 累积；`step_outputs` 在本步结束后由 `DeferredRelease` 回收（`llm_engine.py:974`、`:61-114`） |

ZOS 打开时顺序不变但**语义错位**：第 N 步的 `execute_model_step` 处理的是 `prev_step_state`（N−1 的输出），`schedule(N+1)` 在前一步 `sample_end_evt` 记录后立刻开始（`worker.py:182-187`）。官方文档对这条流水线的描述见 `doc/inference_deployment_qa.md:270-274`（B 级）。

### Q4 `Sequence` / `SequenceMetadata` / `SamplingParams` / 输入元数据分别是什么

**`Sequence`（C++，`csrc/engine/sequence.hpp:275`）** —— 一条请求的全部可变状态，按 7 个子结构聚合（`SequenceInit`，:263-273）：
- `request`（:168-173）：`seq_id`、`request_id`、`user_id`、`arrival_time`
- `input`（:175-179）：`tokens`（`SequenceData`）、`compressed_tokens`（Engram 压缩词表）、`multi_modal_data`（`py::object`）
- `generation`（:193-197）：`params`（`py::object` 指向 Python `SamplingParams`）、`sampling`（`SamplingScalars` 快照，:92-116）、`stopping`（`max_tokens`/`ignore_eos`）、`prompt_logprobs`
- `cache`（`KvCacheState`，:200-228）：`block_table`、`num_tokens`、`previous_num_tokens`、`hit_tokens`、`swa_valid_start`、`state_id`
- `mtp`（`MtpState`，:118-…）：`last_output_token`、`accepted_tokens`、`draft_tokens`、`accept_rates`、`info_slot`
- `structured_output`（:230-233）：`status`（0/1/2/−1）、`matcher`
- `schedule`（`SequenceScheduleState`，:235-259）：`status`（`SequenceStatus` 10 态，:30-41）、`schedulable`、`branch_id`、`chunked_rest_len`、`chunk_checkpoints`、`total_inflight_tokens`
- 关键方法：`get_num_blocks()`（:291-293，含 inflight + nextn）、`append_token_id`、`get_nextn_len`
- Python 侧只绑定少量属性（`pybind_sequence.cpp:113-140`：`status`、`num_cached_tokens`、`branch_id`、`chunk_checkpoints` 等）

**`SequenceMetadata`（C++，`sequence.hpp:325-419`）** —— 某一步的**只读快照 + 借用视图**：
- 借用（pointer/reference）宿主 `Sequence` 的字段：`seq_data`、`compressed_seq_data`、`block_table`、`prompt_tokens`、`prompt_logprobs`、`mtp.accepted_tokens/draft_tokens`
- 本步计算字段：`is_prompt`、`seqlen_q`、`seqlen_k`、`swa_valid_start`、`state_id`、`last_output_token`、`golden_token`、`mtp_info_read_slot/write_slot`、`so_status`、`so_matcher`
- 两种构造：对象池路径（`Sequence*` + 池分配，:374-389）与 pybind dummy 路径（`OwnedStorage`，:392-404，用于 placeholder / profile / CUDA graph padding）
- `refresh_for_sampling(seq, nextn)`（:408-418）：把 `last_output_token`、MTP slot、`golden_token` 刷成最新
- `is_last_prompt_chunk()`（:406）：`seqlen_k >= seq_data->get_len()`，非最后一个 chunk 时 `last_output_token=-1`（:386-388）

**`SamplingParams`（Python，`deepinfer/sampling_params.py:13-119`）**：
`temperature`、`top_p`、`top_k`、`ignore_eos`、`max_tokens`（0→1，:70-71）、`prompt_logprobs`、`top_logprobs`、`first_n_token_no_eos`、`eos_id`、`so_args`（结构化输出 schema/trigger）、`golden_token_ids`、`seq_seed`（64 位随机，:85-91）。
注意：`temperature < 1e-5` 会被改成贪心（:76-79）；`logprobs` 参数被显式丢弃（:63）。

**输入元数据（`deepinfer/model_executor/input_metadata/`）**：
- `InputMetadata`（`input_metadata.py:27-125`）：`is_prompt`、`input_ids[total_q]`、`positions[total_q]`、`ngram_token_ids[total_q, max_ngram]`、`valid_token_mask`、`gate_force_random`、`num_reqs`（"真实请求数，不含 cudagraph padding，非 driver CP rank 需要"，:52-53）、CP-mirror 三件套（`mirror_cp_index`/`reverse_index`/`cp_mirror_len`）
- `InputMetadataV4`（:217-…，`YocoMetadataMixin + DFlashMetadataMixin`）：压缩 KV 相关 `positions_c[N]`、`kv_ca_index[N]`、`kv_ca_index_offset_len[N]`、`slot_mapping_store_swa/ca`、`state_block_idx`；prefill 的 `cu_seqlens_q/k`、`load_segments_swa/ca`、`ks/ke`、`kv_swa_full_index(_pivots)`；decode 的 `block_tables`、`klen`、`query_indices`、`context_lens`、`kv_swa_index(_offset_len)`
- `InputMetadataVL`（:417-…）：`cu_seqlens_vit`、`max_seqlen_vit` 等视觉塔字段；`InputMetadataV4VL`（:472）多重继承
- 工具方法：`to_dict()`（浅 `__dict__`）、`recursive_fn`（H2D / 切片 / clone）、`recursive_copy_`（graph 复用缓冲原地写）、`to_flatten_dict`/`from_flatten_dict`（CP 广播）、`broadcast()`（原地广播）、`get_clean_list()`（graph 复用前清空派生字段）
- 生成方式：C++ `prepare_prompt_v4`/`prepare_decode_v4` 返回 `py::dict`，Python 用 `input_metadata_class(is_prompt=..., num_reqs=..., **dict)` 装成 dataclass（`model_runner.py:142-146`）

**另有**：`SchedulerOutputs`（`scheduler.hpp:45-77`：`scheduled_seqs`/`prompt_run`/`num_batched_tokens`/`finished_seqs`/`ignored_seqs`/`seqlens_q`/`clone_states` 等）、`SamplerOutput`（`deepinfer/sequence.py:172-…`，含 `sampled_token`/`verified_tokens`/`predict_tokens`/`post_funcs`）、`MetadataZOS`（`worker/zos_status.py:23-40`）、`WorkerStepState`（`zos_status.py:9-20`）。

### Q5 CUDA Graph 在哪里、用于哪些阶段

**位置**：
- `deepinfer/worker/cuda_graph_runner.py`：主模型图。`CudaGraphData`（:26-38，5 个字段：`graph`/`graph_handle`/`output`/`input_metadata_source`/`input_metadata_device`）、`build_graph_batch_sizes`（:41-51）、`_run_model`（:54-66）、`capture`（:82-141）、`select`（:143-148）、`replay`（:150-154）
- `deepinfer/worker/dflash_cuda_graph_runner.py`：MTP/DFlash draft 图（:31-99），按 **请求数** 选图（`select(req_cnt)`，:77-82），`do_sample` 只在 driver 为 True
- `csrc/graph_timer/`：图内计时探针。`torch.cuda.CUDAGraph(keep_graph=True)`（`cuda_graph_runner.py:124`）+ `graph_timer.GraphHandle(graph)`（:135）+ `cuda_graph_timer.flip_event_buffer(...)`（:152），NPU 上不支持 `keep_graph`（:123-124 注释）
- 捕获入口：`model_runner.py:515-529` `capture_model`；`model_runner.py:532-559` `capture_dflash_model`；调用者 `base.py:274-288` `warm_up_model`
- 回放入口：`model_runner.py:326-327`（主模型）、`mtp/dflash.py:383-387`（draft）

**覆盖阶段**（A）：
1. **只覆盖 decode 的主模型 forward**：`run_cuda_graph = not is_prompt and ...`（`model_runner.py:182`）；PROMPT engine 完全不建图（:73）。
2. **不覆盖**：prefill、`Sampler`、`prepare_*`（CPU/H2D）、MTP 验证（`_verification` 吃图输出后 eager 执行）、EPLB 相关 kernel。
3. **DFlash draft forward 单独建图**，在 `Sampler.forward` 内部 replay（`sampler.py:237` → `dflash.py:386`）。
4. **ZOS 下 graph 的输入缓冲在 device 上**（`input_on_device=True`，`model_runner.py:78`），`CudaGraphData` docstring："Source tensors are on CPU, or share device tensor storage under ZOS"（:28-32）；CP>1 时 `_run_model` 里 `input_metadata_device.broadcast()`（:64-65）让 follower 在 replay 前拿到一致输入。
5. **图尺寸 = token 数**：`max_batch_size = max_num_seqs * (1 + nextn)`（:77），`select(total_q)`（:185）。
6. `profile_decode_perf`（:562-…）用图 replay 测每个 batch size 的耗时，产出 `decode_perf_curve` 灌给 `Scheduler._mtp_varlen`（`llm_engine.py:268`），并可按模型+并行配置缓存（`decode_perf_cache.py`）。

### Q6 C++ 部分（`csrc/engine/`）承担了哪些原来在 Python 里的工作？为什么用 C++？

**承担的工作**（按模块）：
| 模块 | 文件 | 承担了什么 |
|---|---|---|
| 序列/状态 | `sequence.hpp`（18 KB）、`pybind_sequence.cpp` | `Sequence`/`SequenceData`/`SequenceMetadata`/`SequenceStatus`/`SamplingScalars`/`MtpState` 全部迁移到 C++；Python 只持有 `shared_ptr` 与属性代理 |
| 调度 | `scheduler.hpp`（64.8 KB）、`pybind_scheduler.cpp` | `schedule/_schedule_prefill/_schedule_decode/process_outputs/_check_stop/_preempt/fork_seq/abort_sequence`；DP 投票；MTP varlen 估时（`mtp_varlen.hpp`）；prefill 的 chunk/truncation 决策 |
| 块管理 | `block_manager.hpp`、`pybind_scheduler.cpp:63-71` | `BlockManager` 双向链表 free list、`pin/unpin`、`BlockManagerMutexAcquire` 双管理器加锁 |
| 输入准备 | `input_metadata/prepare_tensor_input_{base,v4}.cpp`、`block_load_plan.cpp` | 从 `SequenceMetadata*` 列表直接生成 pinned 张量字典；只算 offset/len |
| 采样输入 | `prepare_sample.cpp` | 批量装 `SamplerInputs`（`DecodeSamplingInputs`/`PrefillSamplingInputs`/`DriverSamplingInputs`） |
| ZOS | `zos.cpp` | `prepare_zos_inputs`（prev/curr batch 求 remap、按 remap 只写 fresh 行）、`update_metadata_for_sampling_zos` |
| 输出 | `outputs.hpp`、`pybind_output.cpp` | `RequestOutput` 池 + `build_request_outputs` + `get_finished_seqs_info` |
| 杂项 | `flops_estimator.hpp`、`cuda_utils.cpp`、`graph_timer/` | TFLOPS 估算、设备工具、图内计时 |

**为什么用 C++**（A 级证据，按重要性）：
1. **释放 GIL 做 CPU 侧工作**：`scheduler.hpp:276`（schedule）、`outputs.hpp:104`（build_request_outputs）、`prepare_sample.cpp:42`、`pybind_scheduler.cpp:67-71`（block pin/unpin/lock）、`pybind_sequence.cpp:52-55`（ZOS prepare）、`pybind_sequence.cpp:66`（`encode_compressed`）。Python 主线程在等 GPU 时仍要跑调度/后处理，若持 GIL 会卡住 tqdm、SO 编译、多模态线程。
2. **零分配的对象池**：`SequenceMetadata : EnableObjectPool`（`sequence.hpp:325`）、`SequenceMetadataPool`（`llm_engine.py:271`）、`RequestOutputPool`（:230）、`csrc/utils/object_pool.hpp`；配合 `DeferredRelease`（`llm_engine.py:61-114`）延后一轮释放，避免 UAF 又不用每步 malloc。
3. **借用静态/成员缓冲，避免拷贝**：`outputs.hpp:176` 的 `static std::vector<uint64_t>`、`scheduler.hpp:279-284` 注释"上一个 step 的 SchedulerOutputs 借用这三个 vector 的存储…Python 侧必须在下一次 schedule 前消费完"、`scheduler.hpp:879` 注释"返回的 numpy array 借用这块存储（空 capsule）"。
4. **CPU 侧成本必须可预测**：`prepare_tensor_input_v4.cpp:608-612` 明确"只算 offset/len，避免过大的 CPU 开销"，把 O(total_q × max_model_len) 的 index 展开推到 GPU/forward。
5. **数据布局与 Python 解耦**：`Sequence` 的 7 个子结构 + `SequenceInit`（`sequence.hpp:263-273`）把"扁平 Python 参数"在 pybind 边界只转换一次。
- 等级：A（每条都有源码依据；"为什么"的因果解释部分为 E）
- 意味着什么：书的"为什么把调度器搬到 C++"一节不该写成"性能优化"，而应写成**为了让 CPU 在 GPU 忙时能无 GIL、无分配、无拷贝地改账本**——这三条正是 ZOS 能成立的前提。
- 疑点：没有任何设计文档说明迁移前的 Python 版本长什么样（git 只有 4 个压缩 commit，见 R03 #15），所以"原来在 Python 里"只能从 vLLM 血统推断（E）。

### Q7 `deep_cluster/` 如何做 CPU/NIC/GPU 拓扑绑定？

**链路（A）**：
1. **可见设备切片**：`examples/_config.py:55-63` 按 `engine_rank`/`cp_size` 把 `CUDA_VISIBLE_DEVICES` 设成连续 `cp_size` 张卡（Ascend 同步设 `ASCEND_RT_VISIBLE_DEVICES`）。
2. **GPU → 物理 id → NUMA**：`deep_cluster/hardware/nvidia.py:55-67` 读 `/sys/bus/pci/devices/<bus_id>/numa_node`（`bus_id` 由 `physical_device_id(visible_ordinal)` 从 sysfs 枚举换算，`device.py:81-91`）；Ascend 走 `npu_smi`（`hardware/ascend.py:108-133`）。
3. **进程/内存绑定**：`deep_cluster/topology/bind.py:79-93` `bind_numa(node)`：`libnuma.numa_node_to_cpus` → `numa_sched_setaffinity(getpid(), cpumask)`，再 `numa_bitmask_setbit` + `set_mempolicy(MPOL_BIND, ...)` 把**内存分配**钉在该 NUMA 节点。调用点 `worker/base.py:185`（`_bind_numa(self.local_rank)`）和 `llm_engine.py:212`（多模态路径 `_bind_numa(0)`）。
4. **CPU 亲和（更细一层）**：`deepinfer/utils.py:143-151` `bind_cpu(local_rank)`：`phy_cpus // ngpus` 均分物理核，取第 `local_rank` 段，`deep_cluster.topology.set_cpu_affinity` → `psutil.Process().cpu_affinity`。调用点 `worker/base.py:191`。
5. **CPU 拓扑解析**：`deep_cluster/hardware/cpu.py:9-47` 解析 `lscpu -p=CPU,NODE,SOCKET,CORE,ONLINE`，按 `(numa, socket, core)` 只保留每核最小逻辑 CPU（去掉 SMT 兄弟）。
6. **NIC**：`topology/bind.py:65-71` 提供 `which_numa(i_ib=...)`，读 `/sys/class/infiniband/mlx5_<i>/device/numa_node`；**全仓库无调用者**（`grep -rn "which_numa("` 只有 GPU 形式的两处：`base.py:355`、`ve/multimodal_processor.py:178`）。
7. **失败降级**：`_bind_numa` 整个包在 `try/except` 里，失败只 `logger.warning("NUMA 绑定失败，内存落位退化为 first-touch，可能导致 H2D 性能下降")`（`base.py:356-357`）——说明作者认为绑不上不是致命错误。
- 等级：A（1-6 步代码可见；7 是原注释）
- 意味着什么：书上必须写清"绑定 = 三层"：可见设备（谁看得见哪张卡）→ NUMA（进程/内存在哪个节点、`MPOL_BIND` 生效于后续 malloc）→ CPU 亲和（具体哪几个核）。NIC 只留了 helper，不能写成"DeepInfer 会按 IB 网卡绑 NUMA"。
- 疑点：
  - `bind_cpu(self.gpu_id)` 用物理 id 当 `local_rank` 索引 CPU 段（`utils.py:147` + `base.py:191`），与 `_bind_numa(self.local_rank)` 的序号语义不一致；`DEVICE_OFFSET>0` 时 CPU 段会错位（E）。
  - `which_numa(i_ib=...)` 的 `mlx5_{i_ib}` 硬编码网卡名前缀，且在容器里 `/sys/class/infiniband` 常被裁剪（E，未验证）。

## 4. 新发现的问题

1. **"prefix caching" 名不副实**（A）：`scheduler.hpp:373-399` 的 `prefix_caching_hit` 只处理 `branch_id` 以 `_` 开头的 **fork 请求**（复用父请求的 `block_table`，见 `:400-430` 的 clone 逻辑与 `fork_seq` 绑定 `pybind_scheduler.cpp:123`），`block_manager.hpp` 里**没有任何 block hash**。这与 vLLM 的 hash-based prefix caching 不是一回事，书里若混用会错。
2. **一个 step 里必经的显式同步点是 `sample_end_evt.synchronize()`**（A，`worker.py:263`；ZOS 下另加 `d2h_evt.synchronize()`，`worker.py:149`）。`execute_main_model`/`execute_sample` 都被 `forbid_cuda_sync()` 包住（`model_runner.py:324, 359`），torch 的 sync debug mode 会直接抛错——这是一条"不许偷偷同步"的硬约束，值得单独一节。
3. **`has_unfinished_requests` 每步一次全 DP all-reduce**（A，`llm_engine.py:665-678`）。即使 ZOS 把它挪到 gloo/CPU，它仍是每步的 CPU 阻塞点；对"空转 vs 退出"的判断精度是 `request_outputs` 报告 `finished` 的粒度。
4. **`flush_inflight_reqs` 在 follower 上是空操作**（A，`worker.py:281-282`：`prev_step_state` 为 None 时返回空数组）。ZOS 收尾时只有 driver 会真正 flush，follower 的 GPU 侧 kernel 是否已全部结束没有任何显式保证（除 `sample_end_evt` 同 stream 顺序）——不确定是否需要额外 barrier（E，疑点）。
5. **`CudaGraphRunner.select` 在 `total_q > max_batch_size` 时抛 ValueError**（A，`cuda_graph_runner.py:145-146`），而 `max_batch_size = max_num_seqs*(1+nextn)`；若 `seqlens_q` 之和因 MTP varlen 修正而超过该上界（`_correct_seqlens_qk` 允许 q 增长到 `max_seqlen_q`），是否会撞上这个断言，需查 `mtp_varlen` 的上界保证（未能确认）。
6. **`RequestOutput` 是裸指针**（A）：`RequestOutput.clone(ptr)` 之外的所有 `get_*` 都是 `reinterpret_cast`（`pybind_output.cpp:14`）。如果 Python 侧缓存了指针而 `DeferredRelease.release()` 已经跑过，就是 UAF；`llm.py` 靠"每步立即 clone"规避，没有类型层面的保护。
7. **`deepinfer/_C` 的 `.pyi` 与实现可能漂移**：`engine.pyi` 有 33 个导出符号，`pybind.cpp:5-9` 只装配 `sequence/scheduler/output` 三个子模块；`prepare_zos_inputs`/`prepare_decode_zos_inputs` 实际绑在 `pybind_sequence.cpp:52-55`，不在文件名暗示的 `zos.cpp` 绑定点（`zos.cpp` 只被 `pybind_scheduler.cpp` 引用 `update_metadata_for_sampling_zos`，:52）。对读者无碍，对维护者是坑。
8. **`LLM` 没有 online serving 入口**（A，`llm_engine.py:140-141` 注释自陈"This release does not include an online-serving wrapper"），所以本笔记的"driver/worker 通信"只覆盖离线批处理形态；PD 分离（`EngineType.PROMPT`/`COMPLETION`）的跨进程传输在 `init_pd_split`（`llm_engine.py:801`）/`send_processed_prompts`（:848-849）里，本次未展开。

## 5. 可用于图表的材料

**图 A：进程/通信拓扑（单机 8 卡，`--ep 8 --cp 1`）**
```
launch_dp 父进程 (RANK=0, WORLD_SIZE=1)
├── engine_rank 0  [CUDA_VISIBLE_DEVICES=0]  ← 进程内：LLMEngine + Scheduler + BlockManager + RequestOutputPool + driver Worker
├── engine_rank 1  [CUDA_VISIBLE_DEVICES=1]  ← 同上（各自独立 Scheduler，DP 副本）
├── ...
└── engine_rank 7  [CUDA_VISIBLE_DEVICES=7]
     EP 组 = world(8)，DP 组 = world(8)（每步投票），CP 组 = 单 rank
```
`--cp 2` 时：4 个 engine 进程，每个 `[CVD=2k,2k+1]`，进程内 driver(cp_rank=0) + 1 个 spawn 子进程(cp_rank=1)；`MessageQueue` 只在这两个进程间建立。

**图 B：解码步时序（ZOS 关）**
```
CPU:  schedule(N) ──► prepare(N) ──► launch_forward(N) ──► launch_sample(N) ──► [sync] post(N)
GPU:                                  forward(N) ─────────► sample(N) ────────► (idle)
```
**图 C：解码步时序（ZOS 开，官方文档 + 代码对齐）**
```
GPU:  forward(N) ──► sample(N) ──► forward(N+1) ──► sample(N+1)
CPU:  schedule(N+1) ──► launch_forward(N+1) ──► post(N-1) ──► launch_sample(N+1)
       ↑ worker.py:121-187 的固定顺序；post(N) 被推迟到 N+1 步的 :144
```
来源：`doc/inference_deployment_qa.md:274`（B）+ `worker.py:96-104`（A）。

**表 D：C++ / Python 职责分界**
| 关注点 | 在 Python | 在 C++ |
|---|---|---|
| 请求入队 | `LLMEngine.add_request`（:472） | `Sequence` 构造、`Scheduler::add_sequence` |
| 调度决策 | — | `Scheduler::_schedule*`（`scheduler.hpp:275/351/559`） |
| 元数据快照 | 只读属性 | `SequenceMetadata` + `ObjectPool`（`sequence.hpp:325`） |
| 输入张量 | dataclass 装填（`model_runner.py:142`） | `prepare_*_v4`（`prepare_tensor_input_v4.cpp:391/562`） |
| 前向/采样 | `DeepseekModel.forward`、`Sampler.forward` | 只到 `torch::Tensor` 层（`cuda_utils`、`graph_timer`） |
| 输出 | `RequestOutput.clone`（`llm.py:181`） | `build_request_outputs`（`outputs.hpp:94`） |

**表 E：CUDA Graph 覆盖面**
| 阶段 | 是否建图 | 依据 |
|---|---|---|
| prefill forward | 否 | `model_runner.py:182` |
| decode 主模型 forward | 是（PROMPT engine / `ENFORCE_EAGER` 除外） | `model_runner.py:73, 182` |
| DFlash draft forward | 是（driver/follower 各一套，follower 不含 sampling） | `model_runner.py:80-90`、`dflash_cuda_graph_runner.py:32-34` |
| Sampler / verification | 否 | `execute_sample` 无 graph 分支 |
| prepare / H2D / 后处理 | 否 | CPU 侧 |

**数据点（可引用）**
- 图 batch size 集合（`max_num_seqs=4, nextn=5, interval=8`）：`build_graph_batch_sizes(24, 8)` → `[1,2,4,8,16,24]`（`cuda_graph_runner.py:41-51` 手算，E）。
- SHM 环参数：`1 MiB × 10`（`parallel_state.py:124`）。
- 默认 PG 后端 = gloo；模型集合通信 = nccl/hccl（`base.py:334`、`platform.py:58/89`）。
- 默认投机长度：`DEEPINFER_USE_NEXTN_DECODE = 5`（`environ.py:129`）。
