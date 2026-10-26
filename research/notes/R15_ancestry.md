# R15 血统：DeepInfer 与 vLLM 的历史分叉、23 个派生文件比对、借鉴与删改

## 0. 元信息

- 使用的代码版本：
  - DeepInfer `deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21）
  - vLLM `vllm-project/vllm` @ `2d40c0609fd672e845ddfef0324ab6d8dd616af4`（main，2026-10-22）
  - 为做历史对照，用 `git fetch --depth=1 origin refs/tags/<tag>:refs/tags/<tag>` 把 vLLM 的 `v0.2.0 v0.2.2 v0.2.5 v0.2.7 v0.3.2 v0.4.0 v0.4.3 v0.5.4 v0.6.0 v0.6.3 v0.7.3 v0.8.5 v0.9.1` 拉进本地浅克隆（只增加 tag 对象，未改动工作树）
- 对照方法（可复现）：
  1. 对每个派生文件，按 tag 取 vLLM 同名文件，去掉空行与 `#` 注释行、每行 `strip()` 后，用 `difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()` 算相似度；
  2. 相似度绝对值会被"重写"压低（DeepInfer 大面积改名/删依赖），**只有跨 tag 的相对高低有意义**；
  3. 再用"API 指纹"（函数签名、配置元组、方法名）独立验证相似度指向的年代。
- 查阅的主要来源：
  - DeepInfer：`deepinfer/engine/llm_engine.py`、`deepinfer/worker/{base,worker,model_runner,cache_engine}.py`、`deepinfer/model_executor/layers/{sampler,sampler_utils}.py`、`deepinfer/model_executor/layers/tilelang_ops/fused_sample.py`、`deepinfer/sequence.py`、`deepinfer/config.py`、`deepinfer/engine/arg_utils.py`、`deepinfer/entrypoints/llm.py`、`csrc/engine/{scheduler,sequence,block_manager}.hpp`、`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`、`README.md`、`doc/inference_deployment_qa.md`
  - vLLM：`v0.2.0/v0.2.7/v0.3.2/v0.4.3/v0.6.3` 的 `vllm/engine/{llm_engine,arg_utils}.py`、`vllm/worker/{worker,worker_base,cache_engine,model_runner}.py`、`vllm/sequence.py`、`vllm/sampling_params.py`、`vllm/config.py`、`vllm/entrypoints/llm.py`、`vllm/core/scheduler.py`；main 的 `vllm/v1/{engine,core,worker,sample}`
  - SGLang：`python/sglang/srt/managers/scheduler.py`（`event_loop_overlap`）
  - 外部：NanoFlow（arXiv 2408.12757，2024-08 v1 / 2025-05 v2）摘要

## 1. 三个最重要的发现

1. **这不是"某个 vLLM 版本的 fork"，而是"骨架冻结在 vLLM v0.2.x（2023-09 ~ 2024-01），个别基础设施文件后来从 v0.6.3+ 同步"的两层结构。** 21 个引擎骨架文件的**平均**相似度在 v0.2.7 达到峰值 0.174（v0.3.2 0.170），v0.4.0 之后单调下降（v0.4.3 0.115 → v0.6.3 0.071 → main 0.023）；`create_engine_configs()` 返回 4 元组、`profile_num_available_blocks`/`init_cache_engine`、`InputMetadata` 位于 `model_executor/` 这三个 API 指纹只可能来自 v0.2.x；但 `distributed/shm_broadcast.py` 与 v0.6.3 的相似度是 0.68，明显是后来搬进来的。
2. **vLLM 留下的是"壳"，DeepSeek 换掉了"芯"。** 23 个文件里只有 4 个还算接近原样（`entrypoints/llm.py` 0.49、`layernorm.py` 0.41、`rotary_embedding.py` 0.40、`models/__init__.py` 0.38）；四个核心文件 `llm_engine.py`/`worker.py`/`cache_engine.py`/`sampler.py` 与各自最像的 vLLM 版本也只有 0.16/0.09/0.05/0.04。保留下来的是类名、方法名、装配顺序、文档字符串；被替换的是调度器（Python→C++）、KV cache 布局（per-layer dtype 张量→单块 uint8）、采样（Python torch→TileLang kernel）、元数据准备（`_prepare_prompt/_prepare_decode`→C++ `prepare_prompt_v4`）。
3. **"缺失的通用能力"要分三类说，不能一句"删了"：**（a）**确实被删**——LoRA、AWQ/GPTQ/SqueezeLLM 量化后端、TP/PP、CPU swap、多模型注册表（27→1）、beam search、penalties/logits_processors；（b）**大概率是发布裁剪而非代码删除**——OpenAI API Server、AsyncLLMEngine、内置 KV cache 系统与负载均衡（Q&A:21 对 KV cache/负载均衡明说"tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced"；API server 无直接文字证据，属 E）；（c）**被替换**——抢占的 swap 分支删掉、recompute 保留但默认改成 kill。

## 2. 事实条目

### #1 23 个文件用同一行文件头声明血统，全部落在引擎骨架层
- 证据：`grep -rn "Derived from vLLM" --include="*.py" . | grep -v vendor/` → 23 个文件、6,817 行；每个文件第 2 行都是 `# Derived from vLLM; modified by DeepSeek.`（第 1 行 `# SPDX-License-Identifier: Apache-2.0`）。`csrc/`、`tilelang_ops/`、`v4_modules/`、`ve/`、`hugging_face/` 零命中。
- 等级：A
- 意味着什么：vLLM 的影响只在"引擎胶水层"，模型结构、kernel、多模态、EPLB 都是自研；书里讲架构时可以把"派生层"和"自研层"当成两个世界。
- 疑点：无。

### #2 相似度扫描把骨架定位在 v0.2.x：21 个骨架文件的平均相似度在 v0.2.7 见顶，之后单调下降
- 证据（本次实测，非注释非空行，`difflib` ratio）：

  | 文件 | v0.2.7 | v0.4.3 | v0.6.3 | v0.9.1 | main |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | `engine/llm_engine.py` | 0.14 | 0.10 | 0.08 | 0.06 | 0.00 |
  | `config.py` | 0.22 | 0.10 | 0.06 | 0.01 | 0.00 |
  | `sampling_params.py` | 0.27 | 0.22 | 0.13 | 0.10 | 0.02 |
  | `engine/arg_utils.py` | 0.10 | 0.05 | 0.03 | 0.01 | 0.01 |
  | `worker/cache_engine.py` | 0.05 | 0.05 | 0.04 | 0.04 | 0.00 |
  | **21 个骨架文件均值** | **0.174** | **0.115** | **0.071** | **0.051** | **0.023** |
  | `distributed/shm_broadcast.py`（对照，不在均值内） | 0.00 | 0.00 | **0.68** | 0.58 | 0.38 |
- 等级：A（数字可复现）+ E（"v0.2.x 是分叉基线"的推断）
- 意味着什么：血统判断不能靠"看起来像老 vLLM"。逐文件看，23 个里除 `shm_broadcast.py`（v0.6.3）及其测试（v0.8.5）外，其余 21 个中有 20 个的"最高分 tag"落在 v0.2.0–v0.4.3 之间（多数在 v0.2.x–v0.3.2）；唯一例外 `sequence.py` 的相似度只有 0.03、各 tag 无差别，属噪声；但单文件差距只有 0.01–0.03 量级、噪声很大，所以**只有聚合均值可信**：0.174(v0.2.7) → 0.115(v0.4.3) → 0.071(v0.6.3) → 0.023(main)。
- 疑点：相似度会被"删代码"系统性压低（分母是 DeepInfer 行数），所以 0.05 这种值只能说明"几乎全改过"，不能说明"改自哪个版本"；必须配合 #3–#7 的 API 指纹。

### #3 API 指纹一：`create_engine_configs()` 返回 4 元组，只有 v0.2.x 是这样
- 证据：`deepinfer/engine/arg_utils.py:62-64`
  > `def create_engine_configs(self,) -> tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:`
  vLLM v0.2.0/v0.2.7 同为 4 元组；v0.3.2 起变成 6 元组（多 `DeviceConfig`、可选 `LoRAConfig`），v0.4.0 变 7 元组，v0.4.3 起改名 `create_engine_config()` 返回 `EngineConfig`。
- 等级：A
- 意味着什么：这是"分叉点在 v0.3.0 之前"最硬的一条证据；DeepSeek 后来把 `DeviceConfig` 换成了自己的 `deep_cluster.hardware.accelerator_kind()`。
- 疑点：也可能是在 v0.3.x 上手动把 6 元组改回 4 元组，但没人会做这种反向改造。

### #4 API 指纹二：`profile_num_available_blocks` / `init_cache_engine` / `warm_up_model` 三连，v0.4 起被改名
- 证据：`deepinfer/worker/base.py:212` `def profile_num_available_blocks(self) -> int:`、`:270` `def init_cache_engine(self) -> None:`、`:274` `def warm_up_model(...)`；vLLM v0.2.7/v0.3.2 的 `Worker` 同名同序；v0.4.3 已改为 `determine_num_available_blocks()` / `initialize_cache()` / `_warm_up_model()`（`v0.4.3:vllm/worker/worker.py:136,181,197`）。
- 等级：A
- 意味着什么：连"资源层方法的命名 generation"都停在 2024-02 之前；书的第 3 章讲 worker 生命周期时可以直接按这四个阶段讲。
- 疑点：无。

### #5 API 指纹三：`InputMetadata` 还在 `model_executor/` 目录下
- 证据：`deepinfer/model_executor/input_metadata/input_metadata.py:26-27` `@dataclass class InputMetadata:`；vLLM v0.2.7/v0.3.2 为 `vllm/model_executor/input_metadata.py`，v0.4.3 起该类被并入 `vllm/worker/model_runner.py`（`git ls-tree` 可验，v0.4.3 无此文件）。
- 等级：A
- 意味着什么：DeepInfer 保留的是"元数据是模型执行层的公共类型"这一早期分层，后来 vLLM 把它下沉成 runner 私有类型，DeepInfer 反而把它扩成了一个包（`input_metadata/` + `dflash.py` + `yoco.py`）。
- 疑点：目录名相同不能排除"DeepSeek 自己重新拆的包"，但与 #3/#4 同时成立时，指向同一时间段。

### #6 API 指纹四：`SamplingParams` 的字段集是 v0.2.x 的子集
- 证据：`deepinfer/sampling_params.py:47-62` 只有 `temperature/top_p/top_k/ignore_eos/max_tokens/logprobs/prompt_logprobs/top_logprobs`（+ DeepSeek 自己的 `first_n_token_no_eos/eos_id/so_args/golden_token_ids/seq_seed`）。vLLM v0.2.7 已有 `n/best_of/presence_penalty/frequency_penalty/repetition_penalty/min_p/seed/use_beam_search/length_penalty/early_stopping/stop/stop_token_ids/min_tokens/logits_processors/truncate_prompt_tokens`（`v0.2.7:vllm/sampling_params.py:96-140`）。
- 等级：A
- 意味着什么：不是"从更新版本删字段"，而是"从更老版本留下来"——`min_tokens`/`logits_processors`/`truncate_prompt_tokens` 这些 v0.2.7 已经有的东西一个都不在。
- 疑点：`seq_seed` 是 DeepSeek 加的（vLLM V0 只有全局 seed），说明这个文件后来被改过，只是没跟着上游加字段。

### #7 反例：`shm_broadcast.py` 是后来从 v0.6.3+ 搬进来的，血统是"分层"的
- 证据：`deepinfer/distributed/shm_broadcast.py:33` `class ShmRingBuffer`、`:162` `class MessageQueue`、`:432` `def create_from_process_group(`；vLLM v0.6.3 `vllm/distributed/device_communicators/shm_broadcast.py`（491 行）→ DeepInfer 461 行，非注释行相似度 0.68；v0.2.7 全仓无 `shm_broadcast`（当时只有 `parallel_state.py`）。配套测试 `tests/distributed/test_shm_broadcast.py` 与 v0.9.1 相似度 0.52。
- 等级：A
- 意味着什么：DeepSeek 不是"fork 完就冻结"，而是"引擎骨架不动，缺的基础设施从最新 vLLM 摘一个文件过来"；这也解释了为什么单看某个文件会得出完全不同的年代结论。
- 疑点：为什么只同步这一个文件（大概率为了 CP 的 `cp_broadcast_object`），没有 commit 记录可查（仓库只有 4 个提交，见 R03 Q4）。

### #8 `LLMEngine.__init__` 的装配顺序逐行保留
- 证据：`deepinfer/engine/llm_engine.py:200-205` tokenizer、`:222` `self.seq_counter = Counter()`、`:233` `self._init_workers()`、`:241` `self._run_workers("load_model", ...)`、`:244` `self._init_cache()`、`:265-267` `BlockManager(...)` + `Scheduler`；vLLM v0.2.7 `__init__` 为 tokenizer → `seq_counter` → `_init_workers` → `_init_cache` → `Scheduler`（`v0.2.7:vllm/engine/llm_engine.py:96-117`）。
- 等级：A
- 意味着什么：连"先起 worker、再 load model、再 profile KV、最后造 scheduler"这个顺序都没变；变的是每步内部（Ray→多进程、Python Scheduler→pybind11 `Scheduler`）。
- 疑点：无。

### #9 `_init_cache` 的五步骨架保留，但 CPU blocks 被删、跨 rank 取 min 改成 all_reduce 均值
- 证据：`deepinfer/engine/llm_engine.py:416-458` 依次 `_run_workers("profile_num_available_blocks")` → 检查 `num_gpu_blocks<=0` → 检查 `max_seq_len` → `_run_workers("init_cache_engine")` → `_run_workers("warm_up_model")`；vLLM v0.2.7 `_init_cache` 同五步，但取 `min(b[0] for b in num_blocks)` 且同时算 `num_cpu_blocks`（`v0.2.7:...:259-299`）。DeepInfer 在 worker 内部 `dist.all_reduce(SUM) / world_size // 2 * 2`（`deepinfer/worker/base.py:264-266`），并 `available_kv_cache_memory -= state_size * 2`。
- 等级：A
- 意味着什么："min 改均值 + 偶数对齐"是为了 CP/EP 各 rank 的 **state cache** 大小一致；`max_seq_len` 还乘了 0.8 的安全系数，属于 DeepSeek 的生产化改动。
- 疑点：为什么用均值而不是 min（均值可能让某个 rank 超分），代码没写理由；推测是各 rank 显存对称、均值更接近真实可用量（E）。

### #10 `add_request` 的 docstring 原样保留，实现换成"预 tokenize + C++ Sequence"
- 证据：`deepinfer/engine/llm_engine.py:485-493` 仍是 vLLM 的 "Add a request to the engine's request pool..."；`:594-598` 直接 `Sequence(prompt_token_ids=..., compressed_prompt_token_ids=..., **sequence_args)`，`Sequence` 是 C++ 类，由 `deepinfer/sequence.py:14-20` 从 `_C.engine` 重导出、`llm_engine.py:47` 导入。vLLM v0.2.7 的 `add_request` 收 `prompt: Optional[str]`、内部 tokenize、再 `_create_sequence_group_with_sampling()` 造 `SequenceGroup`（`v0.2.7:...:314-355`）。
- 等级：A
- 意味着什么：DeepInfer 把"字符串→token"上移到 entrypoint（`entrypoints/llm.py:_add_request`），引擎只收 `np.ndarray[np.int32]`；`SequenceGroup` 整个概念消失（没有 n/best_of/beam）。这是"离线批处理引擎"和"在线服务引擎"的分界线。
- 疑点：`add_request` 多了 `user_id`、`extra`、`enqueue`、`chunk_checkpoints` 参数，说明它同时被 PD 分离/多模态/分块 checkpoint 复用，不再是 vLLM 那个语义。

### #11 `step()` 的返回类型从 `List[RequestOutput]` 变成 `np.ndarray[np.uint64]` 句柄数组
- 证据：`deepinfer/engine/llm_engine.py:860-871` 文档明写 "The returned RequestOutput objects are managed by RequestOutputPool. RequestOutput will be automatically released in the next step() call."；`:952` 返回 `self._execute_model_async_step(...)`。vLLM v0.2.7 `step()` 返回 `List[RequestOutput]`（`v0.2.7:...:615-642`）。
- 等级：A
- 意味着什么：为了 ZOS 里"上一步的输出还没处理完、下一步已经开始"，输出对象必须能跨 step 存活，所以改成对象池 + 句柄；entrypoint 必须 `RequestOutput.clone()`（`entrypoints/llm.py:181`）才能留到下一轮。
- 疑点：句柄是裸指针语义，越界/双重释放只能靠 `DeferredRelease` 保证；这是 C++ 化的代价。

### #12 `_process_model_outputs` 只剩"一个扁平 SamplerOutput → C++ 状态机"，beam/seq-group 分支全删
- 证据：`deepinfer/engine/llm_engine.py:687-785`，签名 `def _process_model_outputs(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs)`，核心是 `self.scheduler.process_outputs(scheduler_outputs, accepted_token_ids, output_token_ids, so_status)`；vLLM v0.2.7 是 `_process_model_outputs(output, scheduled_seq_groups, seq_group_metadata_list)` + `_process_sequence_group_outputs(seq_group, ...)`（`v0.2.7:...:591-613,420-590`），后者处理 beam search 的 `beam_search_metadata`。
- 等级：A
- 意味着什么：vLLM 的"每个 seq group 一份输出"变成"整批一份 tensor"，per-seq 记账下沉到 C++；这也是为什么 Sampler 不再需要 `SequenceGroupOutput`。
- 疑点：无。

### #13 C++ `Scheduler` 的方法名与 vLLM v0.2.7 一一对应
- 证据：`csrc/engine/scheduler.hpp:135` `add_sequence`、`:144` `abort_sequence`、`:165` `has_unfinished_seqs`、`:275` `_schedule`、`:351` `_schedule_prefill`、`:559` `_schedule_decode`、`:766` `_preempt`、`:817` `_append_slot`、`:738` `free_seq_nolock`、`:752` `free_finished_seqs`；vLLM v0.2.7 `vllm/core/scheduler.py:86 add_seq_group`、`:90 abort_seq_group`、`:111 has_unfinished_seqs`、`:117 _schedule`、`:311 free_seq`、`:314 free_finished_seq_groups`、`:325 _append_slot`、`:339 _preempt`。
- 等级：A
- 意味着什么：这不是"重写了一个调度器"，是"把 vLLM 的 Scheduler 逐方法翻译成 C++ 再改"；连 `AllocStatus{LATER→break, NEVER→FINISHED_IGNORED}` 的分支结构都一样（`scheduler.hpp:388-404` vs `v0.2.7:scheduler.py:117-280`）。
- 疑点：`_schedule_prefill`/`_schedule_decode` 的拆分在 v0.2.7 里没有（v0.4.3 才有 `_schedule_prefills`），可能是 DeepSeek 自己拆的，也可能是从 v0.4.x 反向借的命名。

### #14 `_check_stop` 整体迁到 C++，且删掉了 stop string / stop_token_ids 两个分支
- 证据：`csrc/engine/scheduler.hpp:1035-1052`
  > `void _check_stop(Sequence &seq) { int token_id = seq.get_last_token_id(); if (!seq.generation.stopping.ignore_eos && token_id == eos_token_id) {...FINISHED_STOPPED...} if (seq.input.tokens.get_len() > max_model_len) {...LENGTH_CAPPED...} if (seq.get_output_len() == seq.generation.stopping.max_tokens) {...LENGTH_CAPPED...} }`
  vLLM v0.2.7 `LLMEngine._check_stop` 先查 `sampling_params.stop` 字符串、再查 `stop_token_ids`，最后才是 eos/max_len/max_tokens（`v0.2.7:...:737-766`）。
- 等级：A
- 意味着什么：DeepInfer 的"停止"只有三种：EOS、长度、max_tokens；stop 字符串匹配需要 detokenize，被整体去掉了（tokenizer 只在 entrypoint 出现）。书里讲"生命周期"时要说明这一点，否则读者会以为支持 OpenAI 的 `stop`。
- 疑点：`first_n_token_no_eos` 是 DeepSeek 加的（`sampling_params.py:57`），在 C++ 里怎么实现未查（可能在 sampler 的 eos 抑制里）。

### #15 `Sequence`/`SequenceData`/`SequenceStatus` 整体迁到 C++，API 名逐字保留
- 证据：`csrc/engine/sequence.hpp:30-41` `enum class SequenceStatus { WAITING, RUNNING, SWAPPED, FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED, FINISHED_PREEMPTED, ... }` 与 vLLM v0.2.7 `sequence.py:13-48` 完全同序；`sequence.hpp:291-303` `get_num_blocks/append_token_id/get_len/get_prompt_len/get_output_len/get_last_token_id/get_token_ids` 与 v0.2.7 `SequenceData` 方法同名同义（`v0.2.7:sequence.py:50-97`）。
- 等级：A
- 意味着什么：书里可以直接用 vLLM 的 `Sequence` 心智模型讲 DeepInfer 的 C++ `Sequence`，只需说明"多了 mtp/structured_output/state 三块状态"。
- 疑点：`SWAPPED` 状态保留但已无人设置（见 #19），是纯遗留枚举值。

### #16 Worker 拆成 `WorkerBase`（资源）+ `Worker`（单步），`execute_model` 改名 `execute_model_step` 并改成四段式
- 证据：`deepinfer/worker/base.py:49` `class WorkerBase`（`__init__` 里做完 init_model/init_dist）、`deepinfer/worker/worker.py:51` `class Worker(WorkerBase)`、`:89` `def execute_model_step(...)`；四段注释在 `worker.py:113`（forward curr）、`:143`（post process prev）、`:146`（sample curr）、`:182`（prompt/non-ZOS 当场 post，ZOS decode 留给下一步）。vLLM v0.2.7 只有 `vllm/worker/worker.py:20 class Worker`，单方法 `execute_model(seq_group_metadata_list, blocks_to_swap_in, ...)`（`:157`）。
- 等级：A
- 意味着什么：vLLM 的 `execute_model` 一步做完 forward+sample，DeepInfer 把它拆成 `execute_main_model` / `execute_sample` / `post_process` 三段，**是为了在 forward 和 sample 之间插入上一步的 post-process**——即 ZOS 的实现基础。
- 疑点：`WorkerBase` 这个名字来自 vLLM v0.4+（抽象接口），但 DeepInfer 的 `WorkerBase` 是**具体类**、装的是 v0.2.x `Worker` 的方法体；这是"名字借新的、内容用旧的"的典型混合。

### #17 `run_workers` 用 4 行 shm broadcast + 从进程循环取代 Ray/Executor，`cache_swap` 彻底删除
- 证据：`deepinfer/worker/base.py:290-302`
  > `def run_workers(self, method, driver_kwargs=None, **kwargs): ... if cp_group and cp_group.size() > 1: cp_broadcast_object((method, kwargs)) ... return getattr(self, method)(**(driver_kwargs or {}), **kwargs)`
  从进程侧 `:129-133` `while True: method, kwargs = cp_broadcast_object(None); getattr(self, method)(**kwargs)`。vLLM v0.2.7 `Worker.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)`（`:130-155`）在 DeepInfer 中无对应物（`grep "def swap\|def cache_swap" deepinfer/worker/` 零命中）。
- 等级：A
- 意味着什么：分布式执行被简化成"driver 广播方法名+kwargs，CP follower 本地执行同名方法"；没有 Ray、没有 Executor 抽象、没有 CPU swap，因此也不需要 `cache_swap` 这个 RPC。
- 疑点：这套机制只支持 CP follower（同引擎内的模型并行），不支持跨节点 DP 的 RPC——DP 是"多引擎 + 外部负载均衡"，不在引擎内。

### #18 `CacheEngine`：978 行里只有最后 128 行是 vLLM 的壳，docstring 还在承诺已经不存在的 CPU cache/swap
- 证据：`deepinfer/worker/cache_engine.py:851` `class CacheEngine`，`:852-857`
  > `"""Manages the KV cache. This class is responsible for initializing and managing the GPU and CPU KV caches. It also provides methods for performing KV cache operations, such as swapping and copying."""`
  但类里只有 `__init__`（862）、`get_block_shape`（890）、`get_kvcache_layout`（896）、`get_state_shape`（902）、`do_allocate_kv_cache`（907）、`_alloc_gpu_uint8`（911）、`allocate_gpu_cache`（918）、`get_cache_block_size`（969）；`allocate_cpu_cache/swap_in/swap_out/_swap/copy/get_key_block_shape/get_value_block_shape` 全部不存在。文件前 800 行是全新的 `KVCache`（206）、`StateBlock`（221）、`KVCacheManager`（309）、`GPUCacheInterface`（540）。
- 等级：A
- 意味着什么：这是"壳留下、芯换掉"最直观的证据；书里可以引用这段"名不副实"的 docstring 说明分叉后的演化速度——文档停在 2023 年，实现已经是 2026 年的 V4.1 布局。
- 疑点：`CacheEngine` 的 `gpu_cache` 仍是 `self.gpu_cache`（`base.py:271`），但类型从 `List[Tuple[Tensor,Tensor]]` 变成 `GPUCacheInterface`，所有 attention 后端都按新类型取用。

### #19 抢占：swap 分支删除、recompute 保留、默认改成"直接杀"
- 证据：`csrc/engine/scheduler.hpp:766-791`
  > `const bool so_blocks_recompute = seq->structured_output.status != 0; if (stop_preempted_request || so_blocks_recompute) { ...FINISHED_PREEMPTED... free_seq_nolock... } else { seq->reset_for_recompute(); waiting.push_front(seq); ...WAITING... }`
  vLLM v0.2.7 `_preempt` 二选一：`_preempt_by_recompute` 或 `_preempt_by_swap`（`v0.2.7:scheduler.py:339-388`）。默认值：`EngineArgs.stop_preempted_request=True`（`arg_utils.py:40`），但离线 `LLM.__init__` 里 `kwargs.setdefault("stop_preempted_request", False)`（`entrypoints/llm.py:68`）。
- 等级：A
- 意味着什么：在线服务默认"抢占即杀"（避免重算风暴、保证尾延迟可预测），离线默认"重算"（不浪费已生成 token）；SO 请求因为状态机不能回滚，被强制杀。vLLM 的 `SWAPPED` 语义在这里彻底消失。
- 疑点：`stop_preempted_request` 的默认值在 `SchedulerConfig` 里是 `False`（`config.py:191`），只有经 `EngineArgs` 才是 `True`；两条构造路径默认值不一致，容易踩坑。

### #20 Sampler：temperature/top-p/top-k/log-softmax/采样全部进 TileLang kernel，per-seq uint64 seed
- 证据：`deepinfer/model_executor/layers/sampler.py:114-130` `forward(mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata, nextn_kv_caches, seq_metadata_list)`；采样数学在 `sampler_utils.py:113-127 process_sampling()` → `tilelang_ops/fused_sample.py:18-28 fused_sample(...)`，参数含 `temperatures/top_ps/top_ks/seq_seeds(uint64)/sample_positions`。vLLM v0.2.7 `sampler.py:34 Sampler.forward(logits, sampling_metadata)` + 模块级 `_apply_penalties/_apply_top_p_top_k/_apply_min_p/_greedy_sample/_random_sample/_beam_search_sample/_multinomial/_sample/_get_logprobs/_build_sampler_output`（14 个函数）。
- 等级：A
- 意味着什么：vLLM 的"Python 逐请求循环 + torch 算子"被换成"整批一个 kernel"，这是 DeepInfer 能做到 batch-invariant 和逐序列确定性的前提；代价是 penalties/min_p/beam/logits_processors 这些通用能力全部消失。
- 疑点：`Sampler.forward` 的 docstring 还写着"应用 logits processors、temperature、top-k/top-p/min-p"（`sampler.py:65`），但 `min_p`/logits processors 在代码里不存在——又一处"文档滞后于实现"。

### #21 ModelRunner：`_prepare_prompt/_prepare_decode/_prepare_sample` 删除，改 C++ `prepare_prompt_v4/prepare_decode_v4`
- 证据：`deepinfer/model_executor/model_metadata_utils.py:146-181` `prepare_inputmetadata_dict()` → `PrepareArgs()` → `prepare_func(args)`，`prepare_func` 来自 `deepinfer._C.engine` 的 `prepare_prompt_v4/prepare_decode_v4`（`model_metadata_utils.py:6-10`）；C++ 实现在 `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`（655 行）+ `prepare_tensor_input_base.cpp`（251 行）。vLLM v0.2.7 `model_runner.py:74 _prepare_prompt`、`:151 _prepare_decode`、`:256 _prepare_sample`、`:326 prepare_input_tensors`。
- 等级：A
- 意味着什么：vLLM 里最"模型相关"的 300 行 Python 元数据构造，被换成了 V4.1 专用（MLA/SWA/压缩 KV/Engram/CP-mirror）的 C++ 张量布局代码；Python 只剩 `InputMetadata` 这个可广播的 dataclass（`input_metadata.py:26`，含 `to_dict/from_flatten_dict/broadcast/recursive_fn`，这些在 vLLM 各版本都不存在）。
- 疑点：`prepare_prompt_v4` 的 "v4" 指模型代次（V4.1）还是布局版本，代码未注释（E：更像模型代次，因为 DFlash/YoCo 都在同一命名空间下）。

### #22 `execute_model` 一分为三，是为了让 post-process 能插进 forward 与 sample 之间
- 证据：`deepinfer/worker/model_runner.py:308 execute_main_model`（"IMPORTANT: This function must NOT contain any cuda synchronize."）、`:337 execute_sample`（同样禁 sync）、`:370 post_process`；`worker.py:122` forward → `:144`（`request_outputs = self._post_process_step(self.prev_step_state)`）→ `:154` sample。vLLM v0.2.7 是一个 `execute_model` 里 forward 完直接 `self.model.sample(...)`（`v0.2.7:model_runner.py:440-465`）。
- 等级：A
- 意味着什么：只有拆开才能把"上一步的 post-process"塞进"本步 forward 之后、sample 之前"这段 GPU 忙、CPU 闲的窗口；这是 ZOS 在代码级的落点，比"异步调度"这个名次更具体。
- 疑点：`forbid_cuda_sync()` 上下文（`model_runner.py:324,359`）会在同步时抛错，说明这条流水线对任何隐式 sync 都零容忍；书里讲"零开销"时应给出这个约束。

### #23 ZOS 的机制与 SGLang `event_loop_overlap` / NanoFlow 的异步调度同构，文档自己承认了来源
- 证据（DeepInfer）：`doc/inference_deployment_qa.md:274`
  > "Following existing implementations (SGLang, NanoFlow), the pipeline is overlapped by scheduling asynchronously between GPU and CPU: while the GPU runs `forward(N) → sample(N)`, the CPU concurrently runs `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`"
  代码：`worker.py:143-187` 的 `prev_step_state` + `preferred` 分支；`llm_engine.py:67-127 DeferredRelease`；`llm_engine.py:984-1015 post_process()` 由 worker 回调（`worker.py:271` `engine_callbacks.post_process_fn(step_state)`）。
  证据（SGLang）：`python/sglang/srt/managers/scheduler.py:2020-2030` `def event_loop_overlap(self)` 用 `result_queue: Deque[Tuple[ScheduleBatch, GenerationBatchResult]]` + `pop_and_process()` 处理上一批。
- 等级：A（代码/文档）+ B（Q&A 自述）
- 意味着什么：DeepInfer 借的是"上一批的结果在本批执行时处理"这个**批级流水**范式，不是 SGLang 的 RadixAttention，也不是 NanoFlow 的 nano-batch/SM 切分（代码里 `grep -i "SM 分区\|nano"` 零命中）。书里不要把三者混为一谈。
- 疑点：NanoFlow 是否指 arXiv 2408.12757 那篇（UW，Kan Zhu 等）无法从仓库确认；但该论文确实包含"asynchronous CPU scheduling"这一条（其摘要的主线是 intra-device 的 compute/memory/network 操作重叠，即 nano-batch），与 Q&A 只引用的"GPU 执行时 CPU 排下一批"那一半对应（E，高置信）。

### #24 SGLang 的 RadixAttention **没有**被借鉴：DeepInfer 只有"显式 fork 分支"的块共享
- 证据：`csrc/engine/scheduler.hpp:373-451` 的 `prefix_caching_hit` 只在两种情况下为真：(a) `seq->cache.block_table.empty()`（新请求，首次分配）；(b) `seq->schedule.branch_id` 以 `_` 开头（从父请求 fork 出来的分支），走 `_clone_state(*seq)` + `clone_blocks` 复制尾巴。`grep -rni "radix\|prefix_cach" csrc deepinfer` 只命中这几处 `prefix_caching_hit` 局部变量，没有哈希表、没有 radix tree、没有 `enable_prefix_caching` 开关。
- 等级：A
- 意味着什么：DeepInfer 不做"自动前缀缓存"，跨请求的 KV 复用交给外部 KV cache 系统（README:38-41 明说"A KV cache system"不在发布范围）；引擎内只保证"同一请求多分支/多轮"的块共享。这是与 SGLang/vLLM V1 最大的能力差异之一。
- 疑点：`branch_id` 由 `Scheduler.fork_seq(seq, blocks, target_branch_id, ...)` 传入（`scheduler.hpp:1061-1096`，`pybind_scheduler.cpp:123` 导出），而仓库里唯一的调用者是 `examples/benchmark.py:337`；生产的 fork 调用方（多轮/PD）不在开源范围内，`_` 前缀约定也无注释（E）。

### #25 多模型注册表：机制保留、条目从 27 个砍到 1 个
- 证据：`deepinfer/model_executor/models/__init__.py:15-16`
  > `_MODELS = { "DeepseekV41ForCausalLM": ("deepseek_vl", "DeepseekVLForCausalLM"), }`
  `ModelRegistry.load_model_cls()`/`get_supported_archs()` 与 vLLM v0.2.7 同名（`v0.2.7:models/__init__.py:13-40`，27 个条目，覆盖 Llama/Mistral/Mixtral/Qwen/Baichuan/ChatGLM/Falcon/MPT/OPT/Phi/Yi…）。
- 等级：A
- 意味着什么：这是"删内容不删机制"的典型——`ModelRegistry` 这个插件点还在，只是只剩一个模型；想加模型的人知道该改哪一行。
- 疑点：`_MODELS` 的值指向 `deepseek_vl`，但仓库里还有 `models/deepseek.py`；两者关系（`deepseek.py` 是基类？）未查。

### #26 OpenAI API Server / AsyncLLMEngine：整块缺席，更像"发布裁剪"而不是"代码删除"
- 证据：`deepinfer/entrypoints/` 只有 `__init__.py`（0 字节）与 `llm.py`（198 行）；全仓 `grep -rn "api_server\|AsyncLLMEngine\|OpenAIServing" --include="*.py" deepinfer` 零命中。vLLM v0.2.7 有 `vllm/entrypoints/openai/api_server.py`、`vllm/engine/async_llm_engine.py`。`README.md:3` "This open-source release provides offline inference"；`doc/inference_deployment_qa.md:21` "The KV cache management and load balancing components ... are tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced."
- 等级：B（官方自述）+ E（"内部仍有"的推断）
- 意味着什么：读这本书的读者如果拿 DeepInfer 当"可部署的 OpenAI 服务"会失望；它的定位是"离线批推理 + 可作为在线服务的引擎内核"，HTTP/协议层由 DeepSeek Recipe（另一个仓库）承担。
- 疑点：`entrypoints/llm.py` 保留了 `LLM` 这个离线入口却没有任何 `AsyncLLM`，说明发布时是按"能不能离线跑通"这条线切的，不是按"vLLM 有什么"切的。

### #27 LoRA / 量化后端 / TP：在派生文件里是"删字段"，不是"从来没有"
- 证据：vLLM v0.2.7 `Worker.add_lora/remove_lora/list_loras`（`worker.py:227-234`）、`ModelConfig(quantization=...)` + `_verify_quantization`（`config.py:73,146-160`）、`ParallelConfig(tensor_parallel_size, pipeline_parallel_size, worker_use_ray)`（`config.py:319-345`）；DeepInfer 对应位置：`worker/base.py` 无任何 LoRA 方法、`config.py:43-58` 的 `ModelConfig.__init__` 无 `quantization` 且 `:75` 硬编码 `self.dtype = torch.bfloat16`、`config.py:150-182` 的 `ParallelConfig` 只有 `data_parallel_size/context_parallel_size/expert_parallel_size/engram_parallel_size`，全仓 `grep -rn "tensor_parallel"` 零命中。
- 等级：A
- 意味着什么：`ParallelConfig` 是**重写**过的（字段全换），不是"没实现 TP 所以空着"；同理 `ModelConfig` 保留了 vLLM 的参数顺序（`trust_remote_code → load_format → seed`，见 `config.py:43-58`）却删掉了 `dtype/quantization`——这是"在 vLLM 骨架上做减法"的化石证据。
- 疑点：`doc/inference_deployment_qa.md:53` 说 "TP and CP are not used for now"，但 `README.md:12` 写 "Attention DP / balanced CP"、代码里 `context_parallel_size` 是实打实的一等公民；两处口径不一致（见 §4 问题 2）。

### #28 DeepInfer 同时在做"反向现代化"：batch-invariant、per-seq seed、xgrammar、MTP 都不是老 vLLM 的东西
- 证据：`deepinfer/environ.py:197 DEEPINFER_BATCH_INVARIANT = EnvVar.bool()`；`tests/models/test_batch_invariant.py:6` "Verifies that with DEEPINFER_BATCH_INVARIANT=1, running generate on the same prompt batched and singly..."；`sampling_params.py:61 seq_seed` + `fused_sample.py:69 seq_seeds`（uint64/seq）；`structure_output/matcher.py:14 from xgrammar.matcher import _FULL_MASK, bitmask_dtype, get_bitmask_shape`。
- 等级：A
- 意味着什么：不能把 DeepInfer 说成"2023 年的 vLLM"；它是"2023 年的骨架 + 2025–2026 年的内核/特性"。书的叙事应该是"长期维护的内部分支"，而不是"旧版 vLLM 魔改"。
- 疑点：batch-invariant 在 vLLM 里也有对应概念（V1 的 `VLLM_BATCH_INVARIANT`），但 DeepInfer 的实现是 kernel 级（`attn.py:666,1002` 传 `batch_invariant=`），不是 vLLM 的 Python 级；谁先谁后无法从本地仓库判断。

### #29 23 个文件的"派生程度"分布：4 个近原样、4 个中度、13 个只剩骨架、2 个后同步
- 证据：见 §5 表 A（逐文件相似度与判定）。
- 等级：A（数字）+ E（分类阈值：≥0.35 近原样、0.20–0.35 中度、<0.20 重写）
- 意味着什么：如果书里要画"vLLM 血统图"，不能画成均匀的 23 条线；应该画成"4 条粗线（直接复用）+ 2 条虚线（后同步）+ 17 条细线（只剩 API）"。
- 疑点：阈值是人为定的，换阈值分类会变；但"四个核心文件都 <0.15"这个结论对阈值不敏感。

## 3. 必答问题逐条回答

### Q1 DeepInfer 的六个核心文件最像 vLLM 哪个历史时期？与 V1 的区别？

**结论：最像 vLLM V0 的早期，即 v0.2.0–v0.2.7（2023-09-28 ~ 2024-01-03）的架构；不是 V0 后期（v0.6–v0.7），更不是 V1。**

判断依据（四条独立指纹，任一单独成立都指向 v0.2.x，合起来排除其他年代）：

| 指纹 | DeepInfer | vLLM 只在哪个版本成立 | 事实条目 |
| :--- | :--- | :--- | :--- |
| `EngineArgs.create_engine_configs()` 返回 4 元组 | `arg_utils.py:62-64` | v0.2.0–v0.2.7（v0.3.2 起 6 元组，v0.4 改名） | #3 |
| `profile_num_available_blocks` / `init_cache_engine` / `warm_up_model` | `worker/base.py:212,270,274` | v0.2.x–v0.3.x（v0.4 改 `determine_num_available_blocks`/`initialize_cache`） | #4 |
| `InputMetadata` 在 `model_executor/` | `model_executor/input_metadata/input_metadata.py` | ≤ v0.3.2（v0.4 并入 `worker/model_runner.py`） | #5 |
| `SamplingParams` 无 `min_tokens/logits_processors/truncate_prompt_tokens` | `sampling_params.py:47-62` | v0.2.x 的子集 | #6 |

补充证据：`LLMEngine.__init__` 的装配顺序（#8）、`_init_cache` 的五步（#9）、C++ `Scheduler` 的方法名（#13）、`SequenceData` 的 API（#15）。相似度扫描也同向：21 个骨架文件的平均相似度在 v0.2.7 见顶（0.174），之后单调下降（#2）；逐文件的最高分也全部落在 v0.2.0–v0.4.3 区间。

**不能确定具体 commit**：仓库只有 4 个提交（R03 Q4），没有 fork 记录；只能说到"v0.2.x 的 API 形态"。**例外**：`distributed/shm_broadcast.py` 来自 v0.6.3+（#7），说明血统是分层的，不是单点快照。

**与当前 V1 的区别（架构级，不是版本号级）：**

| 维度 | DeepInfer（V0-early 骨架） | vLLM V1（本地 main） |
| :--- | :--- | :--- |
| 调度器位置/语言 | C++ `Scheduler`（`csrc/engine/scheduler.hpp`，pybind11 暴露），进程内 | Python `vllm/v1/core/sched/scheduler.py`（3,602 行），跑在独立 EngineCore 进程 |
| 请求对象 | C++ `Sequence` + `SequenceMetadata`（对象池），无 `SequenceGroup` | `vllm/v1/request.py:60 Request` dataclass，无 `Sequence` |
| 进程模型 | 单引擎进程 + CP follower 进程（`run_workers` 广播）；无 API server | API server（可多实例，ZMQ）→ EngineCore（每 DP rank 一个）→ 每卡一个 worker |
| KV cache | 单块 uint8 `[num_gpu_blocks, block_shape]` + 独立 state cache；无 CPU swap；无自动前缀缓存 | `KVCacheManager` + `KVCacheBlock` 池 + 哈希前缀缓存 + `KVConnector` 外部存储；无 swap |
| 元数据构造 | C++ `prepare_prompt_v4/prepare_decode_v4` | Python `GPUModelRunner._prepare_inputs`（`gpu_model_runner.py:1924`）+ `InputBatch` |
| 采样 | TileLang `fused_sample`，per-seq uint64 seed，MTP 验证内嵌 | Python `vllm/v1/sample/sampler.py:78 Sampler` + `SamplingMetadata`，rejection sampler 单独模块 |
| 异步 | ZOS：C++ 调度 + worker 内 prev-step post-process；`DeferredRelease` 管生命周期 | `AsyncScheduler` 预填 `-1` 占位 token，批级流水（2 个 batch 在飞） |
| 入口 | 只有 `LLM`（离线），`step()` 返回句柄数组 | `LLM`/`AsyncLLM`/OpenAI server 全套 |

**一句话**：DeepInfer 像"2023 年底的 vLLM V0，被 DeepSeek 把调度器/元数据/采样三块挖出来换成 C++/kernel，再补上 2025 年的确定性特性"；V1 则是"把 V0 的 Python 调度器重写成 EngineCore 多进程架构，并加了哈希前缀缓存"。

### Q2 四个核心派生文件的代码级详细对照

#### Q2-a `deepinfer/engine/llm_engine.py`（1,029 行 vs vLLM v0.2.7 802 行）

**保留的骨架（A）：**
- 类名 `LLMEngine`、`from_engine_args()`、`add_request/abort_request/get_num_unfinished_requests/has_unfinished_requests`、`_process_model_outputs`、`step` 全部同名（`:461,472,633,658,665,687,860`）。
- `__init__` 的装配顺序逐行保留：tokenizer（200-205）→ `seq_counter`（222）→ `_init_workers`（233）→ `load_model`（241）→ `_init_cache`（244）→ `Scheduler`（267）（#8）。
- `_init_cache` 的五步（profile → 校验 → init_cache_engine → warm_up）与 vLLM 同构（#9）。
- `add_request` 的 docstring 仍逐字保留；`step()` 的"一次迭代"语义保留，但 docstring 已改写为 "Run one inference step and return the generated outputs"（`:861-871`）。

**改动的（A）：**
- `step()`：vLLM 是 `schedule() → _run_workers("execute_model", blocks_to_swap_in/out/copy) → _process_model_outputs()` 三步同步；DeepInfer 是 `schedule(pool)` → `clone_and_zero_states_and_blocks_on_default_stream` → `post_schedule` → KV state 落盘 → 多模态 fetch → `_execute_model_async_step`（`:881-952`）。`_process_model_outputs` 不在 `step()` 里调用，而是通过 `engine_callbacks.post_process_fn` 由 worker 在 forward 与 sample 之间回调（`worker.py:271` → `llm_engine.py:984`）。
- 返回类型：`np.ndarray[np.uint64]` 句柄数组（#11），配 `RequestOutputPool`/`DeferredRelease`（`:67-127,230,272`）。
- `add_request`：只收 `prompt_token_ids: np.ndarray[np.int32]`，直接造 C++ `Sequence`（#10）；多了 `user_id/extra/enqueue/chunk_checkpoints` 与 SO 异步编译（`:601-630`）。
- `_process_model_outputs`：`SamplerOutput`（扁平批）→ `scheduler.process_outputs(...)`（#12）。

**彻底删除的（A）：**
- `_init_workers_ray` / `placement_group`（v0.2.7:126-254）→ 只剩 spawn 多进程（`:368-415`）。
- `_process_sequence_group_outputs` + `_check_beam_search_early_stopping`（v0.2.7:376-590）→ `SequenceGroup`/beam 整体消失。
- `_check_stop`（v0.2.7:737-766）→ C++ `scheduler.hpp:1035`，且 stop 字符串/stop_token_ids 分支被删（#14）。
- `_decode_sequence`（v0.2.7:717-736）→ 移到 `entrypoints/llm.py:190` 的最终 decode。
- `_log_system_stats`（v0.2.7:644-716）→ `engine/stats_logger.py` 的 `StatsLogger`（`:298-304`）。
- `_verify_args`、`add_lora/remove_lora/list_loras`、`check_health`、`get_model_config/get_decoding_config` 等在线服务 API。

#### Q2-b `deepinfer/worker/worker.py` + `worker/base.py`（283 + 357 行 vs vLLM v0.2.7 234 行）

**保留的骨架（A）：**
- `Worker` 类名、`self.model_runner`/`self.cache_engine`/`self.gpu_cache`/`self.device`/`self.local_rank` 字段布局（`base.py:76-100`）。
- 生命周期方法名与顺序：`init_model`（182）→ `load_model`（203）→ `profile_num_available_blocks`（212）→ `init_cache_engine`（270）→ `warm_up_model`（274）（#4）。
- `profile_num_available_blocks` 的算法骨架：`empty_cache → profile_run → mem_get_info → (total*util - peak)//block_size`（`base.py:216-266` vs v0.2.7:84-114）。

**改动的（A）：**
- 拆成 `WorkerBase`（资源/生命周期）+ `Worker`（单步），`Worker.__init__` 只加 `engine_callbacks/prev_step_state/nextn` 再 `super().__init__`（`worker.py:57-87`）。
- `execute_model(seq_group_metadata_list, blocks_to_swap_*)` → `execute_model_step(is_prompt, is_placeholder, seq_metadata_list, scheduler_outputs)`，四段式（forward curr → post prev → sample curr → post curr/defer）（#16、#22）。
- `profile_num_available_blocks` 从"返回 (gpu,cpu) 二元组、executor 取 min"改成"返回 int、worker 内 all_reduce 取均值、减 state_size*2、偶数对齐"（#9）。
- `warm_up_model` 增加 `profile_decode_perf()`（MTP varlen 吞吐曲线）与 `capture_dflash_model()`（`:274-288`）。
- `cache_swap` 删除，改为 `clone_and_zero_states_and_blocks_on_default_stream`（`:160-169`）。

**彻底删除的（A）：**
- `determine_num_available_blocks`/`initialize_cache` 这套 v0.4+ 命名（DeepInfer 停在旧名）。
- `cache_swap` / `swap_in/out` RPC（#17）。
- `add_lora/remove_lora/list_loras`（#27）。
- `start_worker_execution_loop`（v0.4.x）—— DeepInfer 是 `start_slave_worker_loop`（`base.py:129-133`），语义也变成"收广播、执行同名方法"。

#### Q2-c `deepinfer/worker/cache_engine.py`（978 行 vs vLLM v0.2.7 160 行）

**保留的骨架（A）：**
- 文件名、模块 docstring "CacheEngine class for managing the KV cache."（`:4`）。
- 类名 `CacheEngine`（`:851`）、`allocate_gpu_cache()`（`:918`）、`get_cache_block_size()`（`:969`）、`self.gpu_cache` 字段。
- `get_cache_block_size` 仍是"单 block 字节数"的静态方法，只是签名从 `(block_size, model_config, parallel_config)` 变成 `(block_size, cfg: AttrDict)`（`:969-977`）。

**改动的（A）：**
- 布局：vLLM 是 per-layer 的 `List[KVCache]`，每个 `KVCache` 是 `(key_cache, value_cache)` 两个 dtype 张量（v0.2.7:70-110）；DeepInfer 是**一整块 uint8** `[num_gpu_blocks, block_shape[0]]`（`_alloc_gpu_uint8` + `cuda_utils.alloc_tensor`，`:911-916,930`），再由 `GPUCacheInterface` 按 layer 切出 `ica_cache/ca_cache/swa_cache` 视图（`KVCache` dataclass，`:206-219`）。
- 新增 `StateBlock`（`:221-306`）：算 SWA/compression state 的字节布局、`state_size_to_disk`、32B 对齐、kvstore ring 重排。
- 新增 `KVCacheManager`（`:309-539`）：逐层 `process()` 累加 offset、`finalize()` 校验、把布局写回 `GPUCacheInterface` 的类字段。
- 新增 `GPUCacheInterface`（`:540-850`）：`_kvcache_layout()`、`_carve_state_views()`、`select_state_slice_to_store()`、`load_state_from_store_to_prefill_cache()`——这是外部 KV cache 系统的客户端。
- `CacheEngine.__init__` 签名：`(cache_config, cfg: AttrDict, max_num_seqs, allocate=True, num_gpu_blocks=None)`（`:862-889`），`allocate=False` 用于"只算布局不分配"的分析模式。

**彻底删除的（A）：**
- `allocate_cpu_cache()`、`_swap()`、`swap_in()`、`swap_out()`、`copy()`（v0.2.7:88-141）——CPU 侧 KV 与 swap 全没了。
- `get_key_block_shape()`/`get_value_block_shape()`（v0.2.7:53-68）——K/V 不再分开，MLA 的 KV 是压缩后的单一 latent。
- `cache_stream`/`events`（v0.2.7:44-51）——没有 swap 就不需要同步事件。
- 而 docstring 仍在说 "GPU and CPU KV caches ... swapping and copying"（#18）。

#### Q2-d `deepinfer/model_executor/layers/sampler.py`（683 行 vs vLLM v0.2.7 560 行）

**保留的骨架（A）：**
- `class Sampler(nn.Module)`、`__init__(vocab_size)`、`forward(...)`（`:59,96,114`）。
- "取最后一个 token 的 hidden state → 算 logits → 处理 → 采样 → 返回 SamplerOutput"的主流程（docstring `:61-80` 仍是 vLLM 的语言）。
- prompt logprobs 的计算与拆分：`compute_prompt_logprobs/split_prompt_logprobs`（`sampler_utils.py`），对应 vLLM `_get_logprobs`。

**改动的（A）：**
- 输入从 `(logits, sampling_metadata)` 变成 `(mtp_module, embedding, hidden_states, sampler_inputs, main_input_metadata, nextn_kv_caches, seq_metadata_list)`（`:114-123`）——采样器现在自己算 logits（`get_logits` 走 `bf16_gemm_nt`），并负责 MTP draft 的 forward/验证。
- 采样数学：`process_sampling()`（`sampler_utils.py:113`）→ `fused_sample()`（`fused_sample.py:18`），temperature/top-p/top-k/log-softmax/采样/拒绝采样全在一个 TileLang kernel；RNG 是 per-seq `uint64 seq_seed` + `sample_positions`，`tag="mtp"` 时偏移 seed 形成独立流（`fused_sample.py:39-40,224`）。
- 新增 `_verification()`（`:411-596`）：按 `mtp_first_token_positions/mtp_seqlens_q` 把 `[total_q]` 行映射到 `(req, step)`，逐 draft 位置算 accept/reject，`get_accept_result()` 汇总成 accepted prefix；这是 vLLM V0 完全没有的。
- 新增 `_get_so_mask()`（`:302-410`）：xgrammar bitmask 按 step 应用（`structure_output/mask.py:33 apply_token_bitmask_by_step_inplace`）。
- `SamplerOutput` 变成了"扁平批 + 后处理闭包"：`post_funcs`（`sequence.py:177`）在 `ModelRunner.post_process` 里统一执行（`model_runner.py:376-378`）。

**彻底删除的（A）：**
- `_apply_penalties`（presence/frequency/repetition）、`_apply_min_p`、`_apply_logits_processors`、`_beam_search_sample`、`_build_sampler_output`、`_prune_hidden_states`、`_get_bin_counts_and_mask`（v0.2.7 sampler.py 的 14 个模块级函数里删掉 7 个）。
- `SamplingMetadata`/`SamplingTensors`（vLLM 的采样批次描述）→ 换成 `SamplerInputs/DecodeSamplingInputs/MTPDecodeInput`（`model_executor/sampling_inputs.py`）。
- Python 侧 `torch.multinomial`/`torch.topk` 逐请求循环 → 全部 kernel 化。
- `n`/`best_of`/beam 相关的一切（`SamplerOutput` 里没有 `SequenceGroupOutput` 列表）。

### Q3 从 SGLang / TensorRT-LLM / NanoFlow 各借鉴了什么？

**SGLang（有代码+文档双证）：**
1. **异步/重叠调度（ZOS）**。Q&A 原话 "Following existing implementations (SGLang, NanoFlow), the pipeline is overlapped by scheduling asynchronously between GPU and CPU"（`doc/inference_deployment_qa.md:274`）。代码落点：`worker.py:143-144` 的 prev-step post-process、`llm_engine.py:67-127` 的 `DeferredRelease`、`:984-1015` 的 `post_process()` 回调。SGLang 对应实现：`scheduler.py:2020-2030 event_loop_overlap` 的 `result_queue` + `pop_and_process()`。**这是"批级流水"，不是 NanoFlow 的 nano-batch。**
2. **结构化输出的 bitmask 路线**（弱证据）。用 xgrammar 的 `fill_next_token_bitmask` + 按 step 应用（`structure_output/matcher.py:14,193`；`structure_output/mask.py:33`），与 SGLang 的 `compressed_fsm` 同属"编译成自动机 + 每步 bitmask"这一族；但 xgrammar 是三方库，不能算 SGLang 专有（E）。
3. **没有借 RadixAttention**（反证）：全仓无 radix tree/哈希前缀缓存，`prefix_caching_hit` 只服务显式 fork（#24）。

**NanoFlow（只有文档证；指向公开论文而非内部系统，E）：**
1. Q&A 把它与 SGLang 并列，作为"CPU 调度与 GPU 执行重叠"的 prior art（`:274`）。
2. 对照 NanoFlow 论文（arXiv 2408.12757）：摘要主线是"LLM serving 是 compute-bound，但 compute/memory/network 三类操作在 device 内被串行执行"，解法是 nano-batch + 操作级重叠；其中"asynchronous CPU scheduling"——CPU 在 GPU 执行 iteration i 时准备 iteration i+1 的 batch——与 DeepInfer 的 ZOS 对应（E，高置信）。DeepInfer 只取了这一半。
3. **没有借**的是 NanoFlow 的另一半：nano-batch 切分 + SM 分区 + 操作级 co-scheduling。DeepInfer 代码里没有任何 SM 分区/多 stream 并行的调度器（`grep -i "sm_partition\|nano"` 零命中），它的重叠粒度是"step 级"，不是"算子级"。

**TensorRT-LLM（只有 README 一句，代码零痕迹）：**
1. `README.md:95` "additional inspiration from ... TensorRT-LLM"——仅此一处（B）。
2. 全仓 `grep -rn "TensorRT\|TRT"` 零命中（A，负证据）。
3. 唯一可指认的结构回声（E）：`cache_engine.py:309 KVCacheManager` 这个类名与 TRT-LLM 的 `KVCacheManager` 同名，且同样采用"主池（GPU block）+ 二级存储（`state_size_to_disk`/`load_state_from_store_to_prefill_cache`）"的分层；但 DeepInfer 的 `KVCacheManager` 只管**布局切分**，不管 block 复用/驱逐，所以只是"形似"。
4. **结论**：书里应写"官方承认受 TensorRT-LLM 启发，但开源代码中找不到可指认的具体机制"；不要把 PagedAttention/连续批处理说成"借鉴 TRT-LLM"（那些是 vLLM 的）。

### Q4 缺失的通用能力：删的还是从来没有的？

| 能力 | 判定 | 证据 | 等级 |
| :--- | :--- | :--- | :--- |
| 多模型注册表 | **分叉后删内容，机制保留** | `_MODELS` 从 v0.2.7 的 27 条 → 1 条（`models/__init__.py:15-16`）；`ModelRegistry` 类原样 | A |
| OpenAI API Server | **大概率发布裁剪，不是代码删除** | `entrypoints/` 只剩 `llm.py`；README:3 "offline inference"；Q&A:21 "KV cache management and load balancing ... not yet fully open-sourced"（未提 API server，但同属服务层）| B+E |
| 多 LoRA | **分叉后删除** | v0.2.7 `worker.py:227-234` 有 `add_lora/remove_lora/list_loras`、`vllm/lora/` 整目录；DeepInfer 派生文件里方法消失、`grep -i lora` 只剩 MLA 的 `q_lora_rank` | A |
| 复杂量化后端 | **分叉后删除（但保留低精度 kernel）** | v0.2.7 `ModelConfig(quantization=...)` + `_verify_quantization`（awq/gptq/squeezellm）+ `layers/quantization/`；DeepInfer `config.py:75` 硬编码 `torch.bfloat16`，无 `quantization` 字段。FP8/FP4 只存在于 KV cache（`V41_FP4_FORMAT`）与固定 kernel，不是可插拔后端 | A |
| TP / PP 并行 | **分叉后删除并替换** | v0.2.7 `ParallelConfig(tensor_parallel_size, pipeline_parallel_size, worker_use_ray)`；DeepInfer `ParallelConfig` 换成 DP/CP/EP/Engram，`grep tensor_parallel` 零命中；Q&A:53 "TP and CP are not used for now"（B） | A |
| 分布式抢占/重算 | **一半删一半留，默认反转** | `_preempt_by_swap` 与 CPU cache 删除；`_preempt_by_recompute` 逻辑保留为 `reset_for_recompute()+push_front`（`scheduler.hpp:781-789`）；默认 `stop_preempted_request=True`（在线杀） / 离线 False（重算） | A |

**统一解释（E）**：DeepSeek 的分叉策略是"**只保留服务 DeepSeek 自家模型的单一路径**"——不是"vLLM 功能太多所以删"，而是"多模型/多量化/多并行/多 LoRA 这些分支在 DeepSeek 的生产里没有第二种用法，留着只会增加维护面"。凡是"每加一种就要改调度器/kernel"的通用性（模型注册表、量化后端、TP）都被砍掉；凡是"能外包给外部系统"的（API server、KV cache 系统、负载均衡、PD 分离的调度）都被挪出仓库。

## 4. 新发现的问题

1. **`SWAPPED` 状态是纯遗留**：`sequence.hpp:33` 仍定义 `SWAPPED`，`pybind_sequence.cpp:78` 仍导出，但全仓没有任何地方把 seq 置为 `SWAPPED`（只有 `FINISHED_PREEMPTED`）。书里若画状态机，应标注"该状态不可达"。
2. **Q&A 与 README 对 CP 的口径矛盾**：Q&A:53 说 "TP and CP are not used for now"，README:12 说 "Attention DP / balanced CP"，代码里 `context_parallel_size` 是一等公民（`config.py:150-182`、`worker/base.py:315-330`）。需要向作者确认：是"生产 V4.1 不用 CP"还是"文档没更新"。
3. **`sampler.py` 的 docstring 与实现不符**：docstring 写 "应用 logits processors、temperature、top-k/top-p/min-p"（`:65`），但 `min_p` 和 logits processors 在 `SamplingParams`/`fused_sample` 里都不存在。与 #18 的 CacheEngine docstring 一样，属于"文档停在分叉时"。
4. **`prepare_prompt_v4` 的 "v4" 语义未确认**：是模型代次（V4/V4.1）还是布局版本？影响书里对"元数据 C++ 化"的叙述粒度。
5. **`ModelRegistry` 的 `deepseek_vl` vs `models/deepseek.py`**：注册表指向 `deepseek_vl`，仓库里却有 `deepseek.py`；两者是"VL 包装 vs 纯文本基类"还是"新旧两代"，未查（可能影响第 3 章模型加载一节）。
6. **`branch_id` 的生产者只在 example 里**：C++ 用 `branch_id` 是否以 `_` 开头来决定 fork 共享，入口是 `Scheduler.fork_seq()`；仓库里唯一调用它的是 `examples/benchmark.py:337`，生产侧（多轮对话/PD 复用）的调用者不在开源范围内。
7. **vLLM 侧没有 fork 记录**：DeepInfer 仓库只有 4 个提交，`git log` 看不到"从 vLLM 导入"的那次提交；本笔记的年代判断全靠 API 指纹 + 相似度，无法给出"fork 自 commit xxxx"的硬结论。

## 5. 可用于图表的材料

### 表 A：23 个派生文件逐文件判定（相似度对 13 个 vLLM tag 取最高）

相似度算法见 §0；行数为 `wc -l`；"最高分 tag"是 13 个 tag（v0.2.0/2.2/2.5/2.7/3.2/4.0/4.3/5.4/6.0/6.3/7.3/8.5/9.1）里得分最高的那个。**单文件最高分只有 0.01–0.03 的领先幅度，tag 精度不可当真**；它只能说明"落在哪个年代区间"，不能说明"fork 自哪个 tag"。

| # | 文件 | 行 | 最高分 tag | 相似度 | 判定 |
| ---: | :--- | ---: | :--- | ---: | :--- |
| 1 | `deepinfer/config.py` | 228 | v0.2.0 | 0.24 | 中度改写：参数顺序保留，dtype/量化/TP 字段删 |
| 2 | `deepinfer/distributed/shm_broadcast.py` | 461 | **v0.6.3** | **0.68** | 后同步：近原样搬入 |
| 3 | `deepinfer/engine/arg_utils.py` | 114 | v0.2.2 | 0.11 | 重写：只剩 4 元组骨架 + dataclass |
| 4 | `deepinfer/engine/llm_engine.py` | 1,029 | v0.3.2 | 0.16 | 重写：装配顺序保留，调度/输出全换 |
| 5 | `deepinfer/entrypoints/llm.py` | 198 | v0.2.5 | **0.49** | 近原样：`_run_engine` 逐行对应 |
| 6 | `deepinfer/logger.py` | 174 | v0.2.2–v0.2.7 | 0.22 | 中度：加 rank/进程 crash header |
| 7 | `deepinfer/model_executor/layers/layernorm.py` | 63 | v0.2.2 | **0.41** | 近原样：类保留，`ops` 换 TileLang `norm` |
| 8 | `deepinfer/model_executor/layers/rotary_embedding.py` | 284 | v0.2.2 | **0.40** | 近原样：加 YaRN/DeepSeek 缩放 |
| 9 | `deepinfer/model_executor/layers/sampler.py` | 683 | v0.2.7 | 0.04 | 重写：只剩类/forward 壳，数学全进 kernel |
| 10 | `deepinfer/model_executor/layers/vocab_parallel_embedding.py` | 30 | v0.2.2–v0.2.7 | 0.14 | 重写：只剩 30 行的薄封装 |
| 11 | `deepinfer/model_executor/model_loader.py` | 153 | v0.2.5–v0.2.7 | 0.15 | 重写：只支持 safetensors/dummy |
| 12 | `deepinfer/model_executor/models/__init__.py` | 37 | v0.2.5 | **0.38** | 近原样：注册表机制保留，27→1 条 |
| 13 | `deepinfer/model_executor/parallel_utils/communication_op.py` | 82 | v0.4.0 | 0.16 | 重写：改成 CP/EP 通信原语 |
| 14 | `deepinfer/model_executor/utils.py` | 101 | v0.2.0 | 0.23 | 中度：`Movable`/`reconstruct_param` 新增 |
| 15 | `deepinfer/sampling_params.py` | 119 | v0.2.0 | 0.27 | 中度：字段大删 + 5 个 DeepSeek 字段 |
| 16 | `deepinfer/sequence.py` | 281 | 噪声（各 tag 均 ≤0.03） | 0.03 | 重写：只剩 4 个类，`Sequence` 进 C++ |
| 17 | `deepinfer/utils.py` | 282 | v0.2.7 | 0.13 | 重写：NUMA/绑核/timer 等生产工具 |
| 18 | `deepinfer/worker/base.py` | 357 | v0.4.0 | 0.16 | 重写：方法名保留，内容全换 |
| 19 | `deepinfer/worker/cache_engine.py` | 978 | v0.3.2 | 0.05 | 重写：前 800 行全新，`CacheEngine` 只剩 128 行 |
| 20 | `deepinfer/worker/model_runner.py` | 644 | v0.3.2 | 0.07 | 重写：`_prepare_*` 进 C++，execute 拆三段 |
| 21 | `deepinfer/worker/worker.py` | 283 | v0.4.3 | 0.09 | 重写：`execute_model_step` 四段式 |
| 22 | `setup.py` | 91 | v0.2.0 | 0.09 | 重写：CUDA/NPU 双扩展 |
| 23 | `tests/distributed/test_shm_broadcast.py` | 145 | **v0.8.5** | **0.55** | 后同步：随 shm_broadcast 一起搬 |

**统计**：近原样（≥0.35）4 个；后同步（换 tag）2 个；中度（0.20–0.35）4 个；重写（<0.20）13 个。合计 6,817 行。

**读法**：除 #2/#23 两条"后同步"支线外，其余 21 个文件的最高分 tag 都落在 **v0.2.0–v0.4.0** 区间；再往后的版本一路下降（均值 0.174 → 0.023）。这与 §2 #3–#6 的 API 指纹一致：**骨架来自 v0.2.x，少量命名/结构可能被 v0.3–v0.4 时期的改动影响**（例如 `WorkerBase` 这个名字 v0.4 才有，但方法体是 v0.2.x 的）。

### 表 B：四代架构对照（可用于书里的"血统"图）

| 维度 | vLLM v0.2.7（2024-01） | vLLM v0.6.3（2024-10，V0 后期） | vLLM V1（本地 main） | DeepInfer（2026-10） |
| :--- | :--- | :--- | :--- | :--- |
| 调度 | Python `Scheduler`（`core/scheduler.py`） | Python `Scheduler` + 多步/异步 | Python `EngineCore` + `Scheduler` | **C++ `Scheduler`**（pybind11） |
| 请求 | `Sequence`/`SequenceGroup` | `SequenceGroup` + `SequenceGroupMetadata` | `Request` dataclass | C++ `Sequence` + `SequenceMetadata`（对象池） |
| KV cache | per-layer KV 张量 + CPU swap | 同左 + 前缀缓存（v0.4+） | `KVCacheManager` + 哈希前缀缓存 + Connector | 单块 uint8 + state cache + 外部 KV store 客户端（无自动前缀缓存） |
| Worker | `Worker`（`init_model/profile/init_cache_engine`） | `WorkerBase` + `Worker` + Executor | `EngineCore` + `GPUWorker` + `GPUModelRunner` | `WorkerBase`（资源）+ `Worker`（step）+ CP follower |
| 采样 | Python `Sampler` + `_sample` | Python `Sampler` + 多种 logits 处理 | `vllm/v1/sample/sampler.py` + rejection sampler | TileLang `fused_sample` + MTP 验证 |
| 入口 | `LLM` + `AsyncLLMEngine` + OpenAI server | 同左 | `LLM`/`AsyncLLM`/server | 只有 `LLM`（离线） |
| 并行 | TP/PP（Ray） | TP/PP/EP | TP/PP/EP/DP | DP/CP/EP/Engram（**无 TP/PP**） |

### 表 C：vLLM 版本时间线（用于标注"分叉窗口"）

| tag | 提交日期 | 与本笔记相关的里程碑 |
| :--- | :--- | :--- |
| v0.2.0 | 2023-09-28 | 4 元组 `create_engine_configs`、`InputMetadata` 在 `model_executor/` |
| v0.2.7 | 2024-01-03 | 同样的 4 元组 + `profile_num_available_blocks`；**DeepInfer 骨架最像这里** |
| v0.3.2 | 2024-02-21 | `create_engine_configs` → 6 元组；`Worker` 仍用旧名 |
| v0.4.0 | 2024-03-30 | `create_engine_config`（单数）+ `EngineConfig`；`WorkerBase`/`determine_num_available_blocks` |
| v0.4.3 | 2024-05-31 | `InputMetadata` 并入 `worker/model_runner.py`；chunked prefill 出现 |
| v0.6.3 | 2024-10-14 | `distributed/device_communicators/shm_broadcast.py`；**DeepInfer 的 shm_broadcast 来自这里** |
| v0.8.5 | 2025-04-28 | `VLLM_USE_V1` 默认 True（`vllm/envs.py:76,530-531`），V1 成为默认引擎；本地 main 已删 V0 |
| main | 2026-10-22 | V1 + MRV2；`vllm/engine/llm_engine.py` 只是别名 |

### 图 D：一张"壳与芯"示意（文字版，可直接画）

```
vLLM v0.2.x 文件                       DeepInfer 文件
─────────────────────────             ─────────────────────────────
LLMEngine (Python)          ──壳──▶   LLMEngine (Python, 装配 + step 驱动)
  └ Scheduler (Python)      ──芯──▶     Scheduler (C++/pybind11)      ← 逐方法翻译
  └ Sequence (Python)       ──芯──▶     Sequence (C++ struct)
SequenceGroup               ──删──▶   （不存在；用 branch_id fork）
Worker.execute_model        ──拆──▶   execute_main_model / execute_sample / post_process
CacheEngine (per-layer KV)  ──芯──▶     CacheEngine (uint8 block) + KVCacheManager + GPUCacheInterface
Sampler (torch)             ──芯──▶     Sampler (Python 壳) + fused_sample (TileLang)
ModelRunner._prepare_prompt ──芯──▶     prepare_prompt_v4 (C++, 655 行)
SamplingParams (20+ 字段)   ──删──▶   8 个字段 + 5 个 DeepSeek 字段
```

### 图 E：ZOS 时序（可直接画成两条泳道）

```
GPU:   forward(N) ──────────▶ sample(N) ──────────▶ forward(N+1) ─────▶ sample(N+1)
CPU:   schedule(N+1) ─▶ launch_forward(N+1) ─▶ post-process(N-1) ─▶ launch_sample(N+1)
                    ↑ worker.py:143 的 _post_process_step(prev_step_state)
                    ↑ worker.py:271 engine_callbacks.post_process_fn
```

来源：`worker.py:113-187` 的四段式 + `llm_engine.py:984-1015` 的 `post_process()`；Q&A 的文字描述在 `doc/inference_deployment_qa.md:274`。
