# R11 批不变、确定性采样、结构化输出、测试与评测

## 0. 元信息

- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21，本地 `research/sources/DeepInfer`）
- 仓库历史只有 `d585e65 initial commit` 与本次 submodule bump，**没有开发过程提交**；因此所有"为什么这么改"的判断都只能从代码与注释推断，没有 commit message 佐证。
- 查阅的主要来源：
  - `deepinfer/environ.py`、`deepinfer/worker/runtime_config.py`、`README.md:16-28`、`doc/inference_deployment_qa.md:276-288`
  - `deepinfer/model_executor/layers/{sampler.py,sampler_utils.py,attention_backend.py,linear.py}`
  - `deepinfer/model_executor/layers/tilelang_ops/{fused_sample.py,random_sampling.py,structure_output.py,schedule_mtp_varlen.py}`
  - `deepinfer/model_executor/layers/tilelang_ops/cuda/{fused_sample.py,random_sampling.py}`
  - `deepinfer/model_executor/layers/tilelang_ops/asc/{rstd_1ulp.py,random_sampling.py,fused_sample.py,norm.py,fused_transform.py}`
  - `deepinfer/model_executor/models/v4_modules/attn.py`、`deepinfer/model_executor/models/mtp/dflash.py`、`deepinfer/worker/worker.py`
  - `deepinfer/structure_output/{compiler.py,mask.py,matcher.py,so_matcher_builder.py}`、`examples/_structured_output.py`、`examples/offline_inference.py`、`examples/_prompts.py`
  - `vendor/FlashMLA/{flash_mla/flash_mla_interface.py,csrc/api/sparse_decode.cpp}`、`vendor/DeepGEMM/csrc/{apis/*.hpp,jit_kernels/heuristics/*.hpp}`
  - `tests/models/{test_batch_invariant.py,test_kl.py,test_sampler.py,test_offline_inference.py,offline_decode_ppl.json}`、`deepinfer/testing.py`
  - `eval/README.md`、`eval/run_eval.py`、`eval/score.py`、`eval/common/passk.py`、`eval/common/judge.py`、`eval/{aime,gpqa,hle}/bench.py`
- 没看到的东西（不要写成"有"）：`tile_kernels` 是外部依赖（`vendor/` 里只有 DeepGEMM/FlashMLA/DeepEP/DeepSelect/DeepGEMM-Ascend/DeepEP-Ascend），`tile_kernels.use_deterministic_algorithms` 的实现不在本仓库，**未能确认它具体关掉了什么**。

## 1. 三个最重要的发现

1. 批不变没有独立的"算法"，它是一串**"把批维度从归约形状里摘出去"**的开关：FlashMLA 关 split-KV、DeepGEMM deterministic 模式关 split-K 与 cuBLASLt、采样器改成"一行一个 CTA + 定点整数 CDF"；三者缺一，同一行文本在不同批组成下就会算出不同的 logits 或不同的 token。
2. 确定性采样的真正机制不是"固定随机种子"，而是 **RNG 的下标是绝对位置**：随机流由 `(seq_seed, 3*sample_positions + case)` 决定，`sample_positions = seqlen_k`，所以重放、抢占、teacher forcing 都能复现；而**变长 MTP 会按全局排名分配验证长度**（`extra_budget = total_q - num_seqs`），使单请求的验证长度取决于同批其他请求，因此必须关闭。
3. 结构化输出后端是**上游 xgrammar**，DeepInfer 自研的是"调度与搬运"：编译放在独立线程池异步做（未编完不入调度）、掩码在 **CPU** 上生成成 pinned uint32 bitmask、投机解码时**预生成 k+1 份掩码**并在验证后用 `rollback` 回退、GPU 侧只用 TileLang kernel 做 in-place `-inf` 填充。

## 2. 事实条目

### #1 批不变只有一个 bool 环境变量，默认关闭
- 证据：`deepinfer/environ.py:197`
  > `DEEPINFER_BATCH_INVARIANT = EnvVar.bool()`
  `README.md:27`：`Enable with DEEPINFER_BATCH_INVARIANT=1 (NVIDIA only)`。
- 等级：A（env 定义）/ B（README 的能力声明）
- 意味着什么：读者要意识到"批不变"不是默认行为，生产默认走的是快路径；所有相关分支都挂在这一个 flag 上。
- 疑点：无。

### #2 开关真正落地只有四处：worker 初始化、attention、EP buffer、一处 prefill 融合分支
- 证据：全仓 `grep batch_invariant` 去掉 `tests/`、`vendor/` 后只剩
  - `deepinfer/worker/runtime_config.py:62-69`（配置 GEMM 库）
  - `deepinfer/model_executor/models/v4_modules/attn.py:89`、`:666`、`:1002`
  - `deepinfer/model_executor/parallel_utils/userbuffer.py:79`
  - `deepinfer/model_executor/layers/attention_backend.py:85,141`（透传 FlashMLA）
- 等级：A
- 意味着什么：影响面很小、很集中，适合在书里画一张"一个 flag → 四个子系统"的图。
- 疑点：无。

### #3 worker 初始化：NPU 直接断言不支持批不变；GPU 上同时打开两个 kernel 库的 deterministic 模式
- 证据：`deepinfer/worker/runtime_config.py:56-69`
  > `if PLATFORM.is_npu:` / `# TODO(NPU): 待完整支持 batch invariant 后移除` / `assert not envs.DEEPINFER_BATCH_INVARIANT`
  > `if envs.DEEPINFER_BATCH_INVARIANT or PLATFORM.is_npu:  # TODO(NPU): 防止 CI 数值不稳定`
  > `deep_gemm.use_deterministic_algorithms(True)` / `tile_kernels.use_deterministic_algorithms(True)`
- 等级：A
- 意味着什么：批不变是"进程级"设定，在建模型前生效；NPU 的数值稳定策略是"无条件 deterministic"，而批不变仍缺。
- 疑点：`tile_kernels` 的实现不在仓库内，无法确认它改了哪些 kernel（见 §0）。

### #4 FlashMLA：批不变 = 关掉 split-KV 解码
- 证据：`vendor/FlashMLA/csrc/api/sparse_decode.cpp:220-223`
  > `#ifdef FLASH_MLA_IS_BUILD_ON_ASCEND` / `bool enable_split_kv = false;   // Ascend does not support split-KV`
  > `#else` / `bool enable_split_kv = !enable_batch_invariant && !(topk + extra_topk <= 640);`
  `vendor/FlashMLA/flash_mla/flash_mla_interface.py:110-112`
  > `If True, the split-KV decoding path is disabled so that results do not depend on how the batch is partitioned`
- 等级：A
- 意味着什么：这是全书最直接的一句"批不变 = 归约不被批切分"。split-KV 会把一条 query 的 topk 归约切成若干段并行、再合并，段数由调度决定；关掉后每条 query 在自己的 CTA 里按固定顺序扫完 topk。
- 疑点：`enable_batch_invariant=True` 必须与 `tile_scheduler_metadata` 的首次创建一致（`vendor/FlashMLA/flash_mla/flash_mla_interface.py:171` 有断言），说明 sched meta 里缓存了这个选择；本书不必展开，但要知道它**不是**每次调用都能翻转。

### #5 DeepGEMM deterministic 模式三个具体作用点
- 证据：
  - 禁 cuBLASLt 分派：`vendor/DeepGEMM/csrc/apis/gemm.hpp:519-522`
    > `if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available() and epilogue_alpha.has_value()): cublaslt_gemm(...)`
  - mega-gate（MoE gate GEMM）禁 split-K：`vendor/DeepGEMM/csrc/jit_kernels/heuristics/mega_gate.hpp:110-111`
    > `// NOTES: deterministic mode is also batch invariant, so K is never split`
    同文件 `:126`：`for (int num_split_k = 1; num_split_k <= (deterministic ? 1 : kNumMaxSplitK); ...)`
  - mega_mhc 固定 split 数：`vendor/DeepGEMM/csrc/apis/mega_mhc.hpp:16-18`
    > `if (heuristics_runtime->get_deterministic_algorithms()) return mhc_layout::kDefaultNumSplits;`
- 等级：A
- 意味着什么：cuBLASLt 会**按 M 选不同 kernel**（同一行数据在 batch=1 与 batch=64 下可能走不同 tile），所以批不变必须绕开它；作者在注释里直接把"deterministic ⇒ batch invariant"写成了前提。
- 疑点：普通 `bf16_gemm_nt`（`deepinfer/model_executor/layers/linear.py:27-33` → `deep_gemm.bf16_gemm_nt`）走的是哪条 heuristic 分支，仓库内看不到它的 split-K 代码，**未能确认**它是否也完全无 split-K。

### #6 lm_head 的 logits 也走 DeepGEMM，所以 sampling 前的最后一步 GEMM 同样受 deterministic 保护
- 证据：`deepinfer/model_executor/layers/sampler_utils.py:20-28`
  > `logits = torch.empty(hidden_states.size(0), embedding.size(0), ...)` / `bf16_gemm_nt(hidden_states, embedding, logits)`
- 等级：A
- 意味着什么：批不变链条是"attention → MoE GEMM → lm_head GEMM → 采样"，四个环节都要钉住，缺一个就前功尽弃。
- 疑点：无。

### #7 DeepEP：开了 EPLB 重排时，批不变会关掉 `allow_multiple_reduction`
- 证据：`deepinfer/model_executor/parallel_utils/userbuffer.py:79`
  > `allow_multiple_reduction=not (envs.DEEPINFER_BATCH_INVARIANT and envs.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL > 0),`
- 等级：A（代码）/ E（原因）
- 意味着什么：多个归约目标写同一 buffer 时，累加顺序随 token→expert 映射变化，而映射随批变化；这是 EP 侧唯一的批不变处理。
- 疑点：DeepEP 自身还有 `deterministic=True` 的确定性 dispatch 路径（`vendor/DeepEP/deep_ep/buffers/ep.py:254,296,733-743`），但 DeepInfer **没有**把它接到 `DEEPINFER_BATCH_INVARIANT` 上；所以 EP 通信的确定性只覆盖了"多次归约"这一项（E）。

### #8 采样前向核：每行一个 CTA，归约形状只由 vocab 决定
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py:36-45, 177, 222`
  > `THREADS, LSE_BD, SAMPLE_BD = 256, 8192, 2048`（仅当 `vocab_size >= 8192`）
  > `with T.Kernel(total_q, threads=THREADS) as row:`（LSE pass）
  > `with T.Kernel(total_q, threads=THREADS) as pid:`（采样 / 验证 pass）
- 等级：A
- 意味着什么：grid 只随行数增长，`threads` 与 tile 只随词表大小变化，因此**单行的 max/sum 归约树与批大小无关**。这是"采样侧批不变"的结构性保证，比"少用浮点"更根本。
- 疑点：无。

### #9 LSE（log-sum-exp）刻意保留了一个特定 layout，"为了和参考 cross-entropy 逐位对齐"
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py:184-190`
  ```python
  if NUM_STAGES:
      # Preserve the original reduction order for bitwise
      # alignment with the reference cross-entropy implementation.
      T.annotate_layout(
          {lf0: T.Fragment((LSE_BD,), forward_fn=lambda i: ((i % (THREADS * 8)) // 8, i // (THREADS * 8) * 8 + i % 8))}
      )
  ```
  逐行：`NUM_STAGES` 只在 `vocab_size > LSE_BD`（大词表）时非 0；注释说明这里不是性能考虑而是**数值顺序**；`Fragment` 的 `forward_fn` 把逻辑下标映射到 (thread, register) 的固定布局，从而固定 8 元素步长的分段归约；随后 `:199-213` 用 `T.alloc_reducer(op="max")` / `+= exp(...)` 做块内归约，再 `sum_v = exp(max_v-new_max)*sum_v + blk_sum` 做跨块合并。
- 等级：A
- 意味着什么：连"哪 8 个元素落在同一个线程"都被钉死，说明作者的验收标准是**逐位复现训练侧 cross-entropy**，而不仅是"批无关"。
- 疑点：参考实现（训练侧 `hai-llm`）不在本仓库，无法核对。

### #10 top-p / top-k 用定点直方图找阈值，阈值本身不含浮点累加顺序
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py:85-140, 272-277`
  - 概率先量化：`SCALE = float(1 << SCALE_BITS)`（`SCALE_BITS = uint32.bits - 1 = 31`，`:62-64`），`T.atomic_add(hist[bucket], T.cast(T.exp(...) * SCALE, uint32))`（`:118-122`，按 float 的**可排序位模式**分桶）
  - 基数（radix）多轮定位：`RADIX_BITS = 8 if LSE_BD <= 1024 else 11`（`:52-54`），`ROUNDS = ceildiv(32, RADIX_BITS)`
  - 阈值还原成概率：`threshold_p = T.exp((threshold + lse1) * inv_temperature + lse0)`（`:275`）
  - 归一化：`lse0 -= T.log(mass)`（`:369`）
- 等级：A
- 意味着什么：top-p 的"截断到哪"由**整数直方图的累加**决定，而整型加法可交换、无舍入，所以阈值与线程数、批大小、归约顺序完全无关。这是把"浮点归约顺序敏感"从采样路径里整体移除的手法，值得在书里单独讲。
- 疑点：直方图用 `atomic_add` 累加 uint32，溢出风险由 `SCALE=2^31` 与 `total<=1` 的假设兜住（E）；`float_to_fixed` 是 floor 而非 round（`deepinfer/model_executor/layers/tilelang_ops/random_sampling.py:38-44`），所以定点 CDF 略偏小，量级 2^-31。

### #11 采样本身：两级定点 CDF + 拒绝采样取均匀数
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py:207-214`（docstring）、`:239-256`
  ```python
  with T.Kernel(num_tokens, threads=THREADS) as row:
      ...
      for chunk in T.unroll(N_CHUNKS, unroll_factor=4):
          T.copy(probs[row, chunk * BD], pf); accumulate(cfg, pf, acc_frag)   # uint32 累加
      seq = sample_positions[row] * 3 + sample_case
      offset = sample_offset(cfg, acc_frag, seq_seeds[row], seq, rt_s)
  ```
  逐行：一行一个 CTA；按 `chunk*BD` 折叠成 BD 条"车道和"，累加在 uint32 里做；随机流下标 = `位置*3 + case`；`sample_offset`（`:95-157`）`T.rng_init(seed, seq)` 后在 `[0,total)` 上取均匀数并二分定位车道，`:128-137` 是**取模拒绝**（`limit = (~0)//total*total`，循环重抽直到 `rnd < limit`），消除取模偏置。
- 等级：A
- 意味着什么：采样决策只做整数比较，因此"同一行 + 同一 seed + 同一位置"必然得到同一 token——这是确定性采样在 kernel 级的落点。
- 疑点：GPU 版用 u32 定点 + 32/64bit 均匀数；参考实现 `random_sampling_ref`（`deepinfer/model_executor/layers/tilelang_ops/random_sampling.py:70-84`）用 SplitMix64 哈希 + **fp64 CDF 反演**，两者只是统计等价、不逐位等价。

### #12 一行一个 CTA 的批不变性有自测：置换不变 + 单条等于批量
- 证据：`deepinfer/model_executor/layers/tilelang_ops/random_sampling.py:309-332`
  > `out_perm = random_sampling(probs[perm].contiguous(), seq_seeds[perm]..., positions[perm]...)` / `assert torch.equal(out_perm, ref[perm])`
  > `single = random_sampling(probs[i:i+1]...)` / `assert single.item() == ref[i].item()`
- 等级：A
- 意味着什么：这是"批不变"最小的可验证单元，比整模型对拍便宜得多；书里可以把它当作"如何证明一个 kernel 是批不变的"范例。
- 疑点：无。

### #13 `seq_seed` 是每请求的 uint64；缺省时随机取
- 证据：`deepinfer/sampling_params.py:44, 61, 74, 86-91, 104-105`
  > `seq_seed: int(uint64), the random seed of this sequence for sampling.`
  > `self.seq_seed = self._make_seed(seq_seed)` / `if seed is None: return random.getrandbits(64)` / `return seed & 0xFFFF_FFFF_FFFF_FFFF`
- 等级：A
- 意味着什么：默认**不确定**；要复现必须显式传值。这与"确定性采样"是两件事：确定性指"给定 seed 后结果只由输入决定"。
- 疑点：无。

### #14 RNG 的下标是"绝对位置"，不是"第几步"
- 证据：`csrc/engine/prepare_sample.cpp:198`（prefill 末块）与 `:206`（decode/spec）
  > `sample_positions_ptr[sample_offset++] = seq_metadata->seqlen_k;`
  > `sample_positions_ptr[i] = seq_metadata->seqlen_k - seqlen_q + 1;`
  `deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py:247`：`seq = sample_positions[row] * 3 + sample_case`
- 等级：A
- 意味着什么：同一段前缀无论在第几步被采样，抽到的均匀数都一样；因此"抢占后重算""teacher forcing 重放""batch 内位置变化"都不会改变采样结果。这是比"每步计数器"更强的设计，也是能通过 `test_kl` 逐位对拍的前提。
- 疑点：`seqlen_k` 在 YoCo 截断/压缩下是否仍单调对应文本位置，取决于 KV 记账；未展开验证（见 §4）。

### #15 每个位置有 3 条互不干扰的随机子流：采样 / 重采样 / 接受检验
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py:392`（接受检验）、`:450`（采样）、`:481-486`（拒绝后的重采样）
  > `verify_seq = sample_positions[row] * 3 + 2`
  > `position = sample_positions[row] * 3`
  > `reject_seq = sample_positions[row] * 3 + 1`
- 等级：A
- 意味着什么：投机解码的"接受/拒绝"与"正式采样"不能共用一条随机数，否则拒绝一次就会把后续所有 token 的随机性错位；`*3` 的槽位划分就是为此。MTP 草稿采样再用 `_MTP_SEED_OFFSET = 1000000007`（`deepinfer/model_executor/layers/tilelang_ops/fused_sample.py:15, 93`）与主模型错开。
- 疑点：`+2` 的接受检验只抽 32 bit 均匀数（`:393-394`，`rng_rand_float(32,"uniform")`），比采样用的 64 bit 精度低；概率上无碍，但不是"bit 级等价于 fp64 拒绝采样"（E）。

### #16 变长 MTP 的验证长度按**全批排名**分配 —— 这就是必须关掉它的原因
- 证据：`deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py:17, 72, 85-98`
  > `Distribute total_q tokens across sequences based on MTP confidence rates.`
  > `extra_budget = total_q - num_seqs  # number of extra tokens to distribute`
  > `sorted_indices = torch.argsort(flat_rates, descending=True, stable=True)` / `wins = (ranks < extra_budget).sum(dim=1)` / `seqlens_q.copy_(1 + wins)`
  调用点：`deepinfer/worker/worker.py:218-219`（每步按上一步 confidence 重算 `seqlens_q`）、`:230-236`（再把 `seqlens_q` D2H 拷回 CPU）
- 等级：A（机制）/ E（后果）
- 意味着什么：`total_q` 是整批的 token 预算，`ranks` 是在**全批** `[num_seqs, nextn]` 上排出来的。同一请求单独跑时排第 1，和 383 个请求一起跑时可能排第 100 → 本步只验证 1 个 draft。批组成改变 → 每步推进的 token 数改变 → 输出流改变。所以"确定性采样"必须 `DEEPINFER_ENABLE_MTP_VARLEN=0`（`README.md:28`），改用固定宽度 MTP（`DEEPINFER_USE_NEXTN_DECODE=5`，`environ.py:129`；varlen 默认就是 False）。
- 疑点：即使关掉 varlen，MTP 的**接受/拒绝**仍依赖 draft 头本身的输出；如果 draft 头在批不变模式下也稳定，则整链稳定——`test_kl`/`test_batch_invariant` 都是在 `use_nextn_decode=5` 下跑的（`test_batch_invariant.py:33`、`:61`），说明固定宽度 MTP 下确实能过逐位对拍。

### #17 双硬件：批不变的**采样器**两边都有，端到端确定性只承诺 NVIDIA
- 证据：
  - 采样器（Ascend 同构）：`deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py:183-192`（"Encode weights at scale 2**31 … RNG sequence is 3 * sample_position + int(is_nextn_predict)"）、`asc/fused_sample.py:588`（`seed = seq_seeds[row] + seed_offset`）
  - 批不变（仅 GPU）：`worker/runtime_config.py:62-63` 的 NPU 断言；`README.md:27-28` 两处 "(NVIDIA only)"；`tests/models/test_offline_inference.py:109-110` 只在 `accelerator_kind()=="nvidia"` 时设 `DEEPINFER_BATCH_INVARIANT=1`
- 等级：A（代码）/ B（README 的能力边界）
- 意味着什么：书里要区分"确定性采样（seed+位置定序，双硬件）"与"批不变推理（logits 与批无关，仅 NVIDIA）"；Ascend 上 seed 定序只能保证"同样的批形状重跑一致"，不能保证"换批形状一致"。
- 疑点：NPU 的 `assert not DEEPINFER_BATCH_INVARIANT` 说明这是**未完成**而非"不需要"（注释：待完整支持后移除）。

### #18 Ascend 的 `rstd_1ulp`：补偿算术对齐训练侧 FP32 顺序，作用是"数值保真"而非"批不变"
- 证据：`deepinfer/model_executor/layers/tilelang_ops/asc/rstd_1ulp.py:1-5, 77-95`
  > `Compensated RSTD, matching hai_llm.tilelang_kernels.ascend.helper.rstd_1ulp.`
  > `keep the FP32/FMA operation order identical so both inference paths follow the hai-llm numerical standard.`
  > `reciprocal_c0_hi, reciprocal_c0_lo = split_fp32(1.0 / hidden_size)` / `estimate = naive_rsqrt(variance)` / `return compensated_rsqrt(variance, variance_error, estimate)`
  使用点：`asc/norm.py:100`、`asc/fused_transform.py:148`
- 等级：A
- 意味着什么：RMSNorm 的 `1/sqrt(mean(x^2)+eps)` 被拆成 hi/lo 两段 fp32，用 TwoSum/TwoProduct 补偿（`:30-74`），把 rstd 的误差压到 ~1 ulp。目的不是抗批变化，而是让 NPU 上的 norm 与训练框架 `hai-llm` 在**同一算式顺序**下对齐，从而 PPL/KL 基线可以跨硬件比对。
- 疑点：`naive_rsqrt` 用 `1/sqrt`（`vdiv`+`vsqrt`）而非硬件 rsqrt 近似，再由 `compensated_rsqrt` 做牛顿修正（`:30-49`）；为什么不用硬件 rsqrt，只能推测是硬件近似精度不够 1 ulp（E）。

### #19 结构化输出后端 = 上游 xgrammar（不是自研）
- 证据：`deepinfer/structure_output/compiler.py:6-7, 34-39`
  > `from xgrammar import CompiledGrammar, TokenizerInfo` / `from xgrammar import GrammarCompiler as InnerCompiler`
  > `self.inner_compiler = InnerCompiler(tokenizer_info=..., cache_enabled=not self.disable_cache, max_threads=num_threads, cache_limit_bytes=...)`
  `matcher.py:12-14`：`from xgrammar import GrammarMatcher as InnerMatcher` / `from xgrammar.matcher import _FULL_MASK, bitmask_dtype, get_bitmask_shape`
- 等级：A
- 意味着什么：DeepInfer 自研的部分是"什么时候编、在哪个线程编、掩码怎么送到 GPU、投机解码怎么回退"，语法自动机本身复用 xgrammar。书里不要写成"DeepSeek 自研语法引擎"。
- 疑点：`_FULL_MASK`、`bitmask_dtype`、`get_bitmask_shape` 是 xgrammar 的私有名（下划线开头），升级 xgrammar 可能破。

### #20 三种 schema 的分派：json（strict）/ regex / ebnf
- 证据：`deepinfer/structure_output/compiler.py:54-63`
  > `if schema_type == "json": ctx = self.inner_compiler.compile_json_schema(schema, strict_mode=True)`
  > `elif schema_type == "regex": ctx = self.inner_compiler.compile_regex(schema)`
  > `elif schema_type == "ebnf": ctx = self.inner_compiler.compile_grammar(schema)`
  入参来自 `SamplingParams.so_args`（`deepinfer/sampling_params.py:39-43`，可带 `trigger_token_ids` / `bound_token_id`）
- 等级：A
- 意味着什么：`strict_mode=True` 表示 JSON schema 会按"严格子集"编译（不允许 xgrammar 自行放宽），这决定了用户写的 schema 能否编译成功的边界。
- 疑点：无。

### #21 编译是**异步 + 独立线程池**：没编完的请求不进调度
- 证据：`deepinfer/structure_output/so_matcher_builder.py:29, 45-49, 58-95, 97-121`
  > `self._tokenizer_info = TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)`
  > `self._pool = ThreadPoolExecutor(max_workers=max_workers, initializer=initializer)`
  > `self._compiler = self._pool.submit(XGrammarCompiler, self._tokenizer_info, num_threads=2).result()`
  > `def submit(self, seq, so_args, on_ready, *, replay_tokens=None): ... self._pending.append((seq, future, on_ready))`
  > `def pump(self): ... matcher = future.try_result()  # 非阻塞，超时返回 aborted_matcher ... seq.so_matcher = matcher; on_ready(seq)`
  调度口：`deepinfer/engine/llm_engine.py:628`（`submit`）、`:842`（每步 `pump()`）、`llm_engine.py:330-342`（线程数与 CPU 亲和：`get_idle_cpus(self.driver_gpu_id)`）
- 等级：A
- 意味着什么：编译（可能几十 ms 到超时 20 s，`environ.py:168`）不阻塞 decode 主循环；未编完的请求被"挂起"，编完由 `pump` 重新入队。这是 SO 能满足在线延迟的关键工程设计。
- 疑点：`pump()` 在主循环每步做一次 O(pending) 轮询，pending 多时是纯 Python 开销；没有上限/背压（E）。

### #22 编译缓存：默认上限 10 GiB，无上限时明确警告是 DoS/CVE 风险
- 证据：`deepinfer/structure_output/compiler.py:17-31`
  > `# xgrammar <= 0.1.x had production incidents with cache enabled; keep this env as the kill switch.`
  > `elif self.cache_limit_bytes <= 0: logger.warning("... grows unboundedly with distinct schemas and may cause OOM (DoS vector, see CVE-2025-32381) ...")`
  `environ.py:169-170`：`DEEPINFER_XGRAMMAR_DISABLE_CACHE=False`、`DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES=10*1024*1024*1024`
- 等级：A
- 意味着什么：schema 是**用户输入**，缓存键是 schema 文本 → 无上限缓存就是内存 DoS；代码里保留了"kill switch"与 10 GiB 默认上限，这是可写进书的安全细节。
- 疑点：10 GiB 对单进程来说仍然很大，实际部署是否调小未确认。

### #23 掩码生成在 CPU，产物是 pinned 的 uint32 bitmask（vocab/32 个 word）
- 证据：`deepinfer/structure_output/matcher.py:36-45`
  > `mask = torch.empty(get_bitmask_shape(batch_size, vocab_size), dtype=bitmask_dtype, pin_memory=pin_memory)`
  `deepinfer/model_executor/layers/sampler.py:320-326`：`self.bitmasks_cpu = [allocate_vocab_mask(num_sampling_token, self.vocab_size, pin_memory=True) for _ in range(self.max_MTP_value + 1)]`（按需扩容、之后复用）
  `deepinfer/model_executor/layers/sampler.py:361-372`：`mask_filled = matcher.fill_vocab_mask(self.bitmasks_cpu[0], idx)`，随后对每个 draft token `fill_vocab_mask(self.bitmasks_cpu[k+1], idx)`
- 等级：A
- 意味着什么：V4.1 词表 129280 → 4040 word = 16 KB/请求/步；开 5 个 draft 就是 6 份 ≈ 97 KB/请求/步。这是 SO 的隐性带宽成本，也是为什么掩码要 pinned + 单独 copy stream。
- 疑点：H2D 量在 384 并发时约 37 MB/步（E），仓库里没有这个数字的实测。

### #24 GPU 侧只做一件事：按 bitmask 把非法 token 的 logits 填 `-inf`（TileLang kernel）
- 证据：`deepinfer/structure_output/mask.py:25-33` → `deepinfer/model_executor/layers/tilelang_ops/structure_output.py:12-37`
  > `bitmask: [num_requests, ceil(vocab_size / 32)], int32` / `indices: [num_selected_requests]` / `ftp: ... first packed row of each request` / `step: Offset from each request's first packed row`
  > `kernel = ops.apply_token_bitmask_by_step_inplace_tl(vocab_size, ftp is not None)`
  参考实现（语义定义）：同文件 `:40-61`，`logits[logits_row, :vocab_size].masked_fill_(~allowed, -torch.inf)`
- 等级：A
- 意味着什么：`ftp`+`step` 的寻址方式解决了"logits 是 `[total_q]` 打包行、而掩码是 `[num_reqs]`"的错位：投机解码第 k 步的 logits 行 = `ftp[req] + k`。这一个 kernel 把 SO 和投机解码接上了。
- 疑点：无。

### #25 投机解码的 SO 正确性靠 `rollback`：被拒的 draft 必须把自动机退回去
- 证据：`deepinfer/model_executor/layers/sampler.py:352-376`
  ```python
  num_accepted_steps = 0
  for k, draft_token in enumerate(seq_metadata.last_next_n_token):
      if matcher.accept_token(draft_token):
          if matcher.constrained():
              mask_filled = matcher.fill_vocab_mask(self.bitmasks_cpu[k + 1], idx)
              so_indices_cpu[k + 1].append(idx)
          num_accepted_steps += 1
      else:
          break
  matcher.rollback(num_accepted_steps)
  ```
  逐行：`accept_token` 推进自动机并**同时**为下一步生成 mask；只对真正被接受的 draft 计数；遇到第一个拒绝就停；`rollback` 把多推进的状态退回（`matcher.py:206-232`，含 trigger 进度历史 `_progress_history`，上限 `MAX_ROLLBACK_TOKENS = DEEPINFER_USE_NEXTN_DECODE + 2`，`matcher.py:22`）。
- 等级：A
- 意味着什么：没有 rollback，被拒 draft 会永久污染语法状态，后续 mask 全错。这是"SO + 投机解码"最容易出错的地方，代码用"预生成 k+1 份 + 回退"解决。
- 疑点：`max_rollback_tokens` 由 xgrammar 侧限制，若 `nextn` 配置大于编译期常量会抛异常（`matcher.py:224-228` 有显式 raise）。

### #26 `trigger_token_ids` / `bound_token_id`：先自由生成，命中触发串后才开始约束
- 证据：`deepinfer/structure_output/matcher.py:115-153`
  > `def _trigger_pending(self): return self.trigger_progress < len(self.trigger_tokens)`
  > `def constrained(self): return self._bound_fired() or not self._trigger_pending()`
  > 匹配用 KMP 失败函数：`self._trigger_failure = _kmp_failure(self.trigger_tokens)`（`:75`），`while progress > 0 and token != self.trigger_tokens[progress]: progress = self._trigger_failure[progress - 1]`（`:144-145`）
  > `bound` 生效但 trigger 未走完时，掩码被手工写成"只有下一个期望 token 合法"（`matcher.py:183-187`：`vocab_mask[idx].zero_()` 后只置 `expected` 那一位）
  `deepinfer/model_executor/layers/sampler.py:361`：只有 `matcher.constrained()` 为真才填 mask
- 等级：A
- 意味着什么：这解决了"想约束最终答案、但不想约束思维链"的实用需求；`bound` 语义是"触发后先逐 token 强制走完 trigger 序列，再进 grammar"（`deepinfer/sampling_params.py:39-43` 的注释）。KMP 让它对任意前缀 O(1) 摊销。
- 疑点：触发串匹配是**跨 token 的字符串匹配**，但比较的是 token id；若 trigger 串跨越分词边界（同一文本不同切分），匹配不到——示例里靠调用方给对 token id。

### #27 PPL 基线是"数值指纹"：逐个样本**浮点全等**比对
- 证据：`tests/models/test_offline_inference.py:153-158`
  > `for sample, (actual, expected) in enumerate(zip(actual_values, expected_values, strict=True)):`
  > `assert actual == expected, f"DP {rank}, sample {sample}: expected PPL {expected}, got {actual}"`
  生成侧：`examples/offline_inference.py:81`，`decode_ppl = math.exp(-sum(logps) / len(logps))`（对生成 token 的 logprob 取几何平均的倒数）
- 等级：A
- 意味着什么：这不是"质量达标线"，而是"任何 kernel 数值变化都会炸"的回归闸门。能这么写的前提正是 #1-#17：批不变 + 固定 `seq_seed`（`examples/_prompts.py:113`：`seq_seed=i`，temperature 0.4）。
- 疑点：全等断言对硬件/驱动/cuDNN 版本极其敏感；跨机器复现要求同型号 GPU（E）。`test_offline_inference.py:109-110` 只在 NVIDIA 上开批不变，Ascend 的 golden 用另一套数值（见 #28）。

### #28 `offline_decode_ppl.json` 的内容：平台 × 负载 × DP rank → PPL 列表
- 证据：`tests/models/test_offline_inference.py:19-24, 35-46, 133-135`；文件 `tests/models/offline_decode_ppl.json`
  - 两层键：`platform`（`nvidia` / `ascend`）→ `workload`（`short` / `long`）→ `dp_rank`（`"0"`,`"1"`）→ `[decode_ppl, ...]`
  - 每个 rank 的样本数 = `5 * (rank + 1)`（`:134` 断言 `len(values) == 5 * (rank + 1)`），即 rank0 5 个、rank1 10 个；共 4 个 (platform, workload) 组合 × 15 = **60 个 float**
  - 负载定义：`short` = 5 条固定小样本 ×(rank+1)，`--long_context` = 16389 输入 / 4096 输出（`examples/offline_inference.py:18`）
  - 并行配置绑死 golden：NVIDIA `EP4/CP2`（dp=2）、Ascend `EP16/CP8`（dp=2，2 节点 ×8 卡），`test_offline_inference.py:21-24`
- 等级：A
- 意味着什么：一个 60 个 float 的小文件同时锁住了"模型数值 + 采样数值 + 并行切分 + 多模态编码"四件事；改并行度就必须重测（`test_offline_inference.py:21` 的注释明说）。
- 疑点：`nvidia/short/0` 里有一个 4.69 的离群值（`offline_decode_ppl.json`，第 3 个样本，图文 carrots 那条），远高于其余 ~1.0-1.5；这说明该样本上模型很"意外"，属于正常数据而非 bug，但书里引用时要避免误读成"平均 PPL"。

### #29 KL 测试：prompt logprobs 必须逐位相等，decode 与全量 prefill 的散度上界默认 0
- 证据：`tests/models/test_kl.py:30`（`max_kl: float = 0.0`）、`:34`（`Case(DEFAULT_MODEL_NAME, ep_size=8, cp_size=2, use_nextn_decode=5)`）、`:107`（`DEEPINFER_BATCH_INVARIANT=1`）、`:194-206`
  > `assert prompt_logprobs == prompt_logprobs_re`
  > `cos_diff, rmse, chebyshev_dis, kl = cal_diff(torch.tensor(output_logprobs...), torch.tensor(output_logprobs_re...))`
  > `assert max(kls) <= case.max_kl`
  度量定义：`deepinfer/testing.py:4-12`
  > `kl = ((torch.exp(x - y) - 1) - (x - y)).mean().item()`
- 等级：A
- 意味着什么：两件事一次测完——(a) 同一段 prompt 两次前向的 prompt logprobs **完全相同**（分块 prefill 的边界不影响结果）；(b) 逐 token decode 与"一次性全量 prefill 再取 logprob"的输出分布一致。`kl` 不是教科书 KL，而是 `exp(Δ)-1-Δ ≥ 0` 的二阶散度代理（≈Var(Δ)/2），**只有 Δ≡0 时才为 0**，所以 `max_kl=0.0` 等价于"逐位相同"。
- 疑点：`max_kl=0.0` 的默认值意味着这个测试在非批不变模式下几乎必然失败；它实际上是"批不变专属测试"。3 条 16384 token 的图文 prompt × 1024 输出，单次要跑很久（E）。

### #30 `test_batch_invariant`：64 条 ~2.5k prompt，批量结果 vs 单条结果四元组全等
- 证据：`tests/models/test_batch_invariant.py:35`（`NUM_SEQS = 64`）、`:61`（`DEEPINFER_BATCH_INVARIANT=1`）、`:73`（`random.normalvariate(2500, 1000)`）、`:87-92`（`temperature=1.0, top_p=0.7, max_tokens=50, top_logprobs=5`）、`:100-105`
  > `sp.golden_token_ids = [x.token_id for x in output]`
  > `output_single = llm.generate(None, sp, [token_ids], use_tqdm=False)[0]`
  > `assert result_batched == result_single`
  比较字段：`:40-45`，`token_ids / logprobs / top_token_ids / top_logprobs`
- 等级：A
- 意味着什么：注意第二遍用 `golden_token_ids` **强制重放**同一串 token，所以这个测试不需要固定 `seq_seed` 也能比 logprobs——它测的是"同样 50 个 token 在不同批组成下的 logprob 是否相同"，即纯 logits 层面的批不变。
- 疑点：`CASES` 只有一个用例（V4.1-Flash, EP8, nextn=5），TP/CP 组合没覆盖（`cp_size` 未参数化）。

### #31 `test_sampler`：PIT + KS 检验采样分布正确性（不是确定性）
- 证据：`tests/models/test_sampler.py:121`（`DEEPINFER_BATCH_INVARIANT=1`）、`:126-127`（`DEEPINFER_COMPUTE_SUM_SMALLER_PROBS=1`）、`:245-247`
  > `elif all_pv <= 1e-5: errors.append(f"DP {dp_rank} {config_name}: all_p_value={all_pv} is too small; the sampler may have a problem")`
  统计量来源：`deepinfer/model_executor/layers/tilelang_ops/sum_smaller_probs.py:10-22`
  > `Inclusive CDF after sorting tokens by ascending probability. Per row, sum probabilities through the sampled token`
  KS 实现：`test_sampler.py:51-70`（Kolmogorov 渐近式 + Stephens 小样本修正）
- 等级：A
- 意味着什么：如果采样正确，`sum_smaller_probs`（概率积分变换）应服从 U(0,1)。这是"采样器无偏"的统计证明，与"确定性"正交——书里要把这两个测试的**目的**分开写。
- 疑点：p 值阈值 1e-5 很宽（多配置多重比较下仍可能偶发失败），属于"防大错"而非"严格检验"（E）。

### #32 eval 是两阶段：生成写盘、打分读盘
- 证据：`eval/README.md:5-11`
  > `Generation and scoring are separate stages. ... Scoring can therefore be repeated (with a different judge, for example) without regenerating.`
  `eval/run_eval.py:170-174`（按 `plan[dp_rank::dp_size]` 切分）、`:265-295`（写 `meta.json`，含 `mtp: 5`、`mtp_varlen: True`、`batch_invariant`）
- 等级：A
- 意味着什么：生成侧把 `raw_text` 原样落盘（`:143-155`），打分侧可换 judge 重跑，成本结构清晰；`meta.json` 里显式记录 `batch_invariant`，说明作者知道"评测分数"与"是否确定性模式"有关。
- 疑点：`run_eval.py` **默认不开** `--batch-invariant`（`:130-131` 是 opt-in），而 `:120` 强制 `DEEPINFER_ENABLE_MTP_VARLEN=1`——即公开分数来自**最快的非确定性路径**。

### #33 打分：三种提取路径 + rule/llm 两种判分
- 证据：
  - AIME：`eval/aime/bench.py:118-136`（`boxed` → `tail_number` → `none`，尾数只扫 `</think>` 之后或末 400 字符）、`:139-148`（整数精确比较）
  - GPQA：`eval/gpqa/bench.py:111-130`（`ANSWER:` 行 → `\boxed{}` → 尾部孤立字母）
  - HLE：`eval/hle/bench.py:18-24`（官方三段式 `Explanation/Answer/Confidence`），必须 LLM judge（`eval/README.md:167`：`score.py` 拒绝 `--judge rule`）
  - judge 协议：`eval/common/judge.py:13`（exact-match prompt）、`:56`（HLE judge prompt）、`:161-162`（`EVAL_JUDGE_TIMEOUT=600`、`EVAL_JUDGE_ATTEMPTS=3`）、`:196`（judge `temperature=1.0`）
- 等级：A
- 意味着什么：提取优先级的设计说明作者清楚"格式遵循率"是分数的一部分：先按 prompt 要求的格式取，取不到才降级到尾数/尾字母，并在 `summary.json` 里单独统计 `extract_source` 占比。
- 疑点：`tail_number` 这种降级路径会**抬高**分数（模型没按格式答也可能被算对）；`score.py` 报告其占比但没有给出剔除后的分数（见 §4）。

### #34 指标：Pass@1 与无偏 Pass@k 估计器
- 证据：`eval/common/passk.py:8-19`
  > `Unbiased Pass@k estimator: 1 - C(n-c, k) / C(n, k). (The Codex/HumanEval estimator.)`
  > `if num_samples - num_correct < k: return 1.0`
  > `return float(1.0 - math.prod(1.0 - k / i for i in range(num_samples - num_correct + 1, num_samples + 1)))`
  `:22-43`：按 task 分组求均值；样本不足 k 时用 `min(k, len(flags))` 近似而不是丢任务（`:31-33` 注释解释了丢弃会向上偏）。
  `eval/README.md:13`：`The reported metric is Pass@1.`
- 等级：A
- 意味着什么：默认只报 Pass@1（= 逐样本正确率的均值），但代码能算 Pass@k；AIME n=32 → 默认 k 列表是 1,2,4,8,16,32（`default_k_list`，`:46-51`）。
- 疑点：`estimator` 在 `n-c<k` 时直接返回 1.0 是标准做法但对小 n 有偏；AIME 只有 30 题，置信区间很宽（E）。

### #35 评测默认配置：reasoning effort 100、144K 上下文、屏蔽 4 个 token、MTP 5 + varlen
- 证据：`eval/run_eval.py:58-60`（`--reasoning-effort` 默认 100，或 low/high/max）、`:61`（`--max-len` 默认 `1024*144`）、`:66`（`--max-num-seqs` 384）、`:67`（`--max-num-batched-tokens` 32768）、`:72`（`--decode-blacklist` 默认 `0,2,129262..129264`）、`:118-120`（强制 MTP 5 + varlen）
- 等级：A
- 意味着什么：`decode_blacklist` 屏蔽的 4 个 token 是"评测协议"的一部分（可能是特殊 token / 失控重复 token），复现分数时必须一致；`reasoning_effort` 是 prompt 侧的控制量（V4.1 的思考预算）。
- 疑点：`reasoning_effort` 如何注入 prompt 在 `eval/common/encoder.py`（23 行）里，未细读。

### #36 公开参考分数（仅 3 个数，且自述"指示性"）
- 证据：`eval/README.md:208-221`
  > `| AIME 2026 | 99.9% |` / `| GPQA Diamond | 93.4% |` / `| HLE-Full | 42.6% |`
  > `Due to the stochastic nature of LLM generation, repeated runs may differ by a small margin. These figures are therefore indicative rather than exact`
  题量与采样：`eval/README.md:15-17`（AIME 30×32、HLE 2500（2158 文本 + 342 图）×1、GPQA 198×8）
- 等级：B（仓库内官方文档）
- 意味着什么：这是本仓库唯一的"公开指标"，也是全书唯一可以引用 DeepInfer 端到端质量的数字；但它是**非批不变 + varlen MTP** 下测的，不能当作"确定性模式"的性能。
- 疑点：没有给出硬件型号/卡数（只说 default configuration）、没有误差棒、没有 AIME 2025 等历史基准。

## 3. 必答问题逐条回答

### Q1 批不变的数学原理与代码实现；改了哪些算子；性能代价
**原理（浮点归约顺序为何受批大小影响）**
- 浮点加法不满足结合律：`(a+b)+c ≠ a+(b+c)`（每一次舍入都丢信息）。并行归约把一行的和拆成若干部分和再合并，**拆法**（几段、每段哪些元素、树多高、段间谁先加）决定最终舍入结果。
- GPU kernel 的拆法由 launch 时的形状决定：GEMM 的 tile/split-K 数、attention 的 split-KV 段数、softmax 的 block size 都是在看到 M（token 数 = 批的组成）之后由 heuristic 选的，目的是把 SM 填满。于是同一行文本在 batch=1 与 batch=64 下会走不同的拆法 → 不同舍入 → logits 的末位不同（差 1 ulp 就足以在 top-p 边界上改变一个 token）。见 #4、#5、#8。
- 因此"批不变"的充要工程条件：**一行的计算图与归约树只能是"这一行内容 + 固定 kernel 配置"的函数，不能含 `total_q`**。DeepInfer 的做法就是逐项把 `total_q` 从形状里摘掉。

**实现（改了哪些算子）**
| 子系统 | 改动 | 证据 |
| --- | --- | --- |
| Attention (FlashMLA sparse decode) | 关 split-KV（`enable_split_kv=false`），一条 query 的 topk 归约在一个 CTA 内按固定顺序完成 | #4 |
| MoE gate / GEMM (DeepGEMM) | deterministic 模式：禁 cuBLASLt 分派（它按 M 选 kernel）、mega-gate 禁 split-K、mega_mhc 固定 split 数 | #5 |
| lm_head | 同一个 `bf16_gemm_nt`，随全局 deterministic 生效 | #6 |
| EP 通信 (DeepEP) | 开 EPLB 重排时关 `allow_multiple_reduction`（多次归约写同一 buffer） | #7 |
| Prefill 融合 | 稠密 packed-KV 的 prefill 不再走 `fused_norm_rope_attn_rope_cast`（稀疏层与无 packed-KV 的情形仍走融合路径；融合核的算子顺序与 decode 路径不同） | `attn.py:86-93` |
| 采样前向 | 一行一个 CTA、tile/threads 只由 vocab 定、LSE 保留固定 layout | #8、#9 |
| top-p/top-k | 概率量化成 uint32（2^31）后用 radix 直方图找阈值，整数累加 | #10 |
| 采样 | u32 定点 CDF + 模拒绝，决策只做整数比较 | #11 |
| 其余 | `tile_kernels.use_deterministic_algorithms(True)`（实现不在仓库，未确认） | #3 |

**性能代价**
- 仓库里**没有任何数字**。唯一带倾向的描述是 `eval/run_eval.py:74` 的帮助文本 `results independent of batch composition (slower)` 与 `README.md:27` 未提代价。**未能确认任何量化代价**——没有 benchmark 表、没有注释里的百分比、git 历史只有一个 initial commit。
- 代价来源（E，可从代码逐条推出）：
  1. 关 split-KV：小批（decode 常见 1-8 条/卡）时，FlashMLA 原本用 split-KV 把一条 query 的 topk 摊到多个 SM；关掉后并行度掉到"query×head"级别。代码里 split-KV 的触发条件是 `topk + extra_topk > 640`（#4）；V4.1 的 `index_topk` 来自模型配置（仓库内不落具体值，`model_metadata_utils.py:61` 只读不设），只要它超过 640，**默认生产路径就在用 split-KV**，关掉它是实打实的损失，且批越小损失越大。
  2. 禁 cuBLASLt：失去 NVIDIA 手工调优的 shape 特化 kernel，改走 DeepGEMM JIT 的固定 heuristic；大 M 时 DeepGEMM 通常不输，小 M 时可能更差。
  3. 关 prefill 融合核：稠密 prefill 多一次 norm/rope/cast 的读写与 launch。
  4. 采样器一行一 CTA + 全词表扫两遍（LSE + 采样/过滤）：相比"一个 CTA 处理多行"的写法，行数少时占用率低。
- 结论写法建议：**"官方不公布代价，只在 CLI help 里写 slower；从实现看，代价集中在 decode 小批时的 attention 并行度"**（E），不要编数字。

### Q2 确定性采样：`seq_seed` 如何起作用；为什么必须关变长 MTP；仅 NVIDIA 还是双硬件
**`seq_seed` 的作用链**
1. 用户传 `SamplingParams(seq_seed=...)`，缺省 `random.getrandbits(64)`，并做 `& 0xFFFF...` 截断与 uint64 范围校验（#13）。
2. 每请求的 seed 展开成每行 `seq_seeds: uint64`（`sampler_utils.py:134` 的接口说明；`sampling_inputs.py:47` 分配 `num_reqs` 长度）。
3. kernel 里 `seq = sample_positions[row] * 3 + sample_case`，`T.rng_init(seq_seeds[row], seq)`（#11、#14）——**seed 只管"哪个流"，位置管"流里的第几个数"**。
4. `sample_positions` 由 C++ 侧填：prefill 末块 `= seqlen_k`，spec/decode `= seqlen_k - seqlen_q + 1`（#14）。所以是"绝对 token 位置"，不是步数。
5. 三条子流：`3p+0` 正式采样、`3p+1` 拒绝后的重采样、`3p+2` 接受检验；MTP 草稿再整体加 `1000000007`（#15）。
6. 结果：给定 (prompt, seed, 采样参数)，每个位置的均匀数固定 → 采样结果固定；与"这一批里还有谁"无关，与"这条请求被调度到第几步"无关，也与"是否重放"无关。

**为什么必须关变长 MTP**
- 变长 MTP（DSpark/confidence-scheduled verification）每一步按 confidence 决定"这条请求验证几个 draft"。分配方式是**全批排名**：`extra_budget = total_q - num_seqs`，把 `[num_seqs, nextn]` 的累积 confidence 拉平后 `argsort` 取前 `extra_budget` 个，每个请求拿 `1 + wins`（#16）。
- `total_q` 是整批预算、`ranks` 是全批排名 → 同一请求的验证长度是**其他请求的函数**。批组成一变，本步推进的 token 数就变，输出流就变（哪怕每个位置的 RNG 都固定）。
- 关掉后每步固定验证 `DEEPINFER_USE_NEXTN_DECODE`（默认 5）个 draft，`seqlens_q` 与批无关。
- 附带代价：varlen 每步还要把 `seqlens_q` 从 GPU 拷回 CPU（`worker.py:230-236`），关掉也省了这次同步。

**硬件范围**
- `DEEPINFER_BATCH_INVARIANT` 在 NPU 上直接 `assert`（#17）→ 批不变 **仅 NVIDIA**，README 两处标注一致。
- 但"seed + 位置定序"的采样器 Ascend 也实现了（`asc/random_sampling.py:183-192`，同样的 `3*pos+case`，同样的 2^31 定点）。所以：**确定性采样的机制双硬件都有；端到端"换批形状也一致"只有 NVIDIA 承诺**，因为 NPU 的 logits 本身还不保证批不变。
- 结论一句话：`DEEPINFER_BATCH_INVARIANT=1 + DEEPINFER_ENABLE_MTP_VARLEN=0 + 固定 seq_seed` 是官方三件套（`README.md:28`），缺一则只能得到"同批重跑一致"，不是"任意批一致"。

### Q3 结构化输出：语法引擎、编译、匹配、掩码的完整流程与位置
**引擎**：上游 **xgrammar**（`compiler.py:6-7`、`matcher.py:12-14`），不是自研。DeepInfer 自研的是编排层（4 个文件共 595 行）。
**流程（按时间顺序，含代码位置）**
1. **入参**：`SamplingParams.so_args = {"schema_type": "json"|"regex"|"ebnf", "schema": ..., 可选 "trigger_token_ids"/"bound_token_id"}`（`deepinfer/sampling_params.py:39-43`；示例 `examples/_structured_output.py:23-36` —— 故意用"让它写 JSON"的反向 prompt 验证约束确实生效）。
2. **建 TokenizerInfo（一次）**：`TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)`（`so_matcher_builder.py:29`）。这是 xgrammar 把 token 串映射到自动机边的前提。
3. **建编译器（一次）**：`ThreadPoolExecutor(max_workers=len(idle_cpus), initializer=绑核)`，池里先 `submit(XGrammarCompiler, tokenizer_info, num_threads=2).result()`（`so_matcher_builder.py:33-49`），worker 用 `os.sched_setaffinity(0, ...)` 绑到空闲 CPU（`:37-43`，注释解释了为什么是 `0` 而不是 `psutil.Process()`）。
4. **异步编译（每请求）**：`submit(seq, so_args, on_ready)` → `_pool.submit(_build)`（`so_matcher_builder.py:78-95, 97-107`）。`_build` 里 `dispatch_and_compile` 按类型编译（`compiler.py:54-63`），再 `XGrammarMatcher.from_compiled(...)`（`matcher.py:94-106`）。编译期异常包装成 `RuntimeError("xgrammar ...")`（`compiler.py:64-69`）。
5. **等待与入调度**：主循环每步 `pump()`（`llm_engine.py:842`），`future.try_result()` 非阻塞，超时（`DEEPINFER_XGRAMMAR_TIMEOUT=20s`，`environ.py:168`）返回 `aborted_matcher`（`matcher.py:87-92`，超时判定在 `matcher.py:319-338`）→ 请求标 `FINISHED_SO_FAILED`（`llm_engine.py:620-626`）。失败只杀单请求，不崩 engine（`deepinfer/model_executor/layers/sampler.py:377-386` 的注释写明了原因：CP rank0 上 raise 会连坐全部请求 + NCCL barrier 超时）。
6. **状态推进（每 decode step）**：`_get_so_mask`（`deepinfer/model_executor/layers/sampler.py:301-408`）。`so_status` 语义：`0`=matcher 已建但未生效、`1`=prefill 最后一块刚结束（首次生效）、`2`=解码中（要先 replay 上一步被接受的 token，`deepinfer/model_executor/layers/sampler.py:352-357`）、`<=0`=不参与。`so_status==1` 的切换点在 `llm_engine.py:914-927`。
7. **掩码生成**：`matcher.fill_vocab_mask(self.bitmasks_cpu[step], idx)` → `inner_matcher.fill_next_token_bitmask(...)`（`matcher.py:181-204`），产物是 `[num_reqs, ceil(V/32)]` int32 的 pinned 张量（#23）。当前步一份，每个 draft 再各一份（`deepinfer/model_executor/layers/sampler.py:361-372`）。
8. **掩码搬运**：专用 stream + event，先 `wait_event(copy_after)` 再 `m.cuda(non_blocking=True)`，最后 `record_stream` 防复用（`deepinfer/model_executor/layers/sampler.py:391-407`）。消费侧 `deepinfer/model_executor/layers/sampler.py:163-164` 等 event。
9. **掩码应用**：prefill 分支 `deepinfer/model_executor/layers/sampler.py:187-188`；decode 分支 `deepinfer/model_executor/layers/sampler.py:433-437`（按 `step` 逐层，带 `ftp` 寻址）。kernel 用 TileLang 写（`deepinfer/model_executor/layers/tilelang_ops/structure_output.py:12-37`），语义是 `masked_fill_(~allowed, -inf)`（`:40-61` 的参考实现）。
10. **回退**：`matcher.rollback(num_accepted_steps)`（#25）。
11. **收尾**：接受 EOS 后 `finished()` 为真，`fill_vocab_mask` 直接 `copy_(self.mask_after_eos)`（一张全 1 掩码，`matcher.py:105, 189-191`）→ 停止约束，让 EOS 之后的 token 不再被限制；`examples/_structured_output.py:46-48` 断言最后一个是 EOS。

**要点**：编译产物不落盘、也不接受序列化（`so_matcher_builder.py:60-61` 显式拒绝 `serialized`）。这与 `doc/inference_deployment_qa.md:282-288` 描述的"生产环境用独立 SO 机器预编译、推理前取结果"**不一致**：开源版是"进程内线程池现编"。见 §4。

### Q4 PPL 基线测试与 KL 测试分别保证什么；`offline_decode_ppl.json` 里是什么
**PPL 基线保证的是"数值指纹不变"**
- 测什么：固定 5 条（按 rank 倍数复制）样本、固定 `seq_seed=i`、temperature 0.4、`max_tokens=128`（long 为 4096），把每条请求生成 token 的 `exp(-mean(logprob))` 记下来，与 golden **浮点全等**比较（#27）。
- 保证：(a) 模型前向数值不变（任何 kernel 改动、并行切分改动都会破坏它）；(b) 采样数值不变（定点 CDF / seed 定序没被改坏）；(c) 多模态编码（图片 → token）不变；(d) 并行配置（EP/CP/DP）与 golden 绑定，改配置必须重测（`test_offline_inference.py:21` 注释）。
- 不保证：语言质量。PPL ≈ 1.0-1.5 只说明"模型对这几条短问答很自信"，与"答得对"无关；`nvidia/short` 里那个 4.69 就说明它测的是"是否复现"，不是"是否好"（#28 疑点）。

**KL 测试保证的是"prefill 与 decode 两条路径算出同一个分布"**
- 测什么：同一批图文 prompt（3 条、展开后 16384 token）跑两遍：第一遍分块 prefill + 逐 token decode，取 prompt logprobs 与输出 logprobs；第二遍把整段当成一个 prefill 再取一次 logprobs（`test_kl.py:180-193`）。
- 保证：
  1. `prompt_logprobs == prompt_logprobs_re` —— **逐位相等**（#29），说明分块 prefill 的 chunk 边界不影响结果，KV cache 写入/读取没有数值漂移。
  2. 输出段的散度 `mean(exp(Δ)-1-Δ) ≤ 0.0` —— 默认 `max_kl=0.0`，等价于逐位相同（#29）。说明"边走边算"与"一次算完"给出同一分布。
- 为什么这条重要：投机解码、chunked prefill、SWA 重放、prefix cache 命中都会让"同一 token 在不同上下文路径下被算出来"；KL=0 是把这些路径钉死的唯一验收标准。
- 局限：只在 `EP8/CP2 + nextn=5` 一个配置下跑（#29），且 `kl` 是散度代理而非真 KL。

**`offline_decode_ppl.json` 的内容**
- 结构：`{platform: {workload: {dp_rank: [ppl, ...]}}}`，`platform ∈ {nvidia, ascend}`，`workload ∈ {long, short}`，`dp_rank ∈ {"0","1"}`。
- 规模：每个 (platform, workload) 下 rank0 有 5 个值、rank1 有 10 个值（`len == 5*(rank+1)`），4 个组合 × 15 = **60 个 float**。文件 1.9 KB。
- 数值范围：long 组 1.0003–1.0339，short 组 1.0068–4.6928；NVIDIA 与 Ascend 的对应值**接近但不相等**（如 `nvidia/long/0[0]=1.0290821` vs `ascend/long/0[0]=1.0297303`），这正是"两套硬件各自的 golden"。
- 用途：CI 闸门 + 平台数值档案；刷新工具 `examples/refresh_offline_decode_ppl.py`（多节点按 rank 分工、跨节点文件锁 `offline_decode_ppl.json.lock`，见 `examples/README.md:15`）。

### Q5 `eval/` 评测流程（AIME/HLE/GPQA）与公开指标
**流程**
1. `eval/<bench>/run.sh` 两步：先 `python eval/run_eval.py --bench X --data-file ... --n-sample N --reasoning-effort J`，再 `python eval/score.py --out-dir ... --judge llm|rule`（`eval/aime/run.sh:18-29`）。
2. 数据：AIME `MathArena/aime_2026` jsonl（30 题，`aime/bench.py:34`）；GPQA Diamond 198 题（支持 parquet 与公开镜像两种格式，`gpqa/bench.py:46-60`）；HLE 2500（2158 文本 + 342 图，`eval/README.md:16`）。
3. 生成：所有 (problem, sample) 组合拉平成 `plan`，按 `plan[dp_rank::dp_size]` 均分到各 DP rank（`run_eval.py:174`，注释：按 sample 切分，避免某 rank 继承长尾）；`SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_len - prompt_len, seq_seed=_seed(problem, sample))`（`:225-235`）；`_seed = (global_index * n_sample + sample_index) % 2**64`（`:136-138`，注释：同样 seed 会产出同样输出，所以每个 sample 必须自己的 seed）。
4. 落盘：`raw_rank*.jsonl`，每行含 `task_id / sample_idx / seq_seed / prompt_len / num_output_tokens / finish_reason / raw_text`（`:145-155`）。
5. 打分：`score.py` 读 `meta.json` + 所有 `raw_rank*.jsonl`，用 bench 的 `score_rollout` 提取（#33），可选 LLM judge，然后 `summarize` 算 `score`（=Pass@1）与 Pass@k（#34）。
6. 覆盖性检查：缺 rank 文件/缺题/缺 sample 都会 warn 并写进 `summary.json`（`score.py:130-144`），不会静默丢数据。
7. judge 一致性：`rule_vs_llm_agreement`、`judge_unparsed`、`judge_verdict_unparsed` 三个比例都进 summary（`score.py:185-191`）；AIME 上两种模式在 960 条 rollout 上**完全一致**（`eval/README.md:186-188`）。

**指标**
- 只报 **Pass@1**（`eval/README.md:13`）：AIME 30 题 ×32 采样取均值、GPQA 198×8 取均值、HLE 2500×1。
- 参考分数（`eval/README.md:213-217`，B 级）：**AIME 2026 99.9%、GPQA Diamond 93.4%、HLE-Full 42.6%**。
- 条件必须带上：V4.1 模型、默认配置（MTP 5 + **varlen 开**、`--batch-invariant` **未开**、reasoning effort 100、max_len 144K、`max_num_seqs=384`、blacklist `0,2,129262..129264`）；文档自述"stochastic，重复跑有小幅差异，仅供 sanity check"。
- 仓库里**没有**分项（HLE 文本/图像分开的分数、GPQA 分学科、AIME 逐题）——只有 `by_modality` 的分组 Pass@1 计算代码（`score.py:175-183`），没有随仓库发布的产物。

## 4. 新发现的问题

1. **生产文档与开源实现的 SO 编译方式不一致。** `doc/inference_deployment_qa.md:282-288` 写"编译与推理解耦，请求进引擎前取编译结果（独立 SO 机器）"；开源代码 `so_matcher_builder.py:60-61` 明确拒绝 `serialized` 语法，只能在进程内线程池现编。要么开源版裁掉了跨机编译产物格式，要么文档描述的是未开源的部署形态。书中引用文档时必须注明这一点。
2. **`replay_tokens` 机制存在但没有调用者。** `SOMatcherBuilder.build/submit` 都支持 `replay_tokens`（`so_matcher_builder.py:58-92`），但 `llm_engine.py:628` 调用 `submit` 时没传。推测是为"抢占后重算"预留：SO 请求被抢占、prefix 被丢弃重算后，matcher 状态无法从 token 流恢复，只能靠 replay 重建，而这条路没接上。影响：SO 请求一旦被抢占，`so_status`/matcher 状态与实际已生成 token 可能不一致（`deepinfer/model_executor/layers/sampler.py:352-357` 会 `accept_token` 失败并判死请求，不会静默错）。**未能确认**抢占路径是否真的会丢 matcher。
3. **`test_batch_invariant` 只覆盖一个用例。** `CASES` 只有 `(V4.1-Flash, ep=8, nextn=5)`（`test_batch_invariant.py:32-34`），`cp_size` 没参数化；而 `test_kl` 用的是 `cp=2`。批不变在 CP（context parallel，KV 跨卡切分）下是否仍成立，没有独立用例。
4. **`test_kl` 的 `max_kl=0.0` 实际上把测试绑死在批不变模式。** 非批不变模式跑必然失败，所以它不是"通用数值回归"而是"批不变专属"。若将来新增硬件（NPU）想复用这个测试，需要参数化阈值。
5. **`score.py` 只报告降级提取路径的占比，不报告剔除后的分数。** `extract_source` 有 `boxed / tail_number / none`（AIME）与 `answer_line / boxed / tail_letter / none`（GPQA），但 summary 里只有 `score`。无法从产物判断"格式遵循率差"对分数的贡献。建议书里提醒读者：跨框架比 AIME 分数时，提取策略不同会造成百分级差异（E）。
6. **PPL golden 对硬件极其敏感，却被放在同一个 JSON 里。** `nvidia` 与 `ascend` 各自一套值（#28），且 `test_offline_inference.py:23` 把 EP/CP 与 golden 绑定；这意味着"换卡型/换并行度 = 必须重测 golden"，不是"跑一下 CI 就行"。书的读者如果照抄这个测试到自己的集群，会立刻失败。
7. **`offline_decode_ppl.json` 的 `short` 组有一个 4.69 的离群 PPL。** 说明该样本的生成 token 概率很低（模型"没料到"）。这不影响测试有效性（全等比较），但如果书里用它当"PPL 约 1.0"的例子会误导。
8. **`DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` 默认 10 GiB 偏大。** 对一个单进程推理服务，10 GiB 的语法缓存可能挤占 KV cache；`compiler.py:24-29` 的警告只在"<=0（无上限）"时打印，10 GiB 这种"有上限但很大"的配置没有提醒。
9. **`test_sampler` 的统计阈值（p>1e-5）与多重比较。** 4 组采样配置 × 多 DP rank 同时检验，族错误率远高于 1e-5；这是"防大错"设计，不是严格检验。书里不要把它描述成"证明了采样分布正确到 1e-5"。
10. **`eval` 默认 `DEEPINFER_ENABLE_MTP_VARLEN=1` + 不开批不变**（`run_eval.py:118-131`），而"确定性采样"要求两者相反。也就是说：**公开分数与可复现性是两条互斥的路径**，仓库没有给出"确定性模式下的分数"。这是全书最值得强调的一句 caveat。

## 5. 可用于图表的材料

**表 A：批不变的四个作用点（"一个 flag → 四个子系统"）**
| 子系统 | 默认（快路径） | `DEEPINFER_BATCH_INVARIANT=1` | 代码 |
| --- | --- | --- | --- |
| FlashMLA sparse decode | split-KV（topk+extra>640 时开） | 关 split-KV，单 CTA 扫完 topk | `sparse_decode.cpp:223` |
| DeepGEMM | cuBLASLt 分派 / mega-gate split-K / 动态 mhc splits | 全部禁用/固定 | `gemm.hpp:519`、`mega_gate.hpp:110-126`、`mega_mhc.hpp:16-18` |
| 稠密 prefill 融合核 | `fused_norm_rope_attn_rope_cast` | 走非融合路径 | `attn.py:86-93` |
| DeepEP（开 EPLB 时） | `allow_multiple_reduction=True` | `False` | `userbuffer.py:79` |

**表 B：采样 RNG 的三个槽位（同一位置的随机流划分）**
| 槽位 | 用途 | 代码 |
| --- | --- | --- |
| `3p + 0` | 正式采样（top-p/top-k 过滤后的 CDF） | `cuda/fused_sample.py:450` |
| `3p + 1` | 拒绝后的残差重采样 | `cuda/fused_sample.py:481-486` |
| `3p + 2` | 投机解码接受检验（32-bit 均匀数 vs `exp(accept_lp)`） | `cuda/fused_sample.py:392-394` |
| `+1000000007` | MTP 草稿采样整体错开 | `deepinfer/model_executor/layers/tilelang_ops/fused_sample.py:15,93` |

**图 C：确定性采样的因果链（可画成一条时间轴）**
`SamplingParams.seq_seed` → `seq_seeds[req]` → 行级 `seq_seeds` → `seq = 3*sample_positions + case` → `T.rng_init(seed, seq)` → 拒绝采样取 64-bit 均匀数 → 定点 CDF 反演 → token。旁挂两个"必须为假"的开关：`DEEPINFER_ENABLE_MTP_VARLEN=0`、NPU 不支持 `DEEPINFER_BATCH_INVARIANT`。

**图 D：SO 的 11 步流水线**（可直接用 Q3 的编号列表做泳道图）
请求入队 → TokenizerInfo（一次性）→ 线程池异步编译（xgrammar）→ `pump` 入调度 → prefill 末块 `so_status=1` → 每步 `accept_token` 回放 → `fill_next_token_bitmask`（CPU, pinned）→ 独立 stream H2D → TileLang `masked_fill(-inf)` → 采样 → `rollback(accepted)`。

**表 E：测试体系四件套**
| 测试 | 规模 | 判据 | 保证什么 |
| --- | --- | --- | --- |
| `test_batch_invariant.py` | 64 条 × ~2.5k prompt × 50 token，EP8 | 批量 vs 单条，token/logprob/top-k 四元组全等 | logits 与批无关 |
| `test_kl.py` | 3 条 × 16384 token 图文 × 1024 输出，EP8/CP2 | prompt logprobs 逐位相等；`max_kl=0.0` | prefill 与 decode 同分布 |
| `test_sampler.py` | 4 组 (top_p, T)（含 greedy）× 128 轨迹 × ≤2048 token，EP8 | PIT 的 KS 检验 p>1e-5 | 采样无偏 |
| `test_offline_inference.py` | 4 组 (platform, workload) × 15 样本 | PPL 浮点全等 | 端到端数值指纹 |

**表 F：评测协议（写进书里必须带的条件）**
| 项 | 值 | 来源 |
| --- | --- | --- |
| AIME 2026 | 30 题 × 32 采样，Pass@1 | `eval/README.md:15` |
| GPQA Diamond | 198 题 × 8 采样 | `:16` |
| HLE-Full | 2500（2158 文本 + 342 图）× 1 | `:16` |
| temperature / top_p | 1.0 / 1.0 | `run_eval.py:24-27` |
| reasoning effort | 100（或 low/high/max） | `run_eval.py:58-60` |
| max_len | 147456（1024×144） | `run_eval.py:61` |
| MTP | 5 + varlen **开** | `run_eval.py:118-120` |
| batch invariant | **默认关**（`--batch-invariant` opt-in） | `run_eval.py:130-131` |
| 参考分 | 99.9 / 93.4 / 42.6 | `eval/README.md:213-217` |

**数据点（可直接引用）**
- V4.1 词表 129280 → 每请求每步 1 份 bitmask = 4040 word = 16160 B；开 5 draft 共 6 份 ≈ 97 KB（#23）。
- `offline_decode_ppl.json`：60 个 golden float，4 个 (platform, workload) 组合，rank0 5 个 / rank1 10 个（#28）。
- `MAX_ROLLBACK_TOKENS = DEEPINFER_USE_NEXTN_DECODE + 2` = 7（默认 nextn=5）；`SO` 每步最多预生成 6 份掩码（`max_MTP_value + 1`）。
- 定点概率刻度 `SCALE = 2^31`，量化为 floor，理论偏差 ≤ 2^-31（#10、#11）。
