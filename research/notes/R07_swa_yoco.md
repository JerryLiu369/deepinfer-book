# R07 SWA 有界重放、YOCO 截断逻辑与精度等价性

## 0. 元信息

- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（本地 `research/sources/DeepInfer`）
- 查阅的主要来源（一手，A 级）：
  - `deepinfer/environ.py`（`DEEPINFER_YOCO_*`、`DEEPINFER_MAX_BACKTRACKING_TOKENS`、`DEEPINFER_PREFILL_STORE_*`）
  - `deepinfer/worker/cache_engine.py`（SWA 环尺寸、state block 布局、`state_size_to_disk`）
  - `deepinfer/worker/model_runner.py`（KEEP_TOKENS 断言、`window_size` 注入）
  - `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`（`yoco_swa_visibility` / `fill_yoco_truncation_metadata` / `make_swa_block_load_plan`）
  - `deepinfer/model_executor/input_metadata/yoco.py`、`.../input_metadata.py`、`deepinfer/model_executor/layers/tilelang_ops/swa_index.py`
  - `deepinfer/model_executor/models/v4_modules/yoco.py`、`attn.py`、`deepinfer/model_executor/models/deepseek.py`、`decoder_layer.py`
  - `csrc/engine/flops_estimator.hpp` + `tests/engine/test_flops_estimator.py`
  - `examples/yoco/yoco_equiv.py`、`examples/yoco/README.md`、`examples/yoco/run_yoco_equiv_l2.sh`、`examples/benchmark.py`、`examples/sweep/run_sweep.sh`
  - `doc/inference_deployment_qa.md`（SWA KV cache / KV cache hits and recomputation）
- 查阅的主要来源（二手，B/C 级）：DeepSeek-V4.1-Flash 技术报告（arXiv 2609.19969，§3.2.2）；vLLM 博客 `2026-10-07-deepseek-v41-flash`；vLLM PR #58132。
- 注意：本笔记不给"我跑过测试"的结论。仓库无 `.github/`，没有任何 CI 产物或测试输出留档；`yoco_equiv` 的 L1 也需要一张可见 GPU（`yoco_equiv.py:376-377`），L2 需要 8 卡 + V4.1 HF checkpoint。所有"结论"均指**代码/断言的字面语义**。

## 1. 三个最重要的发现

1. **有界重放省的是"行"，不是"层"**：`KEEP_TOKENS` 的唯一语义是"consumer（decoder）层只对末尾 K 个 query 行做前向"。C++ 侧用 `cutoff = data_len − keep_tokens` 把 query 行裁到 `(cutoff, data_len]`（`prepare_tensor_input_v4.cpp:222-230`），FLOPs 参考实现写成 `positions[positions > cutoff]`（`test_flops_estimator.py:52-58`）。所有"省一半/省 98%"的说法都能由这一个行数比推出，不需要别的机制。
2. **140 与 134 不是同一个对象，二者是"不等式"关系**：134 = 128（窗口）+ 6（`MAX_BACKTRACKING_TOKENS` = `nextn+1` = 5+1）是**存储/命中口径**（推理环大小、每层落盘 token 数、命中上限 L−134）；140 是**示例/默认策略值**（保留多少末尾 query 行），只受 `KEEP_TOKENS ≥ 窗口 + B` 约束，140 比下限多 6。代码里没有任何地方要求二者相等。
3. **"几乎减半"依赖三个前提，缺一不可**：N ≫ n_win；CED 让 decoder 的块状 KV 由 encoder 末层隐状态**一次性投影**（所以 decoder 不必逐 token 跑）；两半每行算力大致相当。报告把它写成 `O(NL) → O(NL/2 + n_win·L/2) ≈ O(NL/2)`（B 级，arXiv 2609.19969 §3.2）。实测 TTFT 降幅 25–45%（C 级，vLLM/GB200），不是 50%。

## 2. 事实条目

### #1 `KEEP_TOKENS` 的语义与三处校验
- 证据：`deepinfer/environ.py:191-194`
  > `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS = EnvVar.int()` … "截断开启时保留的末尾 query token 数。mask 与 compact 的 SWA 可见域均为原始滑窗和末尾 KEEP_TOKENS 矩形的交集。必须满足 KEEP_TOKENS >= window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS"
- 证据（三处冗余校验）：
  - `deepinfer/worker/model_runner.py:107-113`：`assert self.window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS <= envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`
  - `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:352-357`：`TORCH_CHECK_GE(keep_tokens, args.window_size + args.max_backtracking_tokens)`
  - `examples/yoco/yoco_equiv.py:385-388`、`:793-795`（测试侧同款断言）
- 等级：A
- 意味着什么：这是"重放多长"的唯一开关，值为 0 时必须显式开 mode 才生效（默认 `off`，`environ.py:188`）。
- 疑点：无。

### #2 "6" 的来源：`MAX_BACKTRACKING_TOKENS = nextn + 1`
- 证据：`deepinfer/environ.py:128-132`
  > `DEEPINFER_USE_NEXTN_DECODE = EnvVar.int(default=5)` / `DEEPINFER_MAX_BACKTRACKING_TOKENS = EnvVar(lambda v: max(_int(v), envs.DEEPINFER_USE_NEXTN_DECODE + 1))`
- 等级：A
- 意味着什么：默认 5 个 draft → 需要回退 6 个 token；这就是文档 `doc/inference_deployment_qa.md:191` 里"128-token window + 6 tokens of speculative backtracking = 134"的 6。`examples/benchmark.py:164` 显式设 `DEEPINFER_MAX_BACKTRACKING_TOKENS=1` 也会被抬到 6。
- 疑点：`DOC:165` 另有一句"复制位置最多越过边界 5 个 token，由 ring 比窗口多出的 6 个位置吸收"——同一个 6 被用在两个地方（回退 & 越界吸收），是否同一预算未说明。

### #3 SWA 环：推理 134 / kvstore 402
- 证据：`deepinfer/worker/cache_engine.py:88-110`
  > `def get_inference_swa_block_size(window_size): return (window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS) if window_size > 0 else 0`
  > `def get_yoco_store_extra_tokens(): return max(0, envs.DEEPINFER_YOCO_PREFILL_KEEP_TOKENS) if ...!= "off" else 0`
  > `return get_inference_swa_block_size(window_size) + ((envs.DEEPINFER_PREFILL_STORE_EXTRA_SIZE + get_yoco_store_extra_tokens()) if envs.DEEPINFER_PREFILL_STORE_EXTRA_STATE else 0)`
- 等级：A
- 意味着什么：`window=128, B=6` → 推理环 **134**；开 `PREFILL_STORE_EXTRA_STATE`（截断时强制，`examples/benchmark.py:156-157`、`yoco_equiv.py:474`）+ `KEEP=140` + `EXTRA_SIZE=128`（`environ.py:164`）→ kvstore 环 **402**。140 相对 134 只多占 6 槽/层（3,168 B/层，20 层共 63,360 B/seq），但 EXTRA_SIZE 那 128 槽才是内存上的大头。
- 疑点：`PREFILL_STORE_EXTRA_SIZE=128` 与 `KEEP_TOKENS` 两个额外量是否语义重叠（都在"多存一段再重排"），代码未解释。

### #4 落盘量 = 20 × 134 × 528 B，与 L 和 KEEP 都无关
- 证据：`deepinfer/worker/cache_engine.py:300-306`
  > `self.state_size_to_disk = self.swa_store_total_bytes_inference_pad`（= 非 dropped 层数 × 推理环字节）
- 证据：`doc/inference_deployment_qa.md:169`
  > "Only `state_size_to_disk` = 20 layers × 134 tokens × 528 B = 1,415,040 B per sequence goes into the store."
- 等级：A（代码结构）+ B（数字）；算术复核：`20*134*528 = 1,415,040`，`134*528 = 70,752`，可被 32 整除（32 B 对齐）。
- 意味着什么：**重放长度不改变落盘体积**。KEEP 从 134 调到 140 只影响 kvstore 环与 query 行数，`state_size_to_disk` 仍是 134 口径。书里若写"140 决定落盘大小"是错的。
- 疑点：无。

### #5 decoder 的 SWA 不落盘，只留在 HBM 的 dropped region
- 证据：`deepinfer/worker/cache_engine.py:221-241, 265-306`（`_is_dropped_swa_layer`：`layer_id >= n_prefill_layer`，`:113-116`；`swa_store_num_layer` 只数非 dropped 层，`:266-270`）
  > `[stored region: SWA]` / `[dropped region: YOCO consumer SWA | compression state]`（`:229-231`）
- 证据：`doc/inference_deployment_qa.md:163`
  > "the stored layers are the first 20."
- 等级：A
- 意味着什么：跨请求只能靠"块状 KV + 重算 SWA"恢复；decoder 的 SWA 连落盘都不做，所以它的重放**只服务本次 decode**，不进 prefix cache（`doc:187`）。
- 疑点：40 层全有 SWA（R02a #19），dropped 20 层仍占 HBM，为什么不为 decoder 单独缩容？R02a #20 已提同一疑点。

### #6 可见域的代码表示：每行两个区间（pivots），不是布尔 mask
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:480-486, 500-520`
  > `X[0 : P[1]] = P[0] + arange(P[1])` / `X[P[1] : P[1]+P[3]] = P[2] + arange(P[3])` / `X[P[1]+P[3] : W] = -1`
  > `int32_t abs_lo = std::max(pos - window_size + 1, valid_start); if (abs_lo < 0) abs_lo = 0;`
- 等级：A
- 意味着什么：SWA 的"因果+窗口"被压成 4 个 int：`[history 起点, history 长度, 本 chunk 起点, 本 chunk 长度]`。section1 指向 state ring，section2 指向本 chunk 新算的 KV；剩下的列填 −1。行内可见 key 数 ≤ window。
- 疑点：无。

### #7 YoCo 截断的可见域：在 #6 之上再与"末尾矩形"求交
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:139-168`（`yoco_swa_visibility` 文档注释）
  > "返回当前 seq yoco decoder 截断后的可见区间，在这个 seq 对应的原始 swa 区间中的下标。"
  > `rect_lo`：用来截断的矩形下界（绝对位置，= data_len − keep_tokens）
- 证据：`csrc/.../prepare_tensor_input_v4.cpp:222-232`
  > `// 逻辑保留区间 = [max(0, data_len - keep_tokens), data_len) ∩ [q_start, q_end)` … `clip_prefix = compact ? clamp(data_len - keep_tokens - q_start, ...)`
- 等级：A
- 意味着什么：mask 与 compact 共享**同一套可见域**（滑窗 ∩ 矩形），区别只在"矩形外的 query 行是否物理删除"。这就是两种模式能等价的根据。
- 疑点：无。

### #8 `max(s, i−127)` 的精确落点
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:149-168`（`abs_lo = clamp(max(pos − window + 1, rect_lo, valid_start), 0, pos+1)`；section2 `k_lo = max(abs_lo − q_start, 0)`，`new_len = j − k_lo + 1`）；yoco 分支的调用与枢轴写入在 `:255-281`，无 yoco 的主路径在 `:499-522`
- 等级：A
- 意味着什么：`i−127` 的 127 就是 `window−1`；`s` 在实现里是 `max(rect_lo, valid_start)` 两个下界之一（rect_lo = data_len − keep_tokens 是截断起点，valid_start 是 zero-state 起点）。两个下界共用一套 clamp 逻辑。
- 疑点：无。

### #9 pivots → 稠密 index：`expand_swa_index`
- 证据：`deepinfer/model_executor/layers/tilelang_ops/swa_index.py:10-44`
  > `"""Expand two KV segments per token into dense prefill SWA indices."""`；参考实现 `expand_swa_index_ref`：`output = torch.where(arange < pivots[:,1], pivots[:,0]+arange, -1)`，第二段同理，其余 −1
- 证据：`deepinfer/model_executor/input_metadata/input_metadata.py:336-341`
  > `"""Expand pivots P into a dense prefill SWA index (indices into the gathered kv buffer, -1 = masked)"""` … "Used for both the full index and the YoCo-windowed index (which only differ in how P was built in prepare_prompt_v4)."
- 等级：A
- 意味着什么：整条链路是"pivots（4 int/行）→ [total_q, window] int32 index → 稀疏 attention kernel 按 index gather"。mask 不占额外显存，宽度恒为 window，与 KEEP 无关。
- 疑点：无。

### #10 prefill attention 直接吃 index
- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:603-625`
  > `kv_index = kv_swa_full_index` … `context, _ = prefill_attn_varlen_sparse(q_b, kv, kv_index, self.softmax_scale, attn_sink=..., index_length=dense_index_len, ...)`
- 等级：A
- 意味着什么："只与 max(s,i−127)..i 计算"不是靠 mask 掉大矩阵，而是把每行的 key 列表压到 ≤128 列；省的是 attention 的 key-pair 数，不是 query 数。
- 疑点：无。

### #11 mask vs compact：切 metadata 视图 vs 切行
- 证据：`deepinfer/model_executor/models/deepseek.py:191-210`
  > `if input_metadata.is_prompt and i == self.yoco_boundary_layer_idx:` … compact → `consumer_metadata = input_metadata.get_consumer_view()`；否则 `input_metadata = input_metadata.get_mask_view()`
- 证据：`deepinfer/model_executor/input_metadata/yoco.py:86-120`（两个 view 的字段替换：`kv_swa_full_index_pivots = yoco_mask_swa_index_pivots` / `yoco_consumer_swa_index_pivots`，并清空 `kv_swa_full_index`）
- 等级：A
- 意味着什么：mask 模式"看得少但算得多"（所有 query 行照跑），compact 模式"看得少且算得少"（行被删）。`examples/benchmark.py:46-49` 明确写 mask "consumer 层照常算全部 token，不省计算。用于数值对齐/排查"。
- 疑点：无。

### #12 compact 真的删行：`compact_yoco_boundary` + HCLazy gather
- 证据：`deepinfer/model_executor/models/v4_modules/yoco.py:55-76`
  > `"""x must hold per-token values along the first dimension. Transform it to obtain the new x after compact + [cp]."""` … `return gather(x, input_metadata.yoco_consumer_query_indices)`
- 证据：同文件 `:79-90`（HCLazy 的 4 个 per-token 字段同步 gather）、`:93-123`（`compact_yoco_boundary` 同时处理 hidden/residual/topk/candidate idx，并作废 `candidate_mqa_lens`）
- 等级：A
- 意味着什么：compact 的代价是边界处一次 gather + 派生缓存重建（`:120-122` 注释："Candidate-block query row layout changed … derived caches must be rebuilt"）。
- 疑点：`gather` 是拷贝，140 行的 gather 与 L 无关，成本可忽略——但代码没有量化。

### #13 边界层仍做全长度工作（这是"减半"不到 50% 的原因之一）
- 证据：`deepinfer/model_executor/models/decoder_layer.py:137-141`
  > `compact_inputs, full_inputs = compact_yoco_boundary(...)` / `cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]`
- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:672-709`
  > `"""Update main/index KV with the full-length hidden, for the compact query's forward to use."""` … `projected_kv = self.shared_hidden_kv_proj(full_hidden)` … `self._store_main_kv(...)`
- 等级：A
- 意味着什么：第一个 consumer 层（layer 20）对**全 L 行**做 attn_norm + `shared_hidden_kv_proj` + 写 KV；只有 layer 20 自己的 attention 与 21–39 的全部计算被裁到 140 行。所以"decoder 半边省 98.6%"只在 21–39 成立，20 层仍付 L 行的投影钱。
- 疑点：`flops_estimator.hpp:116-120` 用一个 `boundary` 项把这笔钱记在 consumer 侧（`boundary = compact && n_prefill_layer < n_layer ? pre : 0`），但只算 HC 的 pre，不含 `shared_hidden_kv_proj`（它被记在 `full_attn_per_row`，见 `:204-205`）。

### #14 为什么 decoder 可以只跑 140 行：块状 KV 来自共享隐状态
- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:279-290`（`yoco_swa_share_hidden` 为真时 `kv_swa_proj = ReplicatedLinear(cfg.hidden_size, self.qk_head_dim, ...)`；否则与 Q 融合）
- 证据：`deepinfer/model_executor/models/v4_modules/yoco.py:24-52`（`yoco_shared_hidden_required`：两种消费点——SWA 从 shared hidden 投影；packed-KV source 层用 shared hidden 算新压缩 KV）
- 等级：A
- 意味着什么：decoder 的块状 KV 不需要逐层逐 token 的 decoder 隐状态，只需要 encoder 末层的 shared hidden → 一次投影覆盖全 L 行；decoder 逐层独有的只有 SWA KV，而 SWA 只需要末尾窗口。
- 疑点：代码默认 `cfg.get("yoco_swa_share_hidden", True)`（`yoco.py:39`、`attn.py:249`），但 V4.1 的配置显式设成 `False`（`deepinfer/hugging_face/config.py:47`: `"yoco_swa_share_hidden": False`），即各层 kv_swa 由本层 hidden 投影（R02b #9 同结论）。两种情况下"只跑末尾 K 行"的结论不变，但共享与否影响显存/算力细节。

### #15 FLOPs 估算器里 consumer 的 query range
- 证据：`csrc/engine/flops_estimator.hpp:259-278`
  > `const int cutoff = full_len - keep_tokens;` / `if (compact) consumer.begin = std::clamp(cutoff, start, k - 1);` / `const int swa_start = std::max(swa_valid_start, cutoff);`
- 证据：`csrc/engine/flops_estimator.hpp:281-316`（`full_rows * producer_non_attn_per_row + consumer_rows * consumer_non_attn_per_row`；SWA 用 `full_swa_coef * full.swa_pairs + consumer_swa_coef * consumer.swa_pairs`）
- 等级：A
- 意味着什么：调度器算负载时用的就是"encoder 全行 + consumer 末尾 K 行"，并且**全被裁掉的 chunk 仍保留 1 行**（`clamp(..., k-1)`）。这是全仓库对"有界重放"最精确的形式化。
- 疑点：无。

### #16 官方测试把这条语义钉死
- 证据：`tests/engine/test_flops_estimator.py:24, 52-58`
  > `KEEP_TOKENS = 140` … `queries = positions[positions > cutoff]` … `if len(queries) == 0: queries = positions[-1:]  # Fully clipped chunks still execute one row.`
- 证据：`:148-156`（`mode ∈ {off, mask, compact}` × `{prefill, decode}`，与逐层逐 query 的参考实现逐值相等）
- 等级：A
- 意味着什么：`mask` 与 `compact` 在估算器里**行数相同**（都只算 `> cutoff` 的行），差别只在 compact 才真的少算；测试也确认 mask 不省 FLOPs 的建模。
- 疑点：测试依赖真实模型 metadata（`DEFAULT_MODEL_NAME`），本机无权重跑不了。

### #17 encoder 侧重放的接收端：zero-state + 零填充
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:362-389`
  > `padding_len = std::clamp(p_seq->swa_valid_start - history_begin, 0, window_size)` / `builder.append_zero_fill(padding_len)` / `builder.append_ring_range(p_seq->state_id, valid_begin, history_end)`
- 证据：`csrc/engine/sequence.hpp:204-226`（`hit_tokens`、`swa_valid_start`，`reset_for_recompute()` 把 `swa_valid_start` 清零）
- 证据：`csrc/engine/pybind_sequence.cpp:118, 132`（`cached_hit_tokens`、`swa_valid_start` 都是可写属性）
- 等级：A
- 意味着什么：开源仓库只提供"重放接收端"——把 `[0, valid_start)` 的 SWA state 当零、可见窗口左界 clamp 到 `valid_start`。**"只重放命中前缀的最后 128 个 token"这个决策不在本仓库**：命中长度由 Entrypoint 通过 `cached_hit_tokens` / `swa_valid_start` 注入（`doc:191` 的命中策略）。
- 疑点：`doc:185` 说"重放段只重建 SWA KV，块状 KV 原样复用、既不重算也不覆盖"，代码里确实看不到对该段块状 KV 的写（`_store_main_kv` 只对本次 forward 的新 token 调用），但没有显式 guard；若 `swa_valid_start` 设错会静默写坏。属于 E 级风险推断。

### #18 命中策略（文档口径）
- 证据：`doc/inference_deployment_qa.md:191`
  > "the maximum hit length for an input of length L is at most L − 134 (128-token window + 6 tokens of speculative backtracking). Under that cap, the block KV cache is queried first … If an SWA KV cache exists in the range L1 − 128 to L1, the hit length is L1; otherwise it is L1 − 128"
- 等级：B（文档），A（`swa_valid_start` 机制）
- 意味着什么：134 在这里是**命中上限的安全边界**，与 KEEP=140 无关；两者相差 6，正好是"再退一个回退预算"。
- 疑点：文档没写 `L − 134` 与 `KEEP=140` 同时存在时以谁为准；代码只校验 `KEEP ≥ 134`，不校验命中长度。

### #19 `yoco_equiv` L1：metadata 层等价 + 独立解析参考
- 证据：`examples/yoco/yoco_equiv.py:79-119`（10 个场景矩阵：`N<keep`、`N==keep`、单 chunk 跨界、chunk 全截断、矩形落在 chunk 起点/内部、矩形切进历史窗口、混长同批）
- 证据：`:116-119` 参考式 `lo = max(pos - window + 1, spec.N - keep_tokens, 0)`；`:335-371` 三条断言：可见域 == 参考、单行 key 数 ≤ keep_tokens（compact 侧 ≤ window）、store slot 规则
- 证据：`:447-455`（两模式各生成一份 payload，`_compare_mask` 后打印"输出等价"或 `SystemExit`）
- 等级：A
- 意味着什么：L1 不只是 A/B 对比，**每个模式都先与独立解析式对齐**（`:348-352`）。这比"两边一样"强：两边同错会被参考式抓住。
- 疑点：L1 只覆盖可见域/store slot，不覆盖数值；且需要 GPU（pinned memory，`:376-377`）。

### #20 `yoco_equiv` L2：端到端 logprob 逐位比较
- 证据：`examples/yoco/yoco_equiv.py:482-543`（`LogpConfig(keep_tokens=140, window=128, ep=8, cp=1, chunk=2048, top_logprobs=8)`；两次运行仅 `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` 可不同，`:463`）
- 证据：`:502-531`（长度矩阵 64 / 140 / 512 / 2×2048+140 / +60 / 3×2048+7）；`:672-695`（`golden_token_ids` teacher forcing）；`:864-937`（逐 step 比较 `token_id`/`logprob`/`top_ids`/`top_logprobs`/`confidence`，任一不等即 `first_bad`，并打印 rmse/max_abs/kl）
- 等级：A
- 意味着什么：L2 验证的是**两种截断实现的一致性**，用固定轨迹消除采样噪声；MTP 打开时还把 confidence 当"main cache 内容的 bitwise 探针"（`:903-906`）。
- 疑点：`_compare_logp` 没有容差参数，是严格不等比较（`:899-906`）——等价性要求 bitwise，工程上很脆（kernel 版本/并行度变化都可能破）。L2 需要 8 卡 + ckpt，本地无法复核。

### #21 等价性测试**不**验证"截断 vs 不截断"
- 证据：`examples/yoco/yoco_equiv.py:536`（`mode = "off" if keep_tokens == 0 else ("compact" if compact else "mask")`）+ `:796-797`（keep=0 时"两种模式都不走 yoco 路径，结果必须 bitwise 相同"）
- 等级：A
- 意味着什么：keep=0 只能验证 harness 自身的确定性，不能当"近似无损"的证据。仓库里**没有任何**"truncation on vs off"的数值对比脚本。
- 疑点：这是全书最容易被误读的一点：`yoco_equiv` 通过 ≠ 近似无损。

### #22 精度影响：官方只有定性表述
- 证据（B）：arXiv 2609.19969 §3.2.2 / 摘要
  > "we introduce SWA Bounded Replay, which approximately reconstructs the required SWA KV states by replaying only the most recent n_win tokens. Our experiments show that this incurs only negligible performance degradation."
  > "By design, the reconstructed decoder SWA KV is not mathematically equivalent to that from a full decoder forward pass. Also, we find that this strategy has only a negligible impact on response quality. For added safety, we additionally simulate the same replay during post-training for train-aware adaptation."
- 等级：B
- 意味着什么：报告没有给 PPL/KL/评测分数差值；并且明确说**后训练阶段模拟过该重放**——即"无损"部分是训练换来的，不是纯推理近似。
- 疑点：`train-aware adaptation` 的强度（多少比例的样本、哪些阶段）报告未展开；第三方 checkpoint 若没做同样适配，结论不能直接外推。

### #23 第三方实测：GSM8K/GPQA 在 1~1.5 个标准误内
- 证据（C）：vLLM 博客《DeepSeek-V4.1-Flash on vLLM》(2026-10-07)
  > "We confirmed this in vLLM on benchmarks including GSM8K and GPQA and observed no meaningful accuracy difference (gaps within about 1.5 standard errors)."
- 证据（C）：vLLM PR #58132（Decoder-side SWA bounded replay for DeepSeek-V4.1）
  > GSM8K 5-shot, n=1319, 两次 pass；TP4：on 0.9249 / 0.9234，off 0.9303 / 0.9249；DEP4：on 0.9333 / 0.9287，off 0.9257 / 0.9181；`one standard error is 0.0072`
- 等级：C
- 意味着什么：最大差值 0.0054（< 1 SE），且符号不一致（有时 on 更高）——支持"无明显差异"，但**不是**"零差异"。评测只覆盖 GSM8K 一项给到了逐点数字。
- 疑点：PR 是 vLLM 的实现，不是 DeepInfer 的；两者的裁剪层号/回退预算不同（vLLM 用 128，DeepInfer 默认 140）。

### #24 性能：官方说"减半"，第三方测得 25–45% TTFT
- 证据（B）：arXiv 2609.19969 §3.2（CED 复杂度）
  > "for a sequence length N ≫ n_win, CED reduces the prefill complexity from O(NL) to O(NL/2 + n_win × L/2) ≈ O(NL/2), effectively halving the overall computation."
- 证据（C）：vLLM PR #58132 性能段（GB200，V4.1-Flash）
  > "From 16K up, prefill TTFT drops 25-36% on TP4 and 37-45% on DEP4 at any concurrency"；"Decoder replay with CUDA graphs cuts prefill computation time by 30–40%."
- 等级：B / C
- 意味着什么：FLOPs 口径 ≈ 50%，墙钟口径 25–45%（CUDA graph 口径 30–40%；launch overhead、MoE 不均衡、graph break 等）。书里混用这两个数会出错。
- 疑点：DeepInfer 仓库自己没有任何 prefill 加速比数字（`README.md:77-83` 只给图，图上是吞吐-延迟曲线，没有文字倍数）。

### #25 仓库里的默认值全景
- 证据：`examples/benchmark.py:35-44`（`--yoco_clip` 裸给 = `const=140`，0 = off）；`examples/sweep/run_sweep.sh:46`（"YoCo 截断固定取 benchmark.py 的默认 140，不作为扫描维度"）；`README.md:25`（"`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` for the default V4.1 settings"）；`examples/yoco/yoco_equiv.py:485, 1008`
- 等级：A
- 意味着什么：140 是**示例与默认部署口径**，不是架构常量；134 才是架构常量（窗口+回退）。
- 疑点：为什么选 140 而不是 134，仓库里没有一行注释解释（见 Q4）。

### #26 开启截断会关掉 prompt_logprobs
- 证据：`deepinfer/engine/llm_engine.py:529-533`
  > "prompt_logprobs is forced to False under YoCo prefill truncation … compaction cannot compute clipped rows; mask returns degenerate logp for pre-rectangle tokens"
- 等级：A
- 意味着什么：截断的语义边界很清楚——矩形外的 token 没有"正确的"输出，因此任何需要全位置隐状态的功能（prompt logprobs）必须关。评测/打分类负载要注意。
- 疑点：其他依赖全位置隐状态的功能（如 DFlash target、Engram）用别的方式约束（`deepseek.py:126-135`：DFlash target 必须全在边界同侧；consumer 半区不允许 Engram 层）。

## 3. 必答问题逐条回答

### Q1. 用 L = 10,000 讲清"精确重建 vs 有界重放"，省多少显存流量与算力

**参数**（全部 A 级）：W = 128，B = 6，环 = 134，KEEP = 140，40 层（20 encoder + 20 decoder），SWA 528 B/token/层，KEEP 只作用于 consumer 层。

**(a) 精确重建到底要算什么**

- 冷 prefill 且不截断：40 层 × 10,000 行 = **400,000 行·层**。这是"朴素全量"基线。
- 只重建 decoder 的 SWA：20 层 × 10,000 行 = **200,000 行·层**。因为 SWA 依赖逐层累积（第 k 层的窗口依赖它**下面**那层的隐状态），要拿到最后一层末尾 128 个位置的状态，最省也得沿"依赖金字塔"回放：第 k 层要 128·k 行 → 阶梯总和 `128×(1+…+20) = 26,880 行·层`；若偷懒按"底层 2,560 行 × 20 层"整体跑，则是 **51,200 行·层**。报告写的 `L × n_win = 20 × 128 = 2,560` 指的是**底层需要的历史深度**，不是"2,560 行 × 20 层"。
- 常驻/写回：环只有 134 槽，所以"精确重建"不增加**常驻**体积（每半边 20 层 × 134 槽 × 528 B = 1,415,040 B/seq，与 L 无关），增加的是**写流量**：200,000 次 token-层写 × 528 B = **100.7 MiB**（同一槽被反复覆盖 ~74 次）。

**(b) 有界重放算什么**

- 只把末尾 KEEP=140 行送进 consumer 层：20 × 140 = **2,800 行·层**（`test_flops_estimator.py:52-58` 的 `positions > cutoff`）。
- 写流量：2,800 × 528 B = **1.41 MiB**。
- 每行可见 key：`pivots` 里 section1+section2 ≤ 128，且起点被 `max(rect_lo, valid_start, i−127)` 夹住（`prepare_tensor_input_v4.cpp:149-168`）。

**(c) 数字对比（算术均已在工具里复核）**

| 口径（L = 10,000） | 精确 | 有界重放（KEEP=140） | 节省 |
|---|---|---|---|
| 冷 prefill 总行·层（40 层） | 400,000 | 202,800（200,000 + 2,800） | **49.3%**（≈ 文档"减半"） |
| decoder 半边行·层 | 200,000 | 2,800 | **98.6%**（71.4×） |
| decoder SWA 写流量 | 100.7 MiB | 1.41 MiB | **99.3 MiB**（98.6%） |
| decoder 半边 SWA key-pair（20 层合计） | 25,437,440 | 358,400 | **98.6%** |
| encoder 有命中前缀时（hit=9,860 + 后缀 140） | 200,000 行·层 | (128+140)×20 = 5,360 | **97.3%** |
| 常驻 SWA（state_size_to_disk） | — | 1,415,040 B/seq | 与 L 无关 |

- key-pair 算法：全 10,000 行每层 `Σ min(i,128) = 1,271,872`；140 行 × 128 = 17,920；×20 层。
- 每 pair 的 FLOPs 系数：`core_coef = BF16 × heads × 2 × head_dim`（`flops_estimator.hpp:214`），即 QK+AV 两次 GEMM 共 `4·heads·head_dim` FLOPs/pair。没有 head 数就不硬算绝对值。
- 一句话：**"省一半"来自 encoder 200,000 / decoder 2,800 的行数对比；"省 98.6%"来自 decoder 内部 200,000 → 2,800。两个百分比不是同一个分母。**

### Q2. "位置 i 的 query 只与 max(s, i−127) 到 i 的 key 计算"在代码中如何实现

四步，全部 A 级：

1. **算每行两个区间**：`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:149-168`（`yoco_swa_visibility`）与 `:499-522`（无 yoco 的主路径）——
   `abs_lo = max(pos − window_size + 1, rect_lo, valid_start)`，再 clamp 到 `≥0`。
   于是 `rect_lo = data_len − keep_tokens`（= s）和 `valid_start`（zero-state 起点）与 `i−127` 三者取最大，正是 `max(s, i−127)`。
2. **拆成两段**：`section1` = 历史（state ring）`[abs_lo, q_start)`，`start_slot = min(abs_lo, q_start) − (q_start − window)`；`section2` = 本 chunk `[max(abs_lo, q_start), pos]`，`new_len = j − k_lo + 1`。两段长度和 ≤ window。
3. **写 pivots**：`[sec1_start + start_slot, W − start_slot, sec2_start + k_lo, j − k_lo + 1]`（主路径 `:516-522`，yoco 分支 `:275-280`），即"4 个 int/行"。
4. **展开成 index 并送 kernel**：`expand_swa_index`（`tilelang_ops/swa_index.py:10-44`）把 pivots 展开成 `[total_q, window]` 的 int32，空列填 −1；`attn.py:603-625` 把它作为 `kv_index` 交给 `prefill_attn_varlen_sparse`，kernel 按 index gather，−1 列不参与。

补充：**行被完全裁掉时**（compact 且整 chunk 在矩形之前），pivots 长度为 0 → 整行 −1；compact 更彻底，直接把这些行从 metadata 里删掉，只留 1 行占位（`flops_estimator.hpp:270-272`、`prepare_tensor_input_v4.cpp:222-230` 的 `clip_prefix`）。

### Q3. encoder 有界重放 vs decoder 有界重放，各自在代码哪里

| 维度 | encoder SWA 有界重放 | decoder SWA 有界重放（YOCO 截断） |
|---|---|---|
| 触发量 | `swa_valid_start`（zero-state 下界） | `KEEP_TOKENS` + `TRUNCATION_MODE` |
| 核心代码 | `prepare_tensor_input_v4.cpp:362-389`（`make_swa_block_load_plan`：`append_zero_fill(padding_len)` + `append_ring_range`）、`:499-509`（`abs_lo` 含 `valid_start`） | `prepare_tensor_input_v4.cpp:203-346`（`fill_yoco_truncation_metadata`）、`:348-360`（`update_yoco_prompt_v4`）；Python 侧 `input_metadata/yoco.py:94-120`（`get_consumer_view`）、`models/v4_modules/yoco.py:55-123`（`compact_yoco_boundary`）、`models/deepseek.py:191-210`（边界切视图） |
| 省的是什么 | 命中前缀的重算行数（只重放命中前缀末尾 128 + 未缓存后缀），块状 KV 复用 | 整段 prompt 在 decoder 半边的行数（只算末尾 KEEP 行），块状 KV 由边界层一次性投影 |
| 谁决定长度 | **不在本仓库**：Entrypoint 通过 `cached_hit_tokens` / `swa_valid_start` 注入（`pybind_sequence.cpp:118,132`；`sequence.hpp:204-226`）；策略见 `doc:191` | 本仓库：`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`（`environ.py:191`）+ 硬校验（`model_runner.py:107-113`） |
| 影响落盘 | 命中前缀的 SWA 缺失 → 回退一个 128 块重放；不改变 `state_size_to_disk` | decoder SWA 本来就不落盘（`cache_engine.py:113-116`） |
| 精度性质 | 近似：重放段首部的行看不到 s 之前的 key | 近似：同上，且只在 decoder 半边 |

一句话区别：**encoder 是"状态缺失 → 补一小段状态"（zero-fill + clamp），decoder 是"层根本不必跑全量 → 删行"（compact）**。两者共用同一个可见域公式（`abs_lo` 里同时有 `rect_lo` 和 `valid_start`），所以代码只有一份 `yoco_swa_visibility`。

### Q4. KEEP_TOKENS=140 与 134（128+6）、"SWA 窗口 + 最大回退长度"的关系

四个数，四个不同对象：

| 数 | 含义 | 出处 | 谁在用它 |
|---|---|---|---|
| **128** | SWA 窗口 W（`sliding_window`） | `hugging_face/config.py:284`（`window_size=(text["sliding_window"], 0)`）；`doc:183` | 可见域宽度、`pivots` 展开宽度 |
| **6** | 最大回退 B = `nextn + 1`（默认 5+1） | `environ.py:129-132` | ring 多留、命中上限、KEEP 下限 |
| **134** | W + B = **推理环**大小；也是每层落盘 token 数、命中上限 `L − 134` | `cache_engine.py:88-94`；`doc:169`；`doc:191` | `state_size_to_disk` = 20×134×528 = 1,415,040 B |
| **140** | **示例/默认的 KEEP_TOKENS**（重放多少末尾 query 行） | `benchmark.py:39`（`const=140`）；`yoco_equiv.py:485,1008`；`README.md:25` | consumer 层的 query 行数；kvstore 环额外 +140 槽 |

关系：

- **硬约束是不等式**：`KEEP ≥ W + B = 134`，三处校验（`model_runner.py:110-113`、`prepare_tensor_input_v4.cpp:355-357`、`yoco_equiv.py:793-795`）。140 满足，且富余 6。
- **140 − 134 = 6 = 恰好再一个回退预算**。所以 140 = 128 + 2×6。这是算术事实；"为什么是 2 倍而不是 1 倍"仓库没有任何注释，属于**未确认**（E 级推测：留一轮余量，或有历史原因）。
- **140 不影响落盘**：`state_size_to_disk` 用推理环（134）算（`cache_engine.py:265-306`，赋值在 `:306`）。140 只让 kvstore 环从 396 变 402（+6 槽/层 = 3,168 B/层，20 层 63,360 B/seq）。
- **文档里的"128"是概念口径，代码里的"140"是默认口径**：`doc:187` 说"recomputes only the last 128 tokens"，`README.md:25` 说 140。二者不矛盾（128 是最小窗口，140 是默认保留量），但**写书时必须点明 140 ≥ 128 + 6，而不是"重放 140 个 token 的窗口"**——窗口仍然是 128，140 是 query 行数。
- 另外 `doc:165` 的"6 extra positions"和 `doc:191` 的"6 tokens of speculative backtracking"是同一个 6 的两种用法（环容量 vs 命中回退），不要当成两个不同的 6。

### Q5. `yoco_equiv.py` 在验证什么等价性？方法与结论

**验证对象**：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=mask` 与 `=compact` 两种实现，在**同一份输入、同一套其他环境变量**下是否产出相同结果。**不是**"截断 vs 不截断"的精度验证（见 #21）。

**L1 `mask-compact`（metadata 层，CPU/单卡，`yoco_equiv.py:393-455`）**

- 构造 10 个形状场景（`build_mask_cases`，`:79-119`）：`N<keep`、`N==keep`、单 chunk 跨界、chunk 全被裁、矩形落在 chunk 起点（`clip_prefix==0`）/内部（`>0`）、矩形切进历史窗口、多请求混长同批。
- 对每个场景分别生成 mask / compact 的 `prepare_prompt_v4` 输出，扩成绝对 key 位置列表（`_expand_pivots`，`:122-137`）。
- 三条断言（`:335-371`）：① 每行可见 key == 解析参考 `max(pos−W+1, N−KEEP, 0)..pos`；② 单行 key 数 ≤ KEEP（compact 还要求 ≤ W）；③ store slot 规则（矩形内 + ring 尾段才写，compact 矩形外强制 −1）。
- 然后 `_compare_mask` 比对两份 payload（`:940-977`）。
- 结论字符串：`"[mask-compact] 结论：当前 repo 的 mask/compact 输出等价"`（`:455`），不等价则 `SystemExit`。

**L2 `logp` + `compare`（引擎层，8 卡 + V4.1 ckpt，`run_yoco_equiv_l2.sh`）**

- 两次 `create_llm`，除 mode 外环境逐键相同（`ENV_ALLOWED_DIFF`，`:463`；`_assert_env_comparable`，`:855-861`）；`DEEPINFER_BATCH_INVARIANT=1`、`SAMPLER_FORCE_TOPK=0`、EPLB 关闭（`:469-479`）。
- prompt 长度矩阵 64 / 140 / 512 / 4,236 / 4,296 / 6,151（`:522-529`），`golden_token_ids` 做 teacher forcing（`:682`），`output_len=8`、`top_logprobs=8`、固定 seed。
- 逐 step 严格比较 `token_id`、`logprob`、`top_ids`、`top_logprobs`、`confidence`（`:897-908`）；任一不等即失败，打印首个分歧 step 与 rmse/max_abs/kl（`:909-936`）；`--mtp N` 时 confidence 是 DFlash main cache 的 bitwise 探针（`:903-906`）。
- 结论字符串：`"[compare] 结论：mask/compact 等价（逐位一致）"`（`:997`）。

**测试结论**：

- 代码层面能确认的只是"这两个实现被设计成等价，且仓库提供了可执行的判定器"。**仓库里没有留下任何一次运行的输出**（无 `.github/`、无 `var/yoco_equiv.*` 产物、无 CI 配置）。
- 我无法运行：L1 也要一张可见 GPU（pinned 内存，`:376-377`），L2 要 8 卡和 V4.1 权重。因此本笔记**不声称测试通过**，只记录"判定器存在、断言内容如上"。
- 有一点值得强调：L1 的每个分支都先与**独立解析参考**对齐（`:348-352`），所以它能抓住"两边同错"；这比纯 A/B diff 强。

### Q6. 近似对精度的影响：有公开数据吗

有，但只有三档证据，且都不是 PPL/KL：

1. **官方定性（B）**：报告两处写"negligible performance degradation / negligible impact on response quality"，**没有数字、没有 PPL、没有 KL、没有评测分数差值**。同时明确"reconstructed decoder SWA KV is not mathematically equivalent"，并且**后训练阶段模拟过该重放**（train-aware adaptation）——这点很关键：无损有一部分是训练补偿的，不是纯推理技巧。
2. **第三方博客（C）**：vLLM 在 GSM8K + GPQA 上"no meaningful accuracy difference (gaps within about 1.5 standard errors)"，未列表。
3. **第三方 PR 数字（C）**：vLLM #58132，GSM8K 5-shot，n=1319，两次 pass：
   - TP4：on 0.9249 / 0.9234，off 0.9303 / 0.9249
   - DEP4：on 0.9333 / 0.9287，off 0.9257 / 0.9181
   - 1 SE = 0.0072 → 最大差 0.0054 < 1 SE，且方向不一致（有时 on 更高）。
4. **DeepInfer 仓库内**：没有任何 truncation on/off 的精度脚本或结果；`eval/run_eval.py` 支持 `--env KEY=VALUE` 注入 `DEEPINFER_YOCO_*`（可用于 AIME/GPQA/HLE 自测），但仓库没有发布跑分。

结论：**"近似无损"目前是 B+C 级证据（定性 + 单项 GSM8K 统计），没有 PPL/KL 曲线，也没有长上下文/多轮场景的公开数据**。报告自己提示要 stress-test cache-resumption 场景。书里不能写成"已验证无损"。

### Q7. 文档说 decoder 有界重放"几乎让 prefill 总计算量减半"，依赖什么前提

前提按重要性排序（每条给出处）：

1. **N ≫ n_win**。报告的原式：`O(NL) → O(NL/2 + n_win·L/2) ≈ O(NL/2)`（B）。按 `(N + KEEP) / 2N` 估算（N = 序列长度）：N=512 只省 36.3%，2,048 省 46.6%，8,192 省 49.2%，32,768 省 49.8%。短 prompt 不成立（vLLM 也观察到 1K 时 eager replay 反而 +3~+12%）。
2. **CED 的"半边"结构**：decoder 的块状 KV 必须来自 encoder 末层隐状态的**一次性投影**（`attn.py:672-709` 的 `shared_hidden_kv_proj`；`yoco.py:24-52` 的两种消费点）。如果 decoder 的 KV 依赖各层自己的全量隐状态，就省不掉。
3. **两半每行算力大致相当**（20/20 层、同宽度）。估算器把 producer/consumer 的 MLP/router/HC/Engram 系数分开累加（`flops_estimator.hpp:64-121`），只有量级相同时"减半"才成立。
4. **边界层仍付全行长度的钱**：layer 20 的 attn_norm + `shared_hidden_kv_proj` + 写 KV 都在 L 行上做（`decoder_layer.py:137-141`、`attn.py:672-709`），所以实际略小于 50%。
5. **只对"非缓存后缀"生效，且不含 prompt_logprobs 之类全位置需求**：截断开启后 `prompt_logprobs` 被强制关闭（`llm_engine.py:529-533`）；DFlash target 必须全在边界同侧、consumer 半区不允许 Engram（`deepseek.py:126-135`）。
6. **保留后缀必须 ≥ W + B**，否则 decode 第一步就缺 key（三处硬校验）；这条不影响"减半"，但影响"能不能用"。
7. **口径是 FLOPs，不是墙钟**：FLOPs 口径 ≈ 49.3%（本笔记 Q1 的算术）；vLLM 实测 prefill TTFT −25~−45%（TP4 −25~−36%，DEP4 −37~−45%，16K+），CUDA-graph 口径另有 30–40% 的说法（C）；DeepInfer 仓库没有公布任何 prefill 加速比数字。
8. **需要 checkpoint 配合**：报告说后训练模拟过重放（B）。第三方权重若未做同样适配，"几乎无损"不能直接继承。

## 4. 新发现的问题

1. **`doc:187` 的 2,560 口径混乱**：字面是"2,560 个 token 过 20 层"（= 51,200 行·层），但同一句又说"几乎减半"（对应 140 行 × 20 层）。报告原文是
   > "Exact reconstruction requires replaying the most recent L×n_win tokens, where L is the number of layers and n_win is the SWA window size."
   （= 20×128 = 2,560，指依赖深度/回放总量），vLLM 博客也写"replaying roughly L × 128 tokens"。书里应写"依赖金字塔深度约 2,560 个 token·层"，不要写"2,560 个 token 过 20 层"。
2. **`doc:187` 说"128"，README/默认说"140"**：文档没有解释二者关系，容易让读者以为窗口是 140。建议书里画一张"128 窗口 / 134 环 / 140 重放行"的三层图。
3. **`KEEP=140` 无推导**：代码/文档都没有"为什么是 140 而不是 134"。6 个槽的富余（63 KB/seq）不是性能问题，但作为默认值缺少理由，属于"含义未确认"。
4. **`PREFILL_STORE_EXTRA_SIZE=128` 与 `KEEP_TOKENS` 语义重叠**：两者都让 kvstore 环变大（402 vs 134），代码只说是"落盘粒度 128 的整倍数 + 锚点回退余量"，没有说明为什么需要同时存在。
5. **encoder 重放的"写保护"只是约定**：`doc:185` 说重放段"块状 KV 既不重算也不覆盖"，代码里没有任何 guard，只靠 `swa_valid_start` 正确注入。若上层传错，会静默写坏块状 KV——这是 E 级风险推断，值得在书里作为"实现契约"提一句。
6. **`yoco_equiv` L2 的 bitwise 要求过严**：`_compare_logp` 没有任何容差（`:899-906`），换 kernel/并行度就可能红。它适合做"实现一致性"门禁，不适合做"数值近似"证据。
7. **`mask` 模式的定位容易被误解**：它可见域与 compact 完全相同但不省算力（`benchmark.py:46-49`），仓库把它定位为"数值对齐/排查"工具。书里应明确它不是部署选项。
8. **consumer 半区的功能限制散落在三处**：`deepseek.py:126-135`（DFlash/Engram）、`llm_engine.py:529-533`（prompt_logprobs）、`attn.py:672-709`（全量 KV 投影）。没有一张统一的"截断兼容性矩阵"，这是书里可以做的一张表。
9. **`swa_valid_start` 与 `hit_tokens` 的关系没有单测**：`tests/` 里没有针对 zero-state 重放的用例（只有 `test_state_cache_ops.py` 测 ring 拷贝），encoder 重放的回归覆盖看起来弱于 decoder 侧。

## 5. 可用于图表的材料

**图 1：四个数字（128 / 134 / 140 / 402）**

| 层 | 值 | 作用 | 出处 |
|---|---|---|---|
| 窗口 W | 128 | attention 可见范围 | `config.py:284` |
| 回退 B | 6 | `nextn+1` | `environ.py:129-132` |
| 推理环 | 134 | `state_size_to_disk`、命中上限 L−134 | `cache_engine.py:88-94`、`doc:169,191` |
| 默认重放行 | 140 | consumer query 行数 | `benchmark.py:39`、`README.md:25` |
| kvstore 环 | 402 | 134 + 128(EXTRA) + 140(KEEP) | `cache_engine.py:101-110` |

**图 2：L=10,000 的行·层条形图**（400,000 / 202,800 / 200,000 / 2,800 / 5,360）

**图 3：依赖金字塔**（为什么"精确重建"要 2,560 深度）
- 顶层要 128 行 → 下一层要 256 → … → 底层要 2,560；阶梯总和 26,880 行·层；有界重放把每层都压到 140 行（2,800 行·层）。

**图 4：mask vs compact 流程**
- mask：全行保留 → 可见域收窄到 ≤128 列 → 矩形外行输出退化但无人读；
- compact：边界处 `gather(yoco_consumer_query_indices)` 删行 → 后续 20 层只跑 140 行 → 尾部 decode 用。

**表 A：命中策略（doc:191）**
- 上限 `L − 134` → 查块状 KV 得 L1（128 倍数）→ 查 `[L1−128, L1)` 的 SWA → 命中则 hit=L1，否则 hit=L1−128。

**表 B：精度证据一览**

| 来源 | 等级 | 覆盖 | 数字 |
|---|---|---|---|
| V4.1 报告 §3.2.2 | B | 定性 | "negligible"，无表 |
| vLLM 博客 | C | GSM8K + GPQA | 差 ≤ ~1.5 SE |
| vLLM PR #58132 | C | GSM8K 5-shot n=1319 | on 0.9249/0.9234 vs off 0.9303/0.9249（TP4）；SE=0.0072 |
| DeepInfer 仓库 | A | 仅 mask≡compact | 无 on/off 数据 |

**表 C：性能口径**

| 口径 | 值 | 条件 |
|---|---|---|
| FLOPs（本笔记算术） | 省 49.3% @ L=10,000 | 20/20 层、KEEP=140、忽略边界层全量投影 |
| 报告复杂度 | `O(NL) → O(NL/2)` | N ≫ n_win |
| vLLM TTFT | TP4 −25~−36%，DEP4 −37~−45%；CUDA-graph 口径 −30~−40% | GB200，16K+，prefix cache off |
| DeepInfer | 无公开数字 | — |
