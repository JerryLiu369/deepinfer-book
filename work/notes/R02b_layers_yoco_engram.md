# R02b CED 层结构 / YOCO 跨层复用 / mHC 超连接 / Engram 特征增强

## 0. 元信息
- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21，`work/versions.md` 锁定的基线）
- 查阅的主要来源（全部本地，除标注外）：
  - `deepinfer/hugging_face/config.py`
  - `deepinfer/model_executor/models/deepseek.py`
  - `deepinfer/model_executor/models/decoder_layer.py`
  - `deepinfer/model_executor/models/v4_modules/{cross_layer,yoco,hc,hc_lazy,hc_blocks,mhc_ops,mhc_modeling,attn,mla_attn,compress,engram,engram_metadata,engram_parameters,builder,heads}.py`
  - `deepinfer/worker/cache_engine.py`、`deepinfer/model_executor/model_metadata_utils.py`
  - `deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py`、`deepinfer/model_executor/layers/tilelang_ops/{gather_engram_tokens_zos.py,cuda/gather_engram_tokens_zos.py}`
  - `doc/inference_deployment_qa.md`（B 级，官方 Q&A）、`README.md`（B 级）
  - HF 模型卡 `deepseek-ai/DeepSeek-V4.1-Flash-1017/config.json`（B 级；仓库内**没有** config.json，所有层数/表规模由它给出）
- 注意：仓库里没有 `encoder` 这个词的架构用法（`grep -i "encoder"` 只命中 vision encoder / eval 编码器）。"encoder/decoder 层" 是**文档术语**，代码里的对应物是 `n_prefill_layer` 之前的层与 `is_yoco_consumer_layer` 之后的层。

## 1. 三个最重要的发现
1. V4.1-Flash 是 40 层 = 前 20 层 encoder + 后 20 层 decoder；两者**不是两套类**，而是同一个 `DeepseekDecoderLayer` 按 `n_prefill_layer` 分段，分段点由 `compress_ratios` 里第一个 `1` 的**下标**硬性决定（得 20）。
2. decoder 的块状 KV 只有一份：只有第 20 层真的做投影（`PackedWkv` 权重被拆出来变成 `shared_hidden_kv_proj`，输入是 encoder 末层的 attn-norm hidden），第 21–39 层不持有 KV 参数，靠 cache engine 按 `layer_kv_map` 别名复用第 20 层的 `ca_cache`。
3. Engram 是"超大稀疏 n-gram 查表 + 门控注入主干"，表规模按 fp8 算是两层约 1.57 TB，必须按 hash 跨 `engram_parallel_group` 拉取，所以它在 forward 里以 hook 形式延迟到层内；在 ZOS（CPU 落后一步）decode 下，n-gram 窗口无法在 CPU 侧拼装，这才需要专用的 `gather_engram_tokens_zos`。

## 2. 事实条目

### #1 层类型由 `compress_ratios` 逐层决定，只有三种
- 证据：`deepinfer/hugging_face/config.py:146-155`
  > `specs: list[Any] = [_win_spec(text) if ratio == 0 else _compress_spec(text, ratio, shared) for ratio in ratios[:n_layer]]`
- 证据：`config.py:99-108`（ratio==0 → `sparse_stage=0`，不设 `n_packed_heads`）；`config.py:111-120`（ratio!=0 → `n_packed_heads=num_attention_heads`，`kv_compress_ratio=ratio`，`index_topk`）
- 等级：A
- 意味着什么：读层结构先读 `compress_ratios`，不用猜；ratio 0 = 纯 SWA（无块状 KV），ratio 1/2 = 有块状 KV（C1/C2 压缩）。
- 疑点：无

### #2 `n_prefill_layer` 不是配置里直接写的，而是从 `compress_ratios` 推出来的
- 证据：`config.py:158-161`
  > `"""The first layer with ratio==1: it and everything before it form the prefill side, the layers after it are yoco consumer layers."""`
  > `return next(i for i, ratio in enumerate(ratios) if ratio == 1)`
- 等级：A
- 意味着什么：encoder/decoder 的分界是**模型权重自带的**，不是推理引擎的开关；引擎只是读出来。
- 疑点：如果将来某模型没有 ratio==1 的层，`next()` 会直接抛 `StopIteration`，没有兜底。

### #3 V4.1-Flash 实际是 40 层，前 20 层 encoder
- 证据：HF `config.json`（`text_config`）：`"num_hidden_layers": 40`；`compress_ratios` 前 40 项 = `[0,0]` + 18 个 `2` + 20 个 `1`（后 3 个 `0` 属于 3 个 DFlash 层，`num_nextn_predict_layers: 3`）
- 等级：B（数值）+ A（推导逻辑，见 #2）。用 `_n_prefill_layer` 复算：下标 20 是第一个 `1` → `n_prefill_layer=20`
- 意味着什么：encoder = layer 0–19（2 层纯 SWA + 18 层 C2），decoder = layer 20–39（全 C1）。与 `doc/inference_deployment_qa.md:163`「For V4.1, the stored layers are the first 20」和 `:187`「through 20 decoder layers」两处独立数字吻合。
- 疑点：无

### #4 encoder 层与 decoder 层共用同一个 layer 类，执行顺序也完全一样
- 证据：`deepinfer/model_executor/models/decoder_layer.py:48`（`class DeepseekDecoderLayer(HCLayerMixin)`）、`:143-173`
  > `hidden_states, hidden_states_bf16, residual = self.attn_norm_forward(...)`
  > `hidden_states = self.self_attn(...)` → `self.ffn_norm_forward(...)` → `self.mlp(...)`
- 等级：A
- 意味着什么："encoder/decoder" 是**数据流角色**的命名，不是两套算子栈；差异全部落在 `V4Attention` 的 KV 来源和 `n_packed_heads` 上。
- 疑点：`deepinfer/model_executor/models/deepseek.py:92-103` 用同一个 `DeepseekDecoderLayer` 列表推导构造全部层，没有任何按 `n_prefill_layer` 分流的构造分支。

### #5 decoder 的块状 KV 由 `shared_hidden_kv_proj` 单独投影，不走本层 hidden
- 证据：`deepinfer/model_executor/models/v4_modules/attn.py:345-353`
  > `if self.is_kv_source: self.PackedWkv = Compressor(...)`
  > `kv_weights = ([] if self.is_c1 else [attn_module.PackedWkv.Gate.weight]) + [attn_module.PackedWkv.A.weight]`
  > `if self.is_yoco_hidden_consumer_layer: shared_hidden_kv_weights = kv_weights`
  > `else: add_current_hidden_part("kv", kv_weights)`
- 证据：`attn.py:372-373`（`self.shared_hidden_kv_proj = make_projection(shared_hidden_kv_weights, use_fp32_output=not self.is_c1)`）
- 等级：A
- 意味着什么：YOCO 不是"共享 cache 指针"这么简单，它把 KV 投影的**输入张量**换掉了：从"本层 hidden"换成"跨层传入的 shared hidden"，因此这一层的 KV 与自己的 attention 输出无关，可以离线算。
- 疑点：`use_fp32_output=not self.is_c1` 说明 C2 的投影输出是 fp32（gate+latent 两段），C1 是 bf16 单段；见 #6。

### #6 decoder 层是 C1（不压缩），encoder 的 18 层是 C2（2:1 压缩）
- 证据：`deepinfer/model_executor/models/v4_modules/compress.py:75-76`
  > `if self.kv_compress_ratio not in (1, 2): raise ValueError("Only C1/C2 compression is supported")`
- 证据：`compress.py:89-96`
  > `if self.kv_compress_ratio == 1: assert projected_kv.dtype == torch.bfloat16 and state_cache is None; return projected_kv`
  > `score, latent = projected_kv.chunk(2, dim=-1)`（ratio=2 时拆成 gate 分数与 latent）
- 等级：A
- 意味着什么：C1 = 每 token 一条 512 维 latent（`kv_lora_rank 448 + qk_rope 64`），无 state、无 Gate 权重；C2 = 1024 维 fp32 输出 + `ca_state` 环形累加器做 2:1 压缩。所以 decoder 半区虽然"不压缩"，但全局只有一份 KV，总量仍远小于 encoder。
- 疑点：为什么 decoder 反而用不压缩的 C1？见 §4 问题 4。

### #7 第一个 consumer 层（20）的 shared hidden 就是自己的 attn-norm 输出
- 证据：`attn.py:465-479`
  > `if shared_hidden_bf16 is None and self.is_yoco_hidden_consumer_layer and self.PackedWkv is not None:`
  > `assert self.is_first_yoco_consumer_layer`
  > `return hidden_states_bf16`
- 证据：`attn.py:245-247`（`is_first_yoco_consumer_layer = ... and self.layer_idx == cfg.n_prefill_layer`）
- 等级：A
- 意味着什么：边界层不需要跨层搬运，省一次全量 hidden 的传递；其余 consumer 层才读 `cross_layer_runtime.shared_hidden`。
- 疑点：`hidden_states_bf16` 是"上一个子层的 post + 本层 pre mix + RMSNorm"之后的 bf16，并非 encoder 末层 MoE 的原始输出。文档说"projected from the last encoder layer's hidden state"是简化说法，精确说是**encoder 栈输出经过 HC pre-mix 与 RMSNorm 之后**的 hidden。

### #8 第 21–39 层不写 KV，靠 cache 别名复用第 20 层
- 证据：`deepinfer/model_executor/models/v4_modules/cross_layer.py:306-307`
  > `NOTE the main KV (ca_cache) of share-kv is aliased by the cache engine according to layer_kv_map; loaded_prefill_kv further reuses the temporary tensor decompressed from ca_cache.`
- 证据：`deepinfer/model_executor/models/v4_modules/mla_attn.py:252-269`（`self.PackedWkv = (...) if is_kv_source_layer else None`——非源层的 `n_packed_heads` 仍为 64，但不持有投影权重）
- 证据：`deepinfer/worker/cache_engine.py:165-171`（`_get_kv_mapped_layer_id`：返回值 != 自身即"本层复用了别层的 kv，不占用新的 kv/state 空间"）
- 等级：A
- 意味着什么：所谓"20 个 decoder 层共享 KV"在实现上是"20 层里只有 1 层有 KV 参数、1 份 cache 缓冲"。HF 配置 `kv_source_layer_ids: [2, 8, 14, 20]` 正是这个分段方案。
- 疑点：21–39 层仍然要各自算 SWA KV（见 #9），这部分不能别名。

### #9 decoder 的 SWA KV 不共享，按层各算一份（默认 `yoco_swa_share_hidden=False`）
- 证据：`attn.py:249`（`self.yoco_swa_share_hidden = self.use_swa and self.is_yoco_hidden_consumer_layer and cfg.get("yoco_swa_share_hidden", True)`）
- 证据：`deepinfer/hugging_face/config.py:47`（`_GPT_DEFAULTS` 里 `"yoco_swa_share_hidden": False`）
- 等级：A
- 意味着什么：V4.1 里 consumer 层的 `kv_swa` 仍由本层 hidden 投影（`attn.py:395-397` 的分支不会进），所以"decoder 只需重算最近 128 token 的 SWA"这条优化才有意义；也解释了 state cache 里为什么要把 consumer SWA 放进 dropped region（`cache_engine.py:221-241`）。
- 疑点：`yoco.py:39` 里同名配置的 `get` 默认值是 `True`，与 `_GPT_DEFAULTS` 的 `False` 不一致——只有显式写进 cfg 才安全，属于易踩的坑。

### #10 compact prefill 下，全长度 KV 由边界层的"全量 attn-norm"补出来
- 证据：`decoder_layer.py:134-142`
  > `compact_inputs, full_inputs = compact_yoco_boundary(hidden_states, residual, input_metadata, cross_layer_runtime, hc_enabled=self.hc_enabled)`
  > `cross_layer_runtime.consumer_full_hidden = self.attn_norm_forward(*full_inputs)[1]`
  > `hidden_states, residual = compact_inputs`
- 证据：`attn.py:692-708`（`_store_yoco_full_kv`：`projected_kv = self.shared_hidden_kv_proj(full_hidden)` → `_store_main_kv`）
- 等级：A
- 意味着什么：这是整条 YOCO 优化的"杠杆点"——只要在边界层对**全序列**做一次 5120×512 的投影，就能生成全上下文的 decoder 块状 KV，而 20 层 decoder 的 attention 只需跑保留的 140 个 token。
- 疑点：`consumer_full_hidden` 要为全序列多存一份 `[total_q, hc_mult, 5120]` bf16（`decoder_layer.py:140`），显存代价在长 prefill 下不小。

### #11 YOCO 截断有三档，V4.1 默认参数是 compact + 保留 140 token
- 证据：`deepinfer/environ.py:188-195`
  > `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE = EnvVar.choice(default="off", choices=("off", "mask", "compact"))`
  > `"mask 保留全部 query，仅收窄 SWA 可见域；compact 真正移除截断区 query，以减少 consumer 层计算"`
  > `必须满足 KEEP_TOKENS >= window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS`
- 证据：`README.md:25`（`MODE=compact` + `KEEP_TOKENS=140` 是 V4.1 默认；`sliding_window=128`，回溯 6 → 128+6=134 ≤ 140）
- 等级：A（机制）/ B（140 这个默认值）
- 意味着什么：`mask` 只裁可见域、不省算力；`compact` 才真正省。两个模式的 metadata 切换点在 `deepseek.py:191-207`。
- 疑点：无

### #12 `CrossLayerRuntime` 是"单槽位、按层序覆盖"，不做 layer_id 匹配
- 证据：`cross_layer.py:276-308`
  > `Each value is stored in a single slot: the sharing topology is segmented ..., layers execute in increasing layer_id, and the value a consumer needs is always the one written by the nearest source layer in front of it, so there is no need to record the source layer and match it again.`
- 证据：`cross_layer.py:289-304`（六类跨层值：`shared_hidden` / `pre_rope_kv` / `topk_indices` / `block_candidate_idxs` / `loaded_prefill_kv` / `loaded_prefill_index_kv`）
- 等级：A
- 意味着什么：跨层共享的运行时开销接近零（一个 dataclass，按引用逐层传），但正确性完全依赖"源层必须先于消费者执行"这一单调性假设。
- 疑点：任何流水并行/层重排都会破坏这个假设。

### #13 共享拓扑是"分段 + 三条 map 的细化链"
- 证据：`cross_layer.py:41`（`the segment granularities of the three maps form the chain index-topk ⊆ index-K ⊆ KV`）、`cross_layer.py:94-98`（同一不变量的实现注释）
- 证据：`cross_layer.py:169-180`（`_validate_segmented_map`：消费者必须指向前方最近的源层）、`182-190`（`_validate_map_refines`）、`191-197`（`_validate_index_kv_compatibility` 串起两级细化）
- 等级：A
- 意味着什么：`kv_source_layer_ids` / `index_source_layer_ids` 不是任意集合，是"单调分段"；写配置时错一个 id 会 fail-fast。
- 疑点：`config.py:62-65` 的 `_source_map` 用 `max(s for s in ids if s <= layer)` 生成，所以源列表**必须**单调，否则 `max` 会悄悄吞掉异常。

### #14 share-kv 不允许跨 YOCO 压缩边界
- 证据：`cross_layer.py:114-116`
  > `n_prefill_layer = cfg.get("n_prefill_layer", cfg.num_hidden_layers)`
  > `assert n_prefill_layer == cfg.num_hidden_layers or self.layer_kv_map[n_prefill_layer] == n_prefill_layer`
- 等级：A
- 意味着什么：第 20 层必须是 KV 源（self-reference）。这条断言把 "decoder 第一层是唯一投影层" 从约定升级成不变量；也是 `_store_yoco_full_kv` 能成立的前提。
- 疑点：无

### #15 mHC 把残差主干从 1 路变成 `hc_mult`=4 路
- 证据：`deepinfer/model_executor/models/v4_modules/hc.py:1-6`（模块 docstring）
  > `mHC runtime: mix the hc_mult-way residual trunk around every sub-layer, then fold it back.`
- 证据：`hc.py:57-69`（`the residual trunk is expanded into hc_mult paths, shaped [n, hc_mult, hidden_size]`）
- 证据：HF config `"hc_mult": 4`；`config.py:294-296`（`hc_norm_group_size = hc_mult * hidden_size` = 20480）
- 等级：A（机制）/ B（数值 4）
- 意味着什么：残差流不是 `[n, 5120]` 而是 `[n, 4, 5120]`，attention/FFN 的输入由 4 路加权混合得到（pre），输出再按 4×4 组合矩阵写回 4 路（post）。"超连接"= 把 ResNet 的单条残差高速路换成 4 条互相混合的路，容量换显存与带宽。
- 疑点：`mhc_mult` 只保证 4 可用（`mhc_modeling.py:48` 注释 "currently only 4 is guaranteed to work"）。

### #16 mHC 的每层代价：一个 24×20480 的 fp32 小 GEMM + sinkhorn 归一化
- 证据：`deepinfer/model_executor/models/v4_modules/hc_blocks.py:18-41`
  > `channels = (2 + m) * m`（m=4 → 24）
  > `self.hc_attn_fn = nn.Parameter(torch.zeros(channels, m * cfg.hidden_size, ...))`
- 证据：`mhc_ops.py:95-99`（`sinkhorn_normalize(x, repeat=10, eps=1e-6)`）、`mhc_modeling.py:81`（`sinkhorn_repeat` 来自 `hc_sinkhorn_iters`，V4.1=20）
- 等级：A / 数值为 E（我按 `24 × 4×5120 = 491,520 MAC/token/子层` 算；两层子层 × 40 层 ≈ 39.3M MAC/token，对比 MoE 单层约 `6×3×2304×5120 ≈ 212M`，约 0.5% 量级）
- 意味着什么：mHC 的算术开销很小，贵的是 fp32 中间量 + 每 token 的 4×4 组合矩阵要迭代 20 次 sinkhorn；它是"带宽/延迟"问题，不是 FLOPs 问题。
- 疑点：上面对 MoE 的估算是按 6 个 routed expert 全激活粗算，未计 shared expert 与 gate，只用于量级判断。

### #17 `hc_shifted` 默认 True：没有 HCHead，最后一层的 mix 在模型末尾补
- 证据：`config.py:295`（`hc_shifted=text.get("hc_shifted", True)`；HF config 里没有这个键 → True）
- 证据：`deepseek.py:107`（`self.hc_head = HCHead(head_module) if not self.hc_shifted else None`）
- 证据：`hc.py:340-358`（半层偏移：本子层用上一层算出的 mix，本层算出的 mix 留给下一层）；`deepseek.py:229-232`（末尾 `mhc_pre_apply_mix(hidden_states, residual.pre_layer_mix)` 折回单路）
- 等级：A
- 意味着什么：V4.1 的 HC 是"延迟半层"的：每个子层的输入 mix 是**上一个**子层算出来的，这打乱了"本层算 mix、本层用"的常规数据流，普通推理框架照抄会算错。
- 疑点：`HCHead` 这条分支在 V4.1 走不到（`hc.py:373` 直接 assert 不允许 shifted 建 head），属于为别的 checkpoint 保留的代码。

### #18 `mega_mhc` 把 post+pre+RMSNorm(+FP8 cast) 合成一个 kernel，残差靠 `HCLazyPostData` 延迟携带
- 证据：`hc.py:229-298`（`deep_gemm.mega_mhc(x, residual, prev_pre_mix, post_mix, comb_res_mix, fn=..., ...)` 一次产出 `new_residual / new_post_mix / new_comb_res_mix / y_bf16 / y_fp8`）
- 证据：`hc_lazy.py:14-25`（`HCLazyPostData`：`residual [1,n,hc_mult,H]`、`post_layer_mix`、`comb_res_mix`、`pre_layer_mix`）
- 证据：`hc.py:99-101`（`use_mega_mhc` 的启用条件：GPU、`hc_mult==4`、fn 列数能整除 `4*16*64`）
- 等级：A
- 意味着什么：残差加法的结果**不落成单路 hidden**，而是保留 4 路 + 三个 mix 系数，直到下一个子层才消费。对计算图的影响是：层与层之间的"残差 add"节点消失，被折进下一个 kernel 的 prologue；对 CUDA graph 友好（形状固定），但中间张量从 `[n,5120]` 变成 `[n,4,5120]`。
- 疑点：`mega_mhc` 的实现不在本仓库（`deep_gemm`），只能从调用签名推断融合范围。

### #19 Engram 在层内是"HC post 之后、pre 之前"的主干干预钩子
- 证据：`decoder_layer.py:111-118`
  > `# engram 作为 hc post 之后的主干干预钩子（见 HCLayerMixin.attn_norm_forward）`
  > `hc_post_intervention = lambda trunk: self.engram(trunk, embed_hook, getattr(input_metadata, "images_seq_mask", None))`
- 证据：`hc.py:184-186`（`When hc_post_intervention is not None (e.g. the engram layer) ... the mega_mhc fusion is not used (no hook can be inserted inside the kernel)`）、`hc.py:225-226`（`next_residual = hc_post_intervention(next_residual)`）
- 等级：A
- 意味着什么：Engram 不是"额外一层"，而是插在两个子层之间、改残差主干；**只要这层有 Engram，这层的 `mega_mhc` 融合就失效**，退化成 post / pre 分开调 kernel。这是一个用算子融合换模型结构的真实取舍。
- 疑点：V4.1 的 Engram 层是 1 和 14，都 < 20；`deepseek.py:136-138` 断言 compact 模式不支持 consumer 层的 Engram。

### #20 Engram 表规模：每层 384,006,168 行 × 2048 维，fp8 下约 786 GB/层
- 证据：HF config：`"engram_layer_ids": [1, 14]`、`"engram_num_embeddings": [384006168, 384016682]`、`"engram_max_ngram_size": 4`、`"engram_vocab_size": 16000000`、`"engram_n_heads": 8`、`"engram_head_dim": 256`、`"engram_compressed_vocab_size": 99092`
- 证据：`deepinfer/model_executor/models/v4_modules/engram_metadata.py:67-95`（每个 layer × 每个 ngram × 每个 head 取一个**互不重复的素数**当 vocab size，起始 `vocab_size-1`）；`:71`（表数 = `max_ngram_size - 1` = 3，即 2/3/4-gram）
- 等级：B（数值）+ A（构造逻辑）+ E（字节换算）
- 意味着什么：24 个素数之和正好 `384,006,168`（我复算过），`n_embed_per_ngram = 8×256 = 2048`，所以单层 `384006168×2048 B ≈ 786.4 GB`（fp8），两层约 **1.57 TB**。这不是"每卡放得下"的表，必须 `engram_parallel_group` 分片（`engram_parameters.py:19-32`，`num_per_part = padded_rows / engram_parallel_size`）。
- 疑点：`engram_num_embeddings` 第二个值 384,016,682 与"24 个素数之和"不等（第一个值恰好相等），推测是不同 layer 的素数池不共享导致的偏移（E），未验证。

### #21 Engram 前向：查表 → 一次融合投影 → 门控写回 4 路主干
- 证据：`engram.py:139-161`
  > `kv = self.fused_kv_projection(embed_hook()).unflatten(-1, (hidden_states.size(-2) + 1, hidden_states.size(-1)))`
  > `gated, *_ = tile_kernels.engram.engram_gate_fwd(hidden_states, kv, self.weight_fused, self.eps, self.clamp_value, ..., image_token_mask=images_seq_mask, inplace=True)`
- 证据：`engram_parameters.py:58-64`（`fused_kv_projection` 输出 `n_embed * (hc_mult + 1)`；`weight_hidden` / `weight_embed` 各 `[hc_mult, n_embed]`）
- 等级：A
- 意味着什么：投影输出被拆成 `hc_mult+1` 组：前 4 组对应 4 条超连接路，第 5 组由 gate kernel 消费（推断为门控 logits，E 级）。`inplace=True` 说明它是**原地改主干**，不新增输出张量——这对 decode 的显存/带宽很关键。
- 疑点：`engram_gate_fwd` 的数学（sigmoid? element-wise? 逐路 gate?）在 `tile_kernels` 里，仓库内看不到，**未能确认**。

### #22 `engram_hash` 在模型 forward 里只算一次，跨层复用
- 证据：`deepseek.py:177-188`
  > `hashes = tile_kernels.engram.engram_hash(input_metadata.ngram_token_ids, self.ngram_multipliers, self.ngram_vocab_sizes, self.ngram_offsets, getattr(input_metadata, "images_seq_mask", None))`
  > `ngram_embed_hooks = self.engram_runtime.fetch(hashes)`
- 证据：`deepseek.py:214-215`（逐层传 `embed_hook=ngram_embed_hooks.get(i)`）
- 等级：A
- 意味着什么：hash 与跨 rank 的 embedding 拉取都在**第一层之前**发起（`fetch` 返回 hook，真正取数延迟到层内），这样 DeepEP 的通信可以和前面的层计算重叠：Engram 层是 1 和 14，layer 14 能压住 13 层的时间，layer 1 只有 1 层可重叠。是典型的"预取 + 延迟消费"流水。
- 疑点：`fetch` 的具体通信/去重策略在 `deep_ep.EngramBuffer`（vendor 内未见实现文件），**未能确认**。

### #23 `gather_engram_tokens_zos` 要解决的是"decode 时 n-gram 窗口跨了三种存储"
- 证据：`deepinfer/model_executor/layers/tilelang_ops/gather_engram_tokens_zos.py:27-42`（docstring 的三分支）
  > `j >= r and pidx < 0: out[q_base+j, r] = curr_fresh[i, c]`（fresh 序列：CPU 已压缩）
  > `j >= r and pidx >= 0: out[...] = lookup_table[prev token of seq pidx]`（上一步序列：GPU 上用查找表压缩）
  > `j < r: out[...] = win_compressed[i, window_size - num_reject + j - r]`（更早的回看：压缩历史环形窗）
- 证据：`cuda/gather_engram_tokens_zos.py:41-74`（`with T.Kernel(n_seq, threads=32)`，每个 seq 一个 block、32 线程；`T.pdl_sync()` 做前驱 kernel 的依赖同步；`hit_image` 一旦遇到图像哨兵就把该阶及更高阶填 pad）
- 等级：A
- 意味着什么：这三段数据分别在 CPU 内存、上一步的 GPU 输出、GPU 环形窗里；投机解码被拒还会让窗口整体偏移 `num_reject`。放进一个 kernel 才能既不做 D2H 同步、又不退化成 Python 逐序列循环，并且能进 CUDA graph。
- 疑点：`getattr(input_metadata, "images_seq_mask")` 在多模态下才有值，纯文本路径该参数为 None，kernel 里 `image_sentinel` 分支退化。

### #24 ZOS 是 `gather_engram_tokens_zos` 存在的前提
- 证据：`deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py:225-274`（`metadata_zos.remap is not None` 分支里才调 `gather_engram_tokens_zos`；`remap is None` 时直接 `copy_` CPU 预计算好的 `ngram_token_ids`）
- 证据：`deepinfer/model_executor/model_metadata_utils.py:143`（`prepare_decode = prepare_decode_v4_zos if envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING else prepare_decode_v4`）
- 等级：A
- 意味着什么：非 ZOS 路径下 CPU 知道本步的 token，可以在 CPU 上把 n-gram 拼好再 H2D；ZOS 下 CPU 只知道"上一步"，本步 token 在 GPU 上，所以必须用算子。**这个算子是异步调度的直接产物**。
- 疑点：`ZOS` 全称仓库里没有展开；从 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 推断为 "Zero-Overhead Scheduling"（E，几乎确定）。

### #25 Engram 的前置条件是 tokenizer 压缩
- 证据：`config.py:209`（`enable_tokenizer_compression=bool(text.get("engram_compressed_vocab_size"))`）
- 证据：`model_metadata_utils.py:109-110`（`assert not self.engram_enabled or self.engram_cfg.enable_tokenizer_compression`）
- 证据：`engram_metadata.py:16-55`（`CompressedVocabulary`：逐 token decode + NFKC/NFD/去重音/小写/空白归一，得到 99,092 个压缩词表）
- 等级：A
- 意味着什么：n-gram 是在**归一化后的压缩词表**上取的，不是原始 129,280 词表。这直接决定了 hash 表的命中率与规模（16M/head 的桶是基于 99K 词表设计的）。
- 疑点：压缩词表是"去空白/大小写/重音"后的等价类，`"\ufffd"`（非法 UTF-8 字节片段）走原始拼写保留分支（`:45-47`），细节见代码。

## 3. 必答问题逐条回答

### Q1 层结构：encoder 层、decoder 层各代表什么？总层数？前多少层是 encoder？YOCO 如何让 decoder 的块状 KV 由最后一个 encoder 层的 hidden 投影得到？
- **encoder 层**：`n_prefill_layer` 之前的层（V4.1：layer 0–19）。它们对**全序列**做 attention，产出块状 KV；其中 2 层纯 SWA、18 层 C2 压缩（#1/#3/#6）。
- **decoder 层**（代码叫 consumer 层）：`n_prefill_layer` 及之后的层（V4.1：layer 20–39）。它们不自己产块状 KV，只对"保留的 140 个 token"算 query 和 SWA（#9/#11）。
- **总层数**：40 层主干 + 3 层 DFlash（MTP，target 37/38/39）。前 **20** 层是 encoder（#3）。
- **投影链条**（四步，全部 A 级）：
  1. 第 20 层的 `attn_norm_forward` 输出 bf16 hidden（= encoder 栈输出经 HC pre-mix + RMSNorm），在 `decoder_layer.py:152` 写入 `cross_layer_runtime.shared_hidden`；同时 `decoder_layer.py:140` 用**全量 metadata** 再算一次，得到 `consumer_full_hidden`。
  2. 第 20 层是 KV 源，构造时把 `PackedWkv` 的 `A.weight`（C1 下不含 Gate）单独封成 `shared_hidden_kv_proj`（`attn.py:345-353, 372-373`）。
  3. decode / 非 compact prefill：`attn.py:775-780` 用 `shared_hidden_kv_proj(shared_hidden_bf16)` 得到 `projected_kv`，再走 `Compressor.forward_store_kv_cache` 写 `ca_cache`。
  4. compact prefill：`attn.py:692-708` 用 `consumer_full_hidden` 一次算出**全长度** KV；本层 query 侧的 store 被 `attn.py:537-540` 跳过。
  第 21–39 层不持有 KV 参数，靠 `layer_kv_map` 把 `ca_cache` 别名到第 20 层（#8）。
- 引用：#3 #4 #5 #7 #8 #10 #14。

### Q2 `cross_layer.py` 的跨层 KV 共享机制是什么？
- **静态拓扑** `SharedLayerTopology`：模型构造时建一次，把 config 里稀疏的 `layer_kv_map` / `layer_index_module_map` / `layer_index_k_map` 归一化成覆盖所有层的 list（缺省自指），并做四类 fail-fast 校验：副本一致、消费者必须携带配置、分段单调、`index-topk ⊆ index-K ⊆ KV`（#13）。
- **动态载体** `CrossLayerRuntime`：每次 forward 新建一个，按引用逐层传；6 类值单槽位覆盖，靠"层号递增 + 消费者只读最近源"保证正确（#12）。
- **KV 本体不在 runtime 里**：`ca_cache` 由 cache engine 按 `layer_kv_map` 直接别名，零拷贝（#8）；runtime 里只放 `loaded_prefill_kv`（prefill 时从 `ca_cache` 解压出来的临时张量，share-kv 消费者原地覆盖 SWA 区）。
- **硬约束**：share-kv 不得跨 YOCO 边界，第 20 层必须自指（#14）。
- 设计效果：40 层里只有 4 个 KV 源段（2/8/14/20），decoder 半区 20 层压成 1 份 KV（#8 + §5 的字节估算）。
- 引用：#8 #12 #13 #14。

### Q3 `hc.py` 与 `mhc_ops.py`：mHC 是什么？对推理计算图和残差连接的具体影响？
- **是什么**：把单路残差主干扩成 `hc_mult`=4 路（`[n,4,H]`），每个子层前用 4 路加权混合出该子层输入（pre），子层输出再按 4×4 组合矩阵写回 4 路（post）；混合系数由 `fn`（24×20480 fp32）现算，并用 sinkhorn 迭代 20 次把组合矩阵归一化成双随机矩阵（#15 #16）。
- **对残差连接**：残差不再是 `x + f(x)`，而是 `residual_new = comb_res_mix @ residual + post_layer_mix ⊗ f(x)`（`mhc_post`，`mhc_ops.py:31-37`）。组合矩阵让 4 条路之间可以互相"搬运"信息，表达力换显存。
- **对计算图**：
  1. 层间节点变形状：`[n,5120]` → `[n,4,5120]`；`HCLazyPostData` 让残差加法**延迟**到下一个子层，`mega_mhc` 一个 kernel 做 post+pre+RMSNorm+FP8 cast（#18）。
  2. `hc_shifted=True`（V4.1 默认）把 pre-mix 偏移半层，末尾用 `mhc_pre_apply_mix` 折回单路，**没有 HCHead 参数**（#17）。
  3. 一旦某层有 Engram 干预，`mega_mhc` 被禁用，退化为 post / pre 两次调用（#19）。
  4. RMSNorm 作用在**拼接后的 4×5120** 上（`hc_norm_group_size=20480`，`config.py:296`），不是每路各归一化。
- 引用：#15 #16 #17 #18 #19。

### Q4 `engram.py`：Engram 是什么？N-gram 嵌入表多大？为什么需要 `gather_engram_tokens_zos`？
- **是什么**：稀疏 n-gram 记忆。用最近 1–3 个 token 组成 2/3/4-gram，按"每层 × 每 ngram × 每 head 一个互异素数"做 hash，去一张巨大的分片表里取 2048 维向量，经一次融合投影变成 `hc_mult+1` 组 KV 样张量，再用门控**原地**写回 4 路主干（#20 #21）。
- **表规模**：3 张表（2/3/4-gram）× 8 head × 每 head 约 1600 万桶 ≈ 3.84 亿行/层，行宽 2048；fp8 下 **786 GB/层**，两个 Engram 层共 **约 1.57 TB**（#20，E 级换算）。表按 `engram_parallel_group` 等分，`num_per_part = padded_rows / group_size`（`engram_parameters.py:19-32`）。
- **为什么需要专用 gather 算子**：ZOS decode 下 CPU 落后一步，本步 query token 只在 GPU；构建每个 query 的 4 列 n-gram 需要同时读三类来源（fresh 的 CPU 压缩结果、上一 step 的 GPU 采样/草稿 token 经 `lookup_table` 现场压缩、以及压缩历史环形窗），还要处理投机解码被拒导致的窗口偏移和图像哨兵截断。放 CPU 就要 D2H 同步 + 逐序列 Python 循环；放多个小 kernel 会打断 CUDA graph。所以做成一个 TileLang kernel，每序列一个 block、32 线程，`T.pdl_sync()` 与前驱 kernel 流水（#23 #24）。
- 引用：#20 #21 #22 #23 #24 #25。

## 4. 新发现的问题
1. **"CED" 的展开未确认**。全仓库 `grep -i "ced"` 无架构用法；只有 `doc/inference_deployment_qa.md:187` 出现 "In the CED architecture"。代码里的对应词是 prefill layer / YOCO consumer layer。推测 C = Compressed、E = Encoder、D = Decoder（E 级，无证据）。**建议书里只用代码能证的表述"encoder/consumer 两段"，CED 缩写标注为官方文档用词。**
2. **文档的 "2,560 tokens" 对不上**。`doc/inference_deployment_qa.md:187` 说精确重建需要"running the last 2,560 tokens through 20 decoder layers"，但窗口是 128、保留是 140。2,560 = 128×20 或 140×18 都凑得上，含义未确认，**不建议在书里引用这个数字**。
3. **为什么 decoder 用不压缩的 C1、encoder 用 C2？** 直觉上 decoder 更该省。推测：decoder 只有 1 份 KV，压缩收益小；而 C1 免掉 gate/state 累加，decode 步延迟更低（E）。需查技术报告确认。
4. **1.57 TB 的表怎么放**。`engram_parallel_size` 由 `PG.engram_parallel_group()` 决定，代码里看不到它是否等于全 world size；若按 8 卡分片仍是 196 GB/卡。需要确认部署拓扑（可能与 EP 组复用）。
5. **`yoco_swa_share_hidden` 两处默认值不一致**（`config.py:47` False vs `yoco.py:39` True）。目前 V4.1 走 False；若有人手工构造 cfg 会踩到 True 分支，行为差异是"consumer SWA 也来自 shared hidden"。
6. **`MoeDecoderBlockHC` 是唯一被支持的 block 类型**（`decoder_layer.py:57-58` 断言），`DFlashDecoderLayer` 复用同一 mixin；书里若讲"encoder/decoder 是两种 block"会与代码冲突。
7. **Engram 与 YOCO 的互斥**：`deepseek.py:136-138` 只断言 compact 模式不支持 consumer 层 Engram，`decoder_layer.py:136` 又断言边界层不能是 Engram 层；两条断言合起来意味着"Engram 必须在 encoder 半区"，V4.1 的 [1,14] 满足。
8. **`fetch` 的去重/通信策略不可见**（`deep_ep.EngramBuffer` 无源码），若要写"跨卡按 hash 拉取"的细节需要另找 DeepEP 文档。

## 5. 可用于图表的材料

### 5.1 层结构表（V4.1-Flash）
| 层号 | 数量 | `compress_ratios` | 类型 | 块状 KV | KV 源 | index 源 |
|---|---|---|---|---|---|---|
| 0–1 | 2 | 0 | 纯 SWA（`sparse_stage=0`） | 无 | 自身（无 KV） | — |
| 2–19 | 18 | 2 | C2 压缩 + 稀疏 index | 有（2:1） | 2 / 8 / 14 | 2 / 8 / 14 |
| 20–39 | 20 | 1 | C1 不压缩 + 稀疏 index（YOCO consumer） | 有（1:1，仅 layer 20 持有） | 20 | 20 / 24 / 28 / 32 / 36 |
| 40–42 | 3 | 0 | DFlash 草稿层（target 37/38/39） | — | — | — |

### 5.2 YOCO 数据流（prefill compact 模式）
1. layer 0–19：全序列 attention，写各自段 KV（`ca_cache`）。
2. layer 20 入口：`input_metadata` 切 compact 视图（`deepseek.py:196-198`），同时用 full 视图算 `consumer_full_hidden`（`decoder_layer.py:140`）。
3. layer 20：`shared_hidden_kv_proj(full_hidden)` → 全长度块状 KV 落盘（`attn.py:704-708`）；query 侧只跑 140 token。
4. layer 21–39：`ca_cache` 别名到 20 层，只算 140 token 的 SWA 与 topk。
5. decode：每步由 layer 20 用本步的 shared hidden 投影出新 token 的 KV（`attn.py:775-780`，只有 KV 源层会走），layer 21–39 仍读同一份 `ca_cache`，不产生新的 KV 写入。

### 5.3 KV 字节预算（E 级估算，可做对比柱状图）
- 单 token 单层块状 KV：528 B（MXFP8，`cache_engine.py:212-213`）；C2 层因 2:1 压缩摊到 264 B/token。
- **不共享**：18 层 C2 → 9×528 B，20 层 C1 → 20×528 B，合计 **15,312 B/token**。
- **本方案**：3 个 C2 段 → 3×264 B，1 个 C1 段 → 528 B，合计 **1,320 B/token**。
- 压缩比 **≈11.6×**；1M 上下文对应 1.32 GB/seq（仅块状 KV）。
- SWA KV 另计：20 层（stored region）× 134 token × 528 B = 1,415,040 B/seq（与 `doc/inference_deployment_qa.md:169` 一致），consumer 的 20 层 SWA 在 dropped region，不落盘。

### 5.4 mHC 单个子层的数据流（可直接画框图）
`residual[n,4,H]` + `x[n,H]`（上一子层输出）
→ `mega_mhc`：post(4×4 comb) → pre(4 路加权) → RMSNorm(20480) → FP8 cast
→ 输出 `new_residual[n,4,H]`、`post_mix[n,4,1]`、`comb_res_mix[n,4,4]`、`pre_mix[n,4,1]`、`y_fp8`
→ 子层（attn 或 MoE）消费 `y_fp8`，输出 `[n,H]`
→ 交给下一个子层（不清算，`HCLazyPostData` 延迟）
末尾：`hc_shifted` → `mhc_pre_apply_mix` 折成 `[n,H]`（无 HCHead）。

### 5.5 Engram gather 的取数矩阵（r = 回看阶）
| 条件 | 数据来源 | 位置 |
|---|---|---|
| `j >= r` 且 fresh 序列 | `curr_fresh[i, j-r]` | CPU 预压缩 → H2D |
| `j >= r` 且复用上一步序列 | `lookup_table[prev_sampled / prev_predict]` | GPU 现场压缩 |
| `j < r` | `win_compressed[i, window_size - num_reject + j - r]` | GPU 环形窗 |
| 任一阶命中 `image_sentinel(-1)` | 该阶及更高阶填 `engram_compressed_pad_token` | kernel 内 `hit_image` 标志 |
