# R02a 注意力：MLA、indexer 稀疏选择、C1/C2 压缩、SWA、多级 KV 体积

## 0. 元信息

- 代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21）。本文所有 `路径:行号` 均指该 commit。
- 模型配置：HF `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json`（2026-10-23 读取）。本文引用的值：`num_hidden_layers=40`、`num_nextn_predict_layers=3`、`num_attention_heads=64`、`num_key_value_heads=1`、`head_dim=512`、`qk_rope_head_dim=64`、`q_lora_rank=1280`、`o_lora_rank=1024`、`o_groups=8`、`sliding_window=128`、`compress_ratios=[0,0] + [2]*18 + [1]*20 + [0,0,0]`（43 项）、`index_topk=512`、`index_n_heads=32`、`index_head_dim=128`、`candidate_topk_blocks=2048`、`candidate_block_size=8`、`candidate_source_layer_id=20`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`kv_source_layer_ids=[2,8,14,20]`、`compress_rope_theta=160000`、`max_position_embeddings=1048576`。
  配置文件不在仓库内，故凡只来自 `config.json` 的数值记为 B 级；"代码如何使用这些值"记为 A 级。
- 查阅的主要来源：
  - `deepinfer/hugging_face/config.py`
  - `deepinfer/model_executor/models/v4_modules/attn.py`、`mla_attn.py`、`index_attn.py`、`compress.py`、`cross_layer.py`、`yoco.py`
  - `deepinfer/worker/cache_engine.py`
  - `deepinfer/model_executor/model_metadata_utils.py`
  - `deepinfer/model_executor/layers/attention_backend.py`
  - `deepinfer/model_executor/layers/tilelang_ops/{cache_common.py,store_kv_cache.py,load_paged_kv_cache.py,state_cache.py,compress.py,kv_index_cat.py,candidate_blocks.py}`
  - `deepinfer/environ.py`
  - `vendor/FlashMLA/flash_mla/{flash_mla_interface.py,fused_norm_rope_attn_rope_cast.py}`
  - `doc/inference_deployment_qa.md`

## 1. 三个最重要的发现

1. **V4.1 的"MLA"是 64 头 query 对一条 512 维共享 latent 的 MQA**：`kv_lora_rank=448` + `qk_rope_head_dim=64` = 512，RoPE 打在 latent 末尾 64 维上（不是额外拼接），全模型只有 1 个 KV 头；Q 走 `1280` 维低秩再升到 `64×512`。
2. **块状 KV 只由 4 个层产生**（layer 2/8/14 为 C2 2:1 压缩，layer 20 为 C1 1:1），indexer K 也只有 4 份；按每输入 token 摊算是 **720 B（main KV）+ 170 B（indexer K）= 890 B**，与官方摘要的 890 B/token 精确吻合。
3. **40 个主干层全部有 128-token SWA，没有任何"全量注意力"层**：layer 0–1 纯 SWA，2–19 是 SWA + C2 压缩 KV，20–39 是 SWA + C1 共享 KV（YOCO）；SWA KV 是 528 B/token/层，按序列固定开销（≈2.8 MB/seq），不随上下文长度增长。

## 2. 事实条目

### #1 每层注意力类型只由 `compress_ratios` 的一个整数决定
- 证据：`deepinfer/hugging_face/config.py:146-155`
  > `specs: list[Any] = [_win_spec(text) if ratio == 0 else _compress_spec(text, ratio, shared) for ratio in ratios[:n_layer]]`
  > `specs.append(None)  # lm head takes one entry in attn_module_list but no attn is built`
  > `specs.extend(_dflash_spec(text) for _ in range(nextn))`
- 证据：`config.py:99-108`（ratio==0：`sink_mode="swa"`、`sparse_stage=0`，不设 `n_packed_heads`）；`config.py:111-133`（ratio!=0：`n_packed_heads=num_attention_heads`、`kv_compress_ratio=ratio`、`index_topk`、`attn_index`）
- 等级：A
- 意味着什么：读 V4.1 结构只需看 `compress_ratios`。ratio 0 = 只有滑窗；ratio 1/2 = 滑窗 + 块状 KV（C1 不压缩 / C2 2:1 压缩）。
- 疑点：无

### #2 40 层主干 + 3 层 DFlash 草稿，前 20 层是 encoder
- 证据：HF `config.json`：`num_hidden_layers=40`、`num_nextn_predict_layers=3`、`compress_ratios` 共 43 项 = `[0,0] + 2×18 + 1×20 + [0,0,0]`
- 证据：`config.py:158-161`
  > `"""The first layer with ratio==1: it and everything before it form the prefill side, the layers after it are yoco consumer layers."""`
  > `return next(i for i, ratio in enumerate(ratios) if ratio == 1)`
- 等级：B（数值）+ A（推导）
- 意味着什么：`n_prefill_layer=20`；layer 0–19 为 encoder 侧，20–39 为 YOCO consumer（decoder）侧。与 `doc/inference_deployment_qa.md:163`（"the stored layers are the first 20"）、`:187`（"20 decoder layers"）一致。
- 疑点：无

### #3 MLA 形状：512 维 latent 内含 64 维 RoPE，KV 头数为 1
- 证据：`config.py:287-292`
  > `head_dim=text["head_dim"],`
  > `# HF's shared KV vector includes RoPE; the internal rank excludes it.`
  > `kv_lora_rank=text["head_dim"] - text["qk_rope_head_dim"],`
- 证据：`deepinfer/model_executor/models/v4_modules/mla_attn.py:232-246`
  > `self.Wkv = ColumnParallelLora(cfg.hidden_size, 0, self.kv_lora_rank, num_lora_rope_head=cfg.num_key_value_heads, rope_head_dim=cfg.qk_rope_head_dim, ...)`
- 证据：`mla_attn.py:175-176`（`self.num_lora_rope_head == 1` 断言）
- 等级：A
- 意味着什么：HF 的 "shared KV vector" 宽 512（448 nope + 64 rope），推理内部把 rope 单独计为 64，因此 `kv_lora_rank=448`。这解释了 KV cache 每 token 恰好 512 个量化元素（#16）。
- 疑点：与 DeepSeek-V3 的 576（512 nope + 64 rope）不同，V4.1 把 RoPE 塞进 512 维内，省 64 维/token。

### #4 Q 路径是低秩 1280，且 Wq.A 与 Wkv.A 融合成一个 GEMM
- 证据：`mla_attn.py:198-211`（`Wq = ColumnParallelLora(hidden_size, q_head_dim*n_head, q_lora_rank)` → A: 5120→1280，norm，B: 1280→64×512）
- 证据：`attn.py:278-290`
  > `self.fuse_kv_swa = self.use_swa and not self.yoco_swa_share_hidden`
  > `self.q_a_kv_swa_proj = ReplicatedLinear(cfg.hidden_size, self.q_lora_rank + (self.qk_head_dim if self.fuse_kv_swa else 0), weight=merge_linear_weights(q_a_kv_swa_weights))`
- 等级：A
- 意味着什么：V4.1 默认 `yoco_swa_share_hidden=False`（`config.py:47`），所以每层都是 `5120 → 1792` 的单次融合投影（1280 Q + 512 KV），减少一次 GEMM 与小 kernel 启动。
- 疑点：无

### #5 注意力本身是"共享 512 维 latent 的 MQA"，不是 per-head K/V
- 证据：`attn.py:70-78`
  > `self.qk_head_dim = cfg.head_dim` / `self.num_heads = cfg.num_attention_heads` / `self.softmax_scale = self.qk_head_dim**-0.5`
- 证据：`attn.py:660-669`（decode 调 `decode_mla_mqa_sparse(q_b, head_dim_v=self.qk_head_dim, ...)`）
- 证据：`vendor/FlashMLA/flash_mla/flash_mla_interface.py:80-84`
  > `q: (batch_size, seq_len_q, num_heads_q, head_dim). bfloat16. head_dim must be 512 and num_heads_q must be 64 or 128.`
  > `k_cache: ... num_heads_k must be 1 (only MQA is supported).`
- 等级：A
- 意味着什么：64 个 query head 共用同一条 512 维 KV，attention 直接在 latent 空间算；这正是 MLA 的"吸收"形态，因此 KV 只有一份。
- 疑点：`attention_backend.py:134` 硬编码 `is_fp8_kvcache=True`（注释"V4.1 的 KV cache 为 paged FP8"）。主 cache 是 SWA（确实 FP8），压缩 KV 走 `extra_k_cache`（FP4）；只有当某层 `use_swa=False` 时主 cache 才会是 FP4，届时该 flag 是否仍正确，未验证。

### #6 输出侧：8 组、每组 8 个 head 的 4096 维 context → 1024，再 out_proj 到 5120
- 证据：`mla_attn.py:221-229`
  > `self.n_wv_group = cfg.o_groups`
  > `wv_in_features = (self.kv_lora_rank + self.rope_head_dim) * (self.n_head // self.n_wv_group)`
  > `self.Wv = ReplicatedLinear(wv_in_features, mid_dim, bias=False, use_fp8_gemm=True)`
- 证据：`mla_attn.py:282`（`self.out_proj = ReplicatedLinear(mid_dim, cfg.hidden_size, ...)`）
- 证据：`attn.py:122-156`（`Wv` 权重视图 `[8, 1024, 4096]`，FP8 + per-32 scale，`deep_gemm.transform_sf_into_required_layout`）
- 等级：A
- 意味着什么：attention 输出的 64×512 不是直接 concat 成 32768，而是按 8 组（每组 8 头）先降维到 1024，再 8192→5120。这是 MLA 的 V 吸收在 V4.1 的实现形式，也决定了 decode 融合 kernel 的输出布局 `[s_q, n_wv_group, wv_group_size*d_v]`（`fused_norm_rope_attn_rope_cast.py:151`）。
- 疑点：为什么必须是 8 组而不是 1 组，代码只给了 `o_groups` 配置，未见注释解释；推断是 FP8 GEMM 的 N 维形状/权重布局限制（E）。

### #7 Prefill 把 SWA 与压缩 KV 拼成一条 index，单 query 最多看 640 个 KV
- 证据：`attn.py:812-839`
  > `# Assemble the input kv for prefill attention: [swa cache | current swa kv | compressed cache]`
  > `total_kv = total_k_swa_cached + total_k_swa_cur + total_k_compressed`
- 证据：`attn.py:603-610`（`kv_index = kv_index_cat(kv_index, kv_ca_index, total_k_swa_cached, total_k_swa_cur)`）
- 证据：`deepinfer/model_executor/layers/tilelang_ops/kv_index_cat.py:16-24`
  > `kv_index: [total_q, window_size]` / `kv_ca_index: [total_q, topk_tokens]`
  > `out: [total_q, window_size + topk_tokens]`
- 等级：A
- 意味着什么：prefill 只有一次稀疏 varlen attention，key 空间 = 128（滑窗）+ 512（topk 压缩 token）= 640，不是两趟 attention。这是理解 V4.1 prefill 性能的关键。
- 疑点：无

### #8 Decode 用两个 paged cache：主 cache = FP8 SWA，extra = FP4 压缩 KV
- 证据：`attn.py:640-653`
  > `attn_kv_args = dict(key_cache=kv_cache.swa_cache, indices_in_kvcache=kv_swa_index, key_cache_extra=kv_cache.ca_cache, indices_in_kvcache_extra=kv_ca_index, ...)`
- 证据：`deepinfer/model_executor/layers/attention_backend.py:125-142`（`flash_mla_with_kvcache(..., is_fp8_kvcache=True, causal=False, ...)`）
- 证据：`vendor/FlashMLA/flash_mla/fused_norm_rope_attn_rope_cast.py:138-147`
  > `The format is detected from bytes_per_token: 528 (V4.1 fp8) or 288 (V4.1 with fp4 e2m1 + per-16 e4m3 scales; only valid for extra_k_cache).`
- 等级：A
- 意味着什么：decode 阶段没有任何 BF16 KV 物化；每条 KV 记录 = 数据 + scale 紧挨着按 token 交错存放（`token_interleaved=True`）。
- 疑点：`causal=False`（`:133`）说明稀疏 attention 不做因果掩码，正确性完全依赖 topk 索引只包含 ≤ 当前位的 token（由 `ks/ke` 决定）。

### #9 indexer 是 32 头 × 128 维的轻量 MQA，Q/K 都用 FP4
- 证据：`deepinfer/model_executor/models/v4_modules/index_attn.py:56-57`
  > `"""The indexer of sparse attention: computes the relu(q@k)·weights logits and selects the topk compressed kv for each query"""`
- 证据：`index_attn.py:76-86`
  > `self.n_head = index_attn_module.n_head` / `self.head_dim = index_attn_module.head_dim`
  > `# Indexer Q/K always use FP4 (E2M1 + UE8M0 scale); weights and psum use BF16.`
  > `self.cache_format = FP4_INDEX_KV_FORMAT`
- 证据：`mla_attn.py:125-147`（`q_B: q_lora_rank → 32×128`；`weights_proj: hidden → 32`；`k_proj: 512 → 128` + RMSNorm）
- 等级：A
- 意味着什么：indexer 不做 softmax attention，只算每个 query 对每个压缩 token 的"相关性分数"；Q/K 量化到 FP4，权重与 p_sum 保持 BF16。
- 疑点：`use_q_norm=False` 被硬断言（`mla_attn.py:120`），indexer Q 没有 norm；为什么不需要，无注释。

### #10 选择规则：`p_sum = Σ_h w_h · relu(q_h·k)`，每 query 取 top-512
- 证据：`index_attn.py:284-325`
  > `return deep_gemm.fp8_fp4_mqa_logits(q_chunk, kv, weights[chunk], ks, ke, max_seqlen_k=self.max_kv_len)`
  > `return deep_gemm.fp8_fp4_sparse_mqa_logits(q_chunk, kv, weights[chunk], schedule_metadata, ...)`
- 证据：`index_attn.py:361-372`（`topk(p_sum, self.topk_tokens, ...)`，`topk_index` 形状 `[total_q, topk_tokens]`）
- 等级：A
- 意味着什么：sparse 选择是"逐头 relu 后按头加权求和"，再在 query 维度独立 top-k；与 DeepSeek 的 lightning indexer 同构，但 K 来自压缩 KV 而非原始 token。
- 疑点：`deep_gemm.fp8_fp4_mqa_logits` 的 kernel 实现不在本仓库（vendor/DeepGEMM），无法确认数值细节（缩放、累加精度）。

### #11 两级候选：layer 20 先选 2048 个 8-token block，21–39 只在候选内 top-512
- 证据：`index_attn.py:97-105`
  > `candidate_source = shared_topology.layer_candidate_source_map.get(self.layer_idx) ...`
  > `self.is_candidate_topk_source = ... is_candidate_source_layer(self.layer_idx)`
  > `assert 0 < candidate_cfg.candidate_topk_blocks <= 4096 and candidate_cfg.candidate_topk_blocks % 4 == 0`
- 证据：`index_attn.py:204-226`
  > `block_scores, valid_blocks = candidate_block_max(p_sum, k_len, block_size)`
  > `topk(block_scores, topk_blocks, ..., output_idx=output, idx_oob_fill_value=-1, return_value=False)`
- 证据：HF `config.json`：`candidate_source_layer_id=20`、`candidate_topk_blocks=2048`、`candidate_block_size=8`
- 等级：B（数值）+ A（代码）
- 意味着什么：2048×8 = 16,384 个候选压缩 token（layer 20 在整段上下文上算），layer 21–39 的 top-512 只在这 16,384 个里选；即"粗选 block → 精选 token"，后 19 层省掉全量 logits。
- 疑点：候选块评分用 block 内 max（`candidate_blocks.py:19-21`），对"关键 token 稀疏分布"的召回率影响未评估。

### #12 候选 block 的"压缩前"大小 = 8 × ratio
- 证据：`index_attn.py:560-569`
  > `cross_layer_runtime.candidate_topk_block = candidate_width`
  > `cross_layer_runtime.candidate_block_size_before_compress = self.candidate_topk_cfg.block_size * self.kv_compress_ratio`
- 证据：`index_attn.py:239-240`（消费侧再除回 ratio：`candidate_block_size = ... // self.kv_compress_ratio`）
- 等级：A
- 意味着什么：candidate_block_size=8 是"压缩后 token"的单位，乘以 ratio 换成输入 token 坐标。V4.1 的候选只服务 layer 20–39（ratio 1），所以 8 就是 8 个原始 token；`*ratio` 的写法是为多 ratio 混合的模型预留。跨层共享候选时必须用压缩前坐标，否则 ratio 不同的层无法对齐。
- 疑点：无

### #13 indexer K 的 cache 格式与体积：FP4 E2M1 + UE8M0 per-32 = 68 B/token
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cache_common.py:129-137`
  > `FP4_INDEX_KV_FORMAT = KVCacheFormat(quantized_dtype="e2m1", quantized_elements=128, elements_per_sf=32, ..., sf_format="ue8m0", note="V4/V41, index kv.")`
- 证据：`deepinfer/worker/cache_engine.py:73-80`
  > `# FP4: two values per byte, one UE8M0 scale per 32 channels.`
  > `return ic.index_head_dim // 2 + ic.index_head_dim // 32`
- 等级：A
- 意味着什么：128/2 + 128/32 = 64 + 4 = **68 B**。这是"MXFP4"（E2M1 数据 + UE8M0 per-32 scale），不是 per-16。
- 疑点：无

### #14 `compress.py` 压缩的是时间轴上的 512 维 KV latent：C1 恒等，C2 用逐通道 softmax 门控做 2:1
- 证据：`deepinfer/model_executor/models/v4_modules/compress.py:56-60`
  > `"""Pure KV compressor: compression + norm; returns the pre-rope compressed kv and writes the post-rope value into the main KV cache.`
  > `MQA uses KProject on this compressor's pre_rope_kv to produce index-K; ..."""`
- 证据：`compress.py:89-96`
  > `if self.kv_compress_ratio == 1: assert projected_kv.dtype == torch.bfloat16 and state_cache is None; ...; return projected_kv`
  > `assert projected_kv.dtype == torch.float32 and state_cache is not None`
  > `score, latent = projected_kv.chunk(2, dim=-1)`
- 证据：`deepinfer/model_executor/layers/tilelang_ops/compress.py:199`
  > `result = (latent_slice * score_slice.softmax(dim=0)).sum(dim=0).to(torch.bfloat16)`
- 等级：A
- 意味着什么：C2 = 相邻 2 个 token 的 latent 按"逐通道 softmax 权重"加权平均成 1 条；C1 = 不压缩。因此 C2 层每条 KV 覆盖 2 个输入 token，这就是"压缩 KV"的全部含义（不压缩 hidden、不压缩权重）。
- 疑点：`score` 的 softmax 沿 token 维（dim=0）做，每个 token 的 Gate 输出是 512 维 logits（不是 1 个标量门），因此每个通道有各自的 2-token 混合比例；这比标量门表达力强，但不能解释成"注意力权重"。

### #15 C2 状态环：7 个 token 的 `[score|latent]`，决定 `ca_state` 大小
- 证据：`cache_engine.py:130-133`
  > `"""Compression ring plus rollback history; environment changes require clearing the cache."""`
  > `return compress_ratio + max(0, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS - 1)`
- 证据：`deepinfer/environ.py:129-132`
  > `DEEPINFER_USE_NEXTN_DECODE = EnvVar.int(default=5)`
  > `DEEPINFER_MAX_BACKTRACKING_TOKENS = EnvVar(lambda v: max(_int(v), envs.DEEPINFER_USE_NEXTN_DECODE + 1))`
- 等级：A
- 意味着什么：默认 2 + (6−1) = 7 个 token 的环形缓冲；ring 里存的是每个位置的原始 score/latent，使跨 step 的压缩窗口（以及投机回退后的重算）仍能拼出完整窗口。
- 疑点：若用户把 `USE_NEXTN_DECODE` 调到 0，`MAX_BACKTRACKING` 仍 ≥1，ring = ratio；这条路径未测试。

### #16 主 KV 格式：FP4 E2M1 + E4M3 per-16 = 288 B/压缩 token
- 证据：`cache_common.py:119-128`
  > `V41_FP4_FORMAT = KVCacheFormat(quantized_dtype="e2m1", quantized_elements=512, elements_per_sf=16, ..., sf_format="e4m3", note="V41, FP4 main kv.", token_interleaved=True)`
- 证据：`cache_engine.py:46-56`
  > `layout = V41_FP4_FORMAT if use_fp4 else V41_FORMAT`
  > `assert kv_lora_rank + rope_head_dim == layout.num_elements`
  > `return layout.bytes_per_token`
- 证据：`model_metadata_utils.py:37-39`
  > `"""FP4 主压缩 KV 每 16 个值共用 scale；MXFP8 主 KV 和 SWA 每 32 个值共用 scale。"""`
  > `return 16 if not is_swa and cfg.use_fp4_sparse_kv else 32`
- 等级：A
- 意味着什么：512×0.5 B 数据 + 512/16 = 32 B 的 E4M3 scale = **288 B**。官方文档称之为"per-16, e4m3 — like NVFP4 but without the global scale"（`doc/inference_deployment_qa.md:217`）——注意 per-16 指 scale 粒度，数据本身是 E2M1。
- 疑点：`use_fp4_sparse_kv` 在 HF config 中不存在，代码默认 True（`config.py:303`）；若某 checkpoint 显式设 False，主 KV 会退回 528 B 的 MXFP8。
- 疑点：NVFP4 规范是 per-16 + E4M3 scale + 一个 per-tensor FP32 全局 scale；这里没有全局 scale，所以数值范围依赖 per-16 scale 自身。

### #17 SWA KV 格式：FP8 E4M3 + UE8M0 per-32 = 528 B/token/层
- 证据：`cache_common.py:109-118`
  > `V41_FORMAT = KVCacheFormat(quantized_dtype="e4m3", quantized_elements=512, elements_per_sf=32, ..., sf_format="ue8m0", note="V41, main/swa kv.")`
- 证据：`cache_engine.py:216-218`
  > `swa_cache: [num_state_seqs, swa_block_size, 1, 528], uint8. Each V41 token stores [512B FP8 data | 16B UE8M0 scale].`
- 证据：`attn.py:412-432`（SWA cast：`CastArgs("e4m3", self.swa_cache_num_per_channels, round_sf=True, sf_format="ue8m0")`）
- 等级：A
- 意味着什么：512 + 16 = **528 B**，即 MXFP8（per-32 的 UE8M0 scale）。SWA 永远是 FP8，即使主 KV 用 FP4；因为 SWA 是逐层、窗口内高精度的那一份。
- 疑点：无

### #18 SWA 窗口 = 128 token；缓存环 = 134 = 128 + 6
- 证据：`config.py:284`（`window_size=(text["sliding_window"], 0)`）+ HF `config.json`：`sliding_window=128`
- 证据：`cache_engine.py:88-93`
  > `"""swa_block_size 等于 window_size 加上为回退功能额外存的部分。... nextn 需要回退 n 个 token。"""`
  > `return (window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS) if window_size > 0 else 0`
- 证据：`environ.py:129-132`（5 个投机 token → 回退上界 6）
- 等级：A（机制）+ B（128 这个数值）
- 意味着什么：attention 可见窗口是 128；物理环多 6 个位置。代码注释只说明"nextn 需要回退 n 个 token"（`cache_engine.py:91`），文档给出的另一个用途是吸收落盘位置最多 5 个 token 的 overshoot（`doc/...:165`）；从 `get_kv_index` 的槽位公式看，这 6 个位置同时也容纳 5 个草稿 token 的写入（E，见 #27）。
- 疑点：`doc/...:183` 说"重放时 query i 只看 max(s, i−127)..i"，也是 128 个位置；但 `:191` 说最大命中长度 L−134，把 6 也算进"窗口"，两处措辞需要统一（见 §4）。

### #19 40 个主干层全部有 SWA cache，没有全量注意力层
- 证据：`config.py:104/115/140`（三种 spec 都设 `sink_mode="swa"`）
- 证据：`cache_engine.py:682-687`
  > `# 有意不检查 use_swa：use_swa=False 的层（packed-KV/YoCo consumer）也占一个从不读写的 SWA slot。`
  > `return cfg.cfg.get("sink_mode", "") == "swa"`
- 证据：`cache_engine.py:695-699`（`_is_full` 要求 `kv_compress_ratio==0 and sink_mode!="swa"`）
- 证据：`cache_engine.py:466-468`
  > `# KVCacheManager 目前不支持 full attention，避免套用压缩 KV 的布局。`
  > `raise AssertionError(f"KVCacheManager 不支持 full attention 层：{layer_id=}, ...")`
- 等级：A
- 意味着什么：V4.1 不存在"某些层全量、某些层滑窗"的混合；每一层都有 128 窗口，全局信息一律通过压缩/共享 KV 提供。代码层面 `_is_full` 永远为 False。
- 疑点：`_has_SWA` 的注释暗示存在 `use_swa=False` 的 packed-KV 层；V4.1 配置没有这种层，但布局代码保留了该分支。

### #20 状态缓存的三段布局：stored SWA | dropped SWA | compression state
- 证据：`cache_engine.py:221-241`
  > `[stored region: SWA]`
  > `[dropped region: YOCO consumer SWA | compression state]`
  > `state_size_to_disk 仅包含需落盘 SWA 的 inference 窗口。`
- 证据：`cache_engine.py:265-306`（`swa_store_num_layer`、`swa_drop_num_layer`、`compress_bytes`、`state_size_to_disk = self.swa_store_total_bytes_inference_pad`）
- 证据：`cache_engine.py:113-116`（`_is_dropped_swa_layer`：`layer_id >= n_prefill_layer`）
- 等级：A
- 意味着什么：stored = layer 0–19 的 SWA（20 层，落盘）；dropped = layer 20–39 的 SWA（20 层，靠重放恢复）+ 3 个 C2 层的压缩状态（fp32，不落盘）。
- 疑点：为什么 decoder 的 SWA 还占 HBM（dropped region）而不是直接不分配？因为一次请求内 decode 需要最近 128 token 的 SWA；只有"跨请求恢复"才靠重算。

### #21 `ica_state` 在 V4.1 是死代码：indexer 用 KProject，不再有独立 K 累加器
- 证据：`cache_engine.py:123-127`
  > `def _needs_index_state(attn_module): if attn_module.cfg.get("index_topk", 0) <= 0: return False; index_cfg = ...; return index_cfg is None or not index_cfg.get("use_kproj", False)`
- 证据：`mla_attn.py:271-273`（`assert cfg.attn_index.get("use_kproj", False), "only the use_kproj=True indexer is supported"`）
- 证据：`index_attn.py:537`（`state_cache ... None on the KProject path.`）
- 等级：A
- 意味着什么：V4.1 的 index K 是"主压缩 KV 的再投影"，天然与主 KV 同寿命，不需要单独维护压缩状态；`KVCache.ica_state` 字段与相关分配逻辑只服务旧版 indexer。
- 疑点：保留死代码会增加读者理解成本；书中应明确"V4.1 不分配 ica_state"。

### #22 块状 KV 每 token 890 B 的精确推导（与官方数字一致）
- 证据（组件字节，A 级）：main 288 B/压缩 token（#16）、index K 68 B/压缩 token（#13）；KV 源层 = `[2,8,14,20]`（HF `kv_source_layer_ids`），其中 2/8/14 是 ratio 2、20 是 ratio 1（HF `compress_ratios`）
- 证据（按 ratio 摊算，A 级）：`cache_engine.py:433-438`（index：`token_per_block = block_size // ratio`，`delta = token_per_block * self._qk_topk_bytes`）；`:447-461`（main：同样 `block_size // ratio`）
- 推导（E 级，纯算术）：
  - main：3 个 C2 源层 × 288/2 = 432；1 个 C1 源层 × 288 = 288 → **720 B/token**
  - index：3 × 68/2 = 102；1 × 68 = 68 → **170 B/token**
  - 合计 **890 B/token**
- 等级：A（各分量与公式）+ E（合并）+ B（官方 890 的独立佐证：技术报告摘要"reduce its global KV cache footprint (always in HBM) to 890 bytes per token"，见 R01 笔记 #B5）
- 意味着什么：890 不是近似值，而是精确值；它只包含块状 KV（main + indexer K），不含 SWA KV 与状态。按 1M token 算是 890 MiB/seq；按 128K 算是 111.25 MiB/seq。
- 疑点：若把 index-K 误算成 8 个源层（2/8/14 各一份 + 20/24/28/32/36 各一份），index 部分变成 3×34 + 5×68 = 442 B，总量 1,162 B/token，与官方数字不符——反过来说，890 精确验证了"index-K 只有 4 份"这一读法。

### #23 每 128-token KV block 的字节数：113,920 B
- 证据：`model_metadata_utils.py:133-140`（`resolved_tokens_per_block=128`、`kvcache_align_to=128`）
- 证据：`cache_engine.py:433-461`（每层 `token_per_block = 128 // ratio`，delta 按层累加并做 32B pad）
- 计算（E）：layer 2/8/14 各 64×(288+68)=22,784 → 68,352；layer 20 128×356=45,568；合计 **113,920 B / 128 token** = 890 B/token，无 padding 余量（每段都是 128 的倍数）。
- 等级：A（常量）+ E（乘法）
- 意味着什么：KV pool 按 block 分配时，每 block 113,920 B；这也是 3FS 上每个 128-token 单位的裸大小（不含元数据）。
- 疑点：3FS/持久化层是否额外加 header、索引，代码里看不到（`kvcache-store` 只在文档描述）。

### #24 KV 只由 4 个层产生；indexer 模块 8 个但 index-K 只有 4 份
- 证据：HF `config.json`：`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、`index_k_source_layer_ids` 缺省
- 证据：`config.py:71-75`
  > `layer_index_module_map=_source_map(text["index_source_layer_ids"], n_layer),`
  > `layer_kv_map=_source_map(text["kv_source_layer_ids"], n_layer),`
  > `layer_index_k_map=_source_map(text.get("index_k_source_layer_ids", text["kv_source_layer_ids"]), n_layer),`
- 证据：`cache_engine.py:174-187`（index K 优先用 `layer_index_k_map`，其缺省才是 `layer_index_module_map`）
- 等级：B（数值）+ A（映射逻辑）
- 意味着什么：`layer_index_k_map == layer_kv_map`（因为 config 缺省）→ layer 20–39 的 5 个 indexer（20/24/28/32/36）共用 layer 20 的 ica_cache；layer 2–7/8–13/14–19 各共用 2/8/14 的 ica_cache。主 KV 同理：layer 2–7 用 2 的、8–13 用 8 的、14–19 用 14 的、20–39 全部用 20 的（layer 0–1 没有块状 KV）。
- 疑点：indexer 24/28/32/36 有自己的 Q/K 投影和 `weights_proj`，却读同一份 index K；这意味着它们的"视角"差异只来自 Q 与权重。设计上说得通（K 是内容摘要），但书中要讲清"共享 K、不共享 Q"。

### #25 代码里没有叫 CSA 的标识符
- 证据：`grep -rn "CSA\|Compressed Sparse" --include=*.py --include=*.md`（排除 vendor）无命中；全仓库仅有 `sparse_stage`（`mla_attn.py:113/122/249-250`、`config.py:39/100/106`）
- 证据：`mla_attn.py:249-250`（`assert self.sparse_stage in (0, 2)`）；`config.py:39`（gpt 默认 `sparse_stage=2`）；`config.py:100-107`（ratio 0 层覆盖为 0）
- 等级：A（不存在）+ E（命名对应）
- 意味着什么：如果外部资料说 V4.1 用 "Compressed Sparse Attention 2"，代码里的对应物是 `sparse_stage=2` = 压缩 KV + top-k 稀疏 index；`sparse_stage=0` = 纯 SWA。代码没有 CSA/CSA2 这个名字，也没有 stage=1。
- 疑点：`sparse_stage` 的 0/2 命名暗示曾存在 stage 1（可能是已删除的 dense compressed attention）；`derive_compress_ratios` 仍保留 dense/sparse 两类（`model_metadata_utils.py:29-34`），说明 dense 路径还在，只是 V4.1 不用。

### #26 每序列状态缓存总量：2,916,672 B；落盘 1,415,040 B
- 证据：`cache_engine.py:261-306`（`swa_size_per_token=528`、`swa_block_size_inference=134`、`swa_num_layer=40`、`swa_store_num_layer=20`、`compress_bytes` 只累加 `owns_main` 的 C2 层）
- 计算（E）：
  - 40 层 SWA：40 × 134 × 528 = 2,830,080 B（其中 stored 20 层 = 1,415,040 B）
  - C2 压缩状态：3 层 × 7 × 2 × 512 × 4 B = 86,016 B
  - 合计 2,916,096 B，按 `lcm(32,528)=1056` 对齐 → **2,916,672 B/seq**（2.78 MiB），padding 576 B
  - `state_size_to_disk = 1,415,040 B`（1.35 MiB）
- 等级：A（字段与公式）+ E（算术）
- 意味着什么：状态缓存是 O(1)/seq（约 2.8 MB），与上下文长度无关；块状 KV 是 O(L)（890 B/token）。两条曲线在 L≈3,277 token 处相交（E）——短上下文时状态反而更贵。
- 疑点：`state_blocks` 的槽位数 = `int(max_num_seqs*1.2)+1`（`cache_engine.py:201-202`），实际 HBM 预留还要乘 1.2 冗余。

### #27 DFlash 草稿层复用同一套 SWA 布局，不新增块状 KV
- 证据：`deepinfer/model_executor/models/mtp/dflash.py:140-144`
  > `self.window_size = cfg.dflash_cfg.sliding_window`
  > `# DFlash 的 window_size 必须与 main model 的一致，不然无法使用相同的 swa cache 布局。`
  > `assert self.window_size == getattr(cfg, "window_size", (-1, -1))[0]`
- 证据：`index_attn.py`/`cache_engine.py` 的布局只遍历 `iter_runtime_attn_modules(attn_module_list, cfg.num_hidden_layers)`（前 40 层，`cache_engine.py:631`）
- 等级：A（window 一致）+ E（"不新增块状 KV"的推断：3 个草稿层 ratio=0，没有 PackedWkv）
- 意味着什么：投机解码的 KV 不额外占块状 KV 预算；草稿 token 的 KV 写进 SWA 环的前瞻槽位（6 个额外位置），这正是 #18 的 6 的用途之一。
- 疑点：草稿 KV 的具体 slot 映射在 `tilelang_ops/get_kv_index.py`（如 `cuda/get_kv_index.py:115` 的 `(ap + 1 + bx) % swa_block_size`），本次未逐行核对，标 E。

## 3. 必答问题逐条回答

### Q1. MLA 如何使用？`index_attn.py` 在做什么？`compress.py` 压缩什么？代码中是否存在 CSA 机制？

**MLA 的使用（#3–#8）**
1. **参数**：Q 低秩 `5120 → 1280 →(RMSNorm)→ 64×512`；KV 只有一条 `5120 → 512`（448 nope + 64 rope）的 A 投影，没有 B（`mla_attn.py:198-246`）。`num_key_value_heads=1`，64 个 query 头共享它。
2. **RoPE**：打在 latent 的末尾 64 维（`offset = 512−64 = 448`，`attn.py:436-442`）；因此 512 维 latent 同时承担 nope 与 rope，不需要 V3 的 576 维。
3. **注意力**：prefill 用 `flash_mla_sparse_fwd`，decode 用 `flash_mla_with_kvcache`（`attention_backend.py:26-146`）；两者都是 MQA（`num_heads_k=1`）且 head_dim 必须 512（FlashMLA 接口断言）。
4. **KV 精度**：SWA 走 FP8 E4M3（per-32 UE8M0），压缩 KV 走 FP4 E2M1（per-16 E4M3），decode 时分别作为主 cache 与 `extra_k_cache`（#8、#16、#17）。
5. **输出**：64 个 512 维 context 按 8 组（每组 8 头）经 `Wv` 降到 8×1024，再 `out_proj` 回 5120（#6）。
6. **融合**：`Wq.A` 与 `Wkv.A` 合成一个 `5120→1792` GEMM（#4）；decode 还有 `fused_norm_rope_attn_rope_cast`，把 q 的 norm/rope/量化 + 稀疏 attention + 输出侧共轭 RoPE + FP8 量化串成一个 kernel（`attention_backend.py:107-121`、`fused_norm_rope_attn_rope_cast.py:36`）；`Wv` 不在这个 kernel 里，是随后一次 `deep_gemm.fp8_einsum`（`attn.py:215-221`）。

**`index_attn.py` 的作用（#9–#13）**
- 它是**稀疏选择器**，不是第二套 attention：32 头 × 128 维的 MQA，Q 来自 `q_lora`（与主 Q 共用 A，`q_B` 单独升维），K 来自"主压缩 KV 的再投影"（`KProject: 512→128` + RMSNorm + RoPE）。
- 打分：`p_sum = Σ_h w_h · relu(q_h·k)`，Q/K 均 FP4，权重 BF16（`index_attn.py:284-325`）。
- 选择：每 query top-512（`index_topk=512`），prefill 输出全局索引（加 `ks` 偏移），decode 输出物理 slot（`index_attn.py:541-544`）。
- 两级：layer 20 先选 2048 个 8-token block（16,384 token），21–39 在候选内 top-512（#11、#12）。
- K 缓存：FP4 E2M1 + UE8M0 per-32 = 68 B/压缩 token（#13），只有 4 份（#24）。

**`compress.py` 压缩什么（#14、#15）**
- 压缩对象是**主 KV 的 512 维 latent 在时间轴上的密度**，不是 hidden、不是权重。
- C2（ratio 2）：对相邻 2 个 token 的 `[score|latent]` 做逐通道 softmax 加权求和（`(latent * softmax(score, dim=0)).sum(0)`），输出 1 条 512 维 KV；`ca_state` 保留 7 个 token 的原始 score/latent 供跨 step 窗口使用。
- C1（ratio 1）：恒等（无 Gate、无 state），每 token 一条 KV。
- 压缩后接 `norm → RoPE → cast → store`，一个融合 kernel 完成（`compress.py:131-167`）。

**CSA 是否存在（#25）**
- 代码中**不存在** `CSA` 标识符（全仓库 grep 无命中），也没有 "Compressed Sparse Attention" 字样。
- 对应机制以 `sparse_stage=2` + `compress_ratios` + `index_topk` 的形式存在：`sparse_stage=2` 表示"压缩 KV + top-k 索引"，`sparse_stage=0` 表示纯 SWA。若外部资料称 "CSA2"，应理解为这套机制的命名，而不是代码里的另一个模块（E）。

### Q2. SWA 滑动窗口大小是多少？哪些层使用 SWA，哪些层使用全量注意力？

- **窗口 = 128 token**（HF `sliding_window=128`；`config.py:284`）。物理缓存环 = **134**（128 + 6；`cache_engine.py:88-93`、`environ.py:129-132`），SWA KV 每 token 528 B（#17、#18）。
- **所有 40 个主干层都使用 SWA**：三种 spec 都设 `sink_mode="swa"`（`config.py:104/115/140`），`use_swa` 默认 True（`attn.py:93`），`_has_SWA` 只认 `sink_mode`（`cache_engine.py:682-687`）。3 个 DFlash 层也声明 `sink_mode="swa"`（`config.py:140`）。
- **没有任何层使用"全量注意力"**：`_is_full` 要求 ratio 0 且 `sink_mode != "swa"`（`cache_engine.py:695-699`），V4.1 永远不满足；`KVCacheManager` 对 full attention 层直接抛异常（`cache_engine.py:466-468`）。
- 按"全局信息怎么来"分三层：
  - **layer 0–1（ratio 0）**：纯 SWA，只有 128 窗口，没有块状 KV（`config.py:99-108`）。
  - **layer 2–19（ratio 2，C2）**：SWA + 压缩 KV（2:1），top-512 稀疏选择；KV 源层 2/8/14。
  - **layer 20–39（ratio 1，C1，YOCO consumer）**：SWA + 1:1 共享 KV（layer 20 一份，20 层共用），top-512 稀疏选择。
- 若按"是否覆盖全上下文"说，20–39 层最接近全量注意力：它们对一条未做时间压缩的共享 KV 做稀疏 top-512；但这条 KV 是全模型共享的 1 份，不是每层各自的全量 KV。
- SWA 落盘只存前 20 层（#20），decoder 的 SWA 在 dropped region，靠 128-token 有界重放恢复（`doc/...:183-187`）。

### Q3. KV 的种类与每 token 体积；核实/推算"每 token 约 890 字节"

**四种存储（#13、#16、#17、#21、#26）**

| 种类 | 存什么 | 格式 | 每 token 字节 | 归属层 |
|---|---|---|---|---|
| 主 KV `ca_cache` | 每 ratio 个输入 token 一条 512 维 post-RoPE latent（448 nope + 64 rope） | FP4 **E2M1** 数据 + **E4M3** scale，**per-16**（`V41_FP4_FORMAT`） | **288 B / 压缩 token**；C2 摊成 144 B/输入 token | 2、8、14（C2）、20（C1），共 4 份 |
| indexer K `ica_cache` | 每条主 KV 再投影出的 128 维 index-K（KProject + RMSNorm + RoPE） | FP4 **E2M1** + **UE8M0** scale，**per-32**（`FP4_INDEX_KV_FORMAT`，即 MXFP4） | **68 B / 压缩 token**；C2 摊成 34 B/输入 token | 2、8、14、20，共 4 份（24–36 的 indexer 复用 20 的） |
| SWA KV `swa_cache` | 每层每 token 的 512 维 post-RoPE `kv_swa` | FP8 **E4M3** + **UE8M0** scale，**per-32**（`V41_FORMAT`，即 MXFP8） | **528 B / token / 层** | 全部 40 层（前 20 层落盘） |
| 压缩状态 `ca_state` | C2 环 `[7, 2, 512]` fp32：槽 0 = Gate score，槽 1 = latent | FP32，未量化 | 7×2×512×4 = **28,672 B / seq / 层**（3 层共 86,016 B） | 2、8、14 |
| index 状态 `ica_state` | 旧版 indexer 的 K 累加器 | fp32 | **0**（V4.1 不分配，#21） | 无 |

**"890 B/token"的精确推导（#22）**
- 每压缩 token 每 KV 源层：main 288 + index 68 = **356 B**；C2 层按 2:1 摊半 = **178 B/输入 token**。
- 全模型只有 3 个 C2 源层 + 1 个 C1 源层：
  - `3 × 178 + 356 = 534 + 356 = 890` B/输入 token
  - 等价写法：`3 × (288+68)/2 + (288+68) = 890`
- 分项：main = 3×144 + 288 = **720 B**；index = 3×34 + 68 = **170 B**。
- 与官方口径一致：技术报告摘要"global KV cache footprint (always in HBM) ... 890 bytes per token"（R01 笔记 #B5）；`890 B × 1,048,576 = 890 MiB` 每 1M token。
- **890 不包含**：SWA KV（528 B/token/层，40 层共 2.83 MB/seq，固定开销）、压缩状态（86 KB/seq）、对齐 padding。把 SWA 加进来做"每 token"比较是错的——SWA 是 O(1)/seq，块状 KV 是 O(L)。
- **一致性校验**：如果 index-K 不是 4 份而是 8 份（把 24/28/32/36 也算成独立 K），index 部分会变成 3×34 + 5×68 = 442 B，总量 1,162 B/token，与 890 不符；890 反过来证明"index-K 只有 4 份"的读法正确。

## 4. 新发现的问题

1. **R02b 笔记 §5.3 的 KV 预算用错了格式**：它按 528 B/token（MXFP8）算块状 KV，得出 1,320 B/token；但 `cache_engine.py:327-332` 在 `use_fp4_sparse_kv=True`（V4.1 默认）时用的是 `V41_FP4_FORMAT` = 288 B。正确值是 890 B/token（1M 上下文 890 MiB/seq），建议修正，否则与官方 890 冲突。
2. **"block KV 在 3FS" vs "always in HBM"**：`doc/inference_deployment_qa.md:141` 说块状 KV 存专用 3FS 集群、不在本地内存；技术报告摘要说 890 B 是 "global KV cache footprint (always in HBM)"。两句需要区分"运行时驻留"与"持久化位置"，书中不要混用。
3. **`doc/...:183` 与 `:191` 对"窗口"的用法不一致**：前者说可见范围 128（i−127..i），后者把最大命中长度写成 L−134 并把 6 算进"窗口"；实际是 128 可见 + 6 回退槽。建议统一措辞为"128 token 窗口 + 6 个回退位置"。
4. **为什么 decoder（20–39）反而用不压缩的 C1？** 直觉上 decoder 更该省。可能的解释：decoder 只有一份 KV（20 层共享），C1 的全分辨率换来更低的精度损失，而总量已由"共享"解决；C2 只用在 encoder 的前 18 层以进一步压 prefill 的 KV 写入量。未找到注释或文档说明，标 E。
5. **`ica_state` / `dense` 压缩路径是死代码还是备用**：`_needs_index_state` 与 `derive_compress_ratios` 的 dense 分支在 V4.1 都不走（`use_kproj=True`、`index_topk=512`）。书中若引用 `ica_state` 会误导读者。
6. **indexer 共享 K、不共享 Q**：24/28/32/36 四个 indexer 有自己的 `q_B`/`weights_proj`，但读 layer 20 的 index-K cache。这是刻意的设计还是权宜？代码只断言了 segment 关系（`cross_layer.py:93-96`），没有说明。
7. **`is_fp8_kvcache=True` 的硬编码**（`attention_backend.py:134`）与 FP4 主 KV 的兼容性：当前 V4.1 所有层都有 SWA，主 cache 确实是 FP8，所以没问题；但一旦有 `use_swa=False` 的层，主 cache 会变成 FP4，这个 flag 需要重新检查。
8. **DFlash 草稿 KV 的槽位**：`cuda/get_kv_index.py:115` 用 `(ap + 1 + bx) % swa_block_size` 把草稿 token 写进环里"当前位置之后"的槽位；这意味着 6 个额外位置既是回退余量也是投机前瞻区。需要和 R02c 的 MTP 笔记对齐（本次未逐行验证）。
9. **SWA 的 `attn_sink` 全零初始化**（`config.py:103/114/139` 的 `attn_sink_init_value=0`，`mla_attn.py:286-292`）：如果训练后仍是 0，它只是占位；如果被学习，则是一个 per-head bias。权重里是否非零，代码看不出来。

## 5. 可用于图表的材料

### 5.1 层类型与 KV 归属（V4.1-Flash）

| 层号 | 数量 | ratio | 注意力 | 块状 KV | main/index-K 源层 | indexer 模块源层 |
|---|---|---|---|---|---|---|
| 0–1 | 2 | 0 | 纯 SWA（128） | 无 | — | — |
| 2–19 | 18 | 2 | SWA + C2（2:1）top-512 | 有 | 2 / 8 / 14 | 2 / 8 / 14 |
| 20–39 | 20 | 1 | SWA + C1（1:1，共享）top-512 | 有（仅 20） | 20 | 20 / 24 / 28 / 32 / 36 |
| 40–42 | 3 | 0 | DFlash 草稿 | 无（复用 SWA 布局） | — | — |

（前 40 行来自 HF `config.json`；源层划分来自 `kv_source_layer_ids` / `index_source_layer_ids` 与 `config.py:71-75` 的映射逻辑。）

### 5.2 每 token 890 B 的分解（可直接做堆叠柱状图）

| 来源 | 每压缩 token | 摊到每输入 token | 小计 |
|---|---|---|---|
| main KV @ layer 2 | 288 | 144 | 144 |
| main KV @ layer 8 | 288 | 144 | 144 |
| main KV @ layer 14 | 288 | 144 | 144 |
| main KV @ layer 20 | 288 | 288 | 288 |
| index K @ layer 2 | 68 | 34 | 34 |
| index K @ layer 8 | 68 | 34 | 34 |
| index K @ layer 14 | 68 | 34 | 34 |
| index K @ layer 20 | 68 | 68 | 68 |
| **合计** | **356/层** | — | **890** |

### 5.3 每 128-token block 的 KV 字节

| 层 | ratio | block 内 KV token 数 | main | index K | 小计 |
|---|---|---|---|---|---|
| 2 | 2 | 64 | 18,432 | 4,352 | 22,784 |
| 8 | 2 | 64 | 18,432 | 4,352 | 22,784 |
| 14 | 2 | 64 | 18,432 | 4,352 | 22,784 |
| 20 | 1 | 128 | 36,864 | 8,704 | 45,568 |
| **合计** | | | **92,160** | **21,760** | **113,920 B / block** |

### 5.4 StateBlock 布局（每序列）

```
[ stored region: 20 层 SWA × 134 × 528 B = 1,415,040 B ]  ← state_size_to_disk，落盘
[ dropped region: 20 层 SWA × 134 × 528 B = 1,415,040 B ]  ← decoder，靠重放
[                3 层 C2 ca_state × 7×2×512×4 = 86,016 B ]  ← 不落盘
合计 2,916,096 B → 按 lcm(32,528)=1056 对齐 → 2,916,672 B/seq（2.78 MiB）
```

### 5.5 一次 query 的可见 KV（prefill 与 decode 相同）

- 对 ratio>0 的层：128 个 SWA token（当前窗口）+ 512 个 top-k 压缩 token = **最多 640 个 key**（layer 0–1 只有 128 个 SWA key）。
- prefill：`kv_index_cat` 把两段索引拼成 `[total_q, 640]`，一次 `flash_mla_sparse_fwd`（`attn.py:603-625`、`kv_index_cat.py:16-24`）。
- decode：`flash_mla_with_kvcache` 的 `indices`（SWA，640 中的前 128）+ `extra_indices_in_kvcache`（压缩，后 512）（`attn.py:640-669`）。

### 5.6 时序（decode 单步，单层）

1. `hidden_states` → `q_a_kv_swa_proj`（5120→1792，融合 Q.A + Wkv.A）。
2. 拆成 `q_a`（1280）与 `kv_swa`（512）；`kv_swa` → RMSNorm → RoPE(末尾 64) → cast E4M3/UE8M0 per-32 → 写 `swa_cache` 当前 slot（`attn.py:402-451`）。
3. `q_a` → `q_b_proj`（1280→64×512 + 32×128）→ 拆出 `index_q_b`（32×128）；`q_b` 的 Norm/RoPE/FP4 量化在 decode 融合 kernel 内完成，`index_q_b` 由 `_quantize_q` 单独做 RoPE + FP4（`attn.py:453-463`、`index_attn.py:164-179`）。
4. `index_attn`：读共享 index-K cache（lazy load + FP4），算 `p_sum`，top-512（`index_attn.py:456-515`）。
5. 主 attention：`flash_mla_with_kvcache`，SWA FP8 主 cache + 压缩 FP4 extra（`attn.py:628-670`）。
6. 输出 context（64×512）→ 8 组 `Wv`（4096→1024）→ `out_proj`（8192→5120）（`mla_attn.py:221-229`、`282`）。
