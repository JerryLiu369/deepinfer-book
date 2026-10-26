# R02c MTP / DFlash 草稿模型、视觉编码器、参数总览

## 0. 元信息

- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（本地 `research/sources/DeepInfer`）
- 权重与配置来源：HF `deepseek-ai/DeepSeek-V4.1-Flash-1017`（`config.json`、`model-00001..00048-of-00048.safetensors` 头部元数据、`README.md`、`inference/{config.json,model.py,vision.py}`）；抓取日期 2026-10-23
- 查阅的主要文件：
  - `deepinfer/hugging_face/config.py`、`deepinfer/hugging_face/processor.py`
  - `deepinfer/model_executor/models/mtp/dflash.py`、`models/decoder_layer.py`、`models/deepseek.py`、`models/deepseek_vl.py`
  - `deepinfer/model_executor/models/v4_modules/{attn.py,heads.py,mlp.py}`
  - `deepinfer/model_executor/models/vision_encoder/{vit.py,mlp_projector.py,vl_aligner.py}`
  - `deepinfer/model_executor/layers/{sampler.py,tilelang_ops/dflash.py,tilelang_ops/schedule_mtp_varlen.py,tilelang_ops/get_kv_index.py,tilelang_ops/image_embed_ops.py}`
  - `deepinfer/environ.py`、`deepinfer/worker/cache_engine.py`、`deepinfer/ve/processor/image_processor.py`
- 未运行任何代码（无对应硬件）。所有数字要么来自权重元数据，要么是手算，方法写在条目里。

## 1. 三个最重要的发现

1. **763.2B 总参数里只有 566.3B 是"稠密主干"**：196.9B（25.8%）是第 1、14 层的两张 Engram n-gram 查表，靠 sparse gather 命中，几乎不参与每 token 的算术。
2. **"3 层 MTP" 与 "5 个草稿 token" 是两个独立超参**：`num_nextn_predict_layers=3` 是堆叠的草稿层数（权重 `mtp.0/1/2`），`dspark_block_size=5` 是每步一次性猜的位置数；3 层各自对同一 5 位置块做前向，块内**非因果**，token 间依赖靠 rank-256 的 Markov 头在采样时补回。
3. **置信度头不是打分装饰，而是显存/算力分配器**：它预测的每位置接受率被累乘后做**全 batch 全局 top-k**，决定下一轮每个请求实际送进主模型验证的 query 数（`compute_seqlens_q_by_confidence_rates`），因此 MTP 的验证长度在 GPU 上按 batch 动态分配，CPU 侧不知道。

## 2. 事实条目

### #1 总参数 763.2B，构成以 FP4 路由专家和 FP8 Engram 表为主
- 证据：HF 权重 48 个 shard 的 safetensors 头部逐 tensor 求和（脚本见第 5 节），与 HF API `safetensors.total = 763,205,315,794` 完全一致；权重张量合计 510.3 GB（48 个文件 510.5 GB）
- 分项（参数 / 存储）：

  | 部分 | 参数 | 存储 |
  |---|---|---|
  | 路由专家 40 层 × 384（FP4 packed，`I8` 存 2 值/字节） | 543.58 B | 271.8 GB |
  | Engram 表（第 1、14 层，FP8 `[384006168,256]` / `[384016682,256]`） | 196.93 B | 196.9 GB |
  | MTP/DFlash 3 层（含 3×128 专家） | 14.23 B | 7.5 GB |
  | 注意力 MLA 40 层 | 5.13 B | 5.15 GB |
  | 共享专家 40 层 × 1 | 1.42 B | 1.42 GB |
  | token embedding / lm_head | 0.662 / 0.662 B | 1.32 / 1.32 GB |
  | 视觉 ViT 32 块 | 0.412 B | 0.82 GB |
  | MoE gate / aligner / HC | 0.079 / 0.073 / 0.039 B | 0.16 / 0.15 / 0.16 GB |
  | 合计 | **763.21 B** | **510.3 GB** |

- 等级：B（权重元数据）+ E（分项求和；`layers.0.ffn.experts.0.w1.weight` 为 `I8 [2304, 2560]`，2560 = 5120/2，即 4bit 两两打包，逻辑参数 = 2×元素数）
- 意味着什么：讲"763B 模型怎么跑"时必须先拆开这三块，否则权重体积（510GB）和参数量（763B）看起来对不上。
- 疑点：无。

### #2 每 token 激活参数约 16.4B（含词表），不含词表 15.1B
- 证据：按 `config.json` 尺寸手算，单层构成 = 注意力 126.62M（`wq_a 1280×5120`、`wq_b 32768×1280`、`wkv 512×5120`、`wo_a 8192×4096`、`wo_b 5120×8192`）+ gate 1.97M + 共享专家 35.39M + 6×路由专家 6×35.39M = 212.34M + HC 0.98M ≈ **377.3M/层**
- 40 层 = 15.09B；加 embedding 0.662B + lm_head 0.662B = **16.42B**；占总量 2.15%（若把 Engram 196.9B 视为非稠密参数，则 16.42/566.3 = 2.90%）
- 等级：E（推理过程即上式；尺寸来自 `config.json`，tensor 形状来自权重头部，均为 A/B）
- 意味着什么：激活比 2.1%，说明这是"极端稀疏 + 超大字表/查表"的组合，不是传统 671B/37B 那种配比。
- 疑点：Engram 查表命中量未计入（每 token 每层最多 3 个 n-gram × 8 头 × 256 维 ≈ 6K 个值，可忽略，但属于 E 级估计）。

### #3 40 层全是 MoE，没有 dense 前缀层；词表 129,280，上下文 1,048,576
- 证据：`config.json` `text_config`：`num_hidden_layers=40`、`n_routed_experts=384`、`n_shared_experts=1`、`num_experts_per_tok=6`、`moe_intermediate_size=2304`、`hidden_size=5120`、`vocab_size=129280`、`max_position_embeddings=1048576`、`sliding_window=128`、`scoring_func=sqrtsoftplus`、`routed_scaling_factor=1.5`、无 `first_k_dense_replace` / 无 `intermediate_size`
- 权重侧核对：`layers.{0..39}.ffn.gate.weight` 与 `layers.{l}.ffn.experts.0..383` 全部存在（脚本枚举，0 缺失）
- 代码侧：`deepinfer/model_executor/models/v4_modules/mlp.py:167-174`（`num_topk`、`topk_method=noaux_tc`、`norm_topk_prob` 断言）、`mlp.py:148-152`（`n_routed_experts` 与 EPLB 物理副本数区分）
- 等级：A（仓库+权重）/ B（HF config）
- 意味着什么：第 3 章讲层结构时不能沿用 V3"前 3 层 dense"的叙述，V4.1-Flash 每层都有 384 专家。
- 疑点：`config.json` 没有 `intermediate_size`，说明不存在 dense FFN 分支，与权重一致，可确认。

### #4 MTP 在代码里有三个名字：`nextn` / `mtp.*` / DFlash（dspark_*）
- 证据：`config.py:300` `num_nextn_predict_layers=nextn`；`config.py:177-192` `_build_dflash_cfg` 把 `dspark_block_size`、`dspark_target_layer_ids`、`dspark_markov_rank`、`dspark_n_routed_experts` 装进 `cfg.dflash_cfg`；权重命名空间是 `mtp.0/1/2.*`（HF 头部）；`config.py:308` 挂 `dflash_cfg`
- 等级：A
- 意味着什么：读代码时"DFlash 层"= "MTP 层" = "nextn 层"，`dspark_*` 是同一组超参；书里应统一到一个名字并给出别名表。
- 疑点：`dspark` 与 `DFlash` 哪个是模型侧原名、哪个是引擎侧改名，仓库里没有说明（含义未完全确认）。

### #5 每步草稿数是 5，草稿层数是 3，两个数字都写死在配置里
- 证据：`config.json` `dspark_block_size=5`、`num_nextn_predict_layers=3`；`mtp/dflash.py:141` `self.dflash_block_size = self.max_MTP_value`；`environ.py:129-131` `DEEPINFER_USE_NEXTN_DECODE = EnvVar.int(default=5)`（注释："最大支持投机几个；默认 5，显式设为 0 可关闭投机解码"）
- 等级：A
- 意味着什么：引擎的草稿宽度由环境变量决定，默认 5 与权重训练的 block size 一致；改这个 env 不会改权重，但会改变草稿分布的语义（见第 4 节第 2 条）。
- 疑点：`dflash.py:141` 用 env 覆盖 `dspark_block_size`，代码里没有任何 `assert dflash_block_size == cfg.dflash_cfg.dspark_block_size`。

### #6 草稿块 = 最后接受位置之后的连续 5 个位置，块内共享同一份 KV 索引
- 证据：`tilelang_ops/get_kv_index.py:169-197`
  > `accepted_pos = positions[ftp] + last_valid_offset`；`draft_positions = accepted_pos + 1 + arange(dflash_block_size)`
  > `kv_len = window_size + dflash_block_size`；`kv_index_per_req.unsqueeze(1).expand(-1, dflash_block_size, -1)`
- 等级：A
- 意味着什么：5 个 query 位置看到的是同一段 `[accepted_pos-127 … accepted_pos+5]`，即 128 窗口 + 整个草稿块，**块内非因果**；draft 自己的 K 写在 `newx_slot_mapping = draft_positions % swa_block_size + state_offset`（同一条 128 槽环形 SWA 缓存）。
- 疑点：块内非因果意味着位置 i 的 hidden 会看到位置 j>i 的 noise token 嵌入；这一点从索引构造能确定，但训练时是否同构无法从仓库确认（E 级推测：训练也用同一 block 布局）。

### #7 草稿输入：目标层（37/38/39）的"注意力输入残差"拼接 15360 维 → 5120
- 证据：`models/decoder_layer.py:144-145` `self._store_dflash_input(residual.residual, dflash_input)`（在 `attn_norm_forward` 之后取残差）；`models/deepseek.py:73-74` `dflash_input_size = hidden_size * len(target_ids)`；`mtp/dflash.py:148-153` `projection = ReplicatedLinear(hidden*n_target, hidden)` + `projection_norm`；权重 `mtp.0.main_proj.weight [5120, 15360]`（FP8 e4m3 + `[160,480]` block scale）
- 等级：A
- 意味着什么：草稿模型不吃主模型最后一层输出，而吃三个中间层（`dspark_target_layer_ids=[37,38,39]`）的残差流，这与 V3 MTP 只吃最后一层 hidden 不同——是"多层特征融合式"草稿。
- 疑点：`dspark_target_attn_input=True`（`config.py:183`）说明取的是 attention 输入而非 MLP 输出；为什么选这个位置仓库没写理由（E：与 HC 的 norm 前状态一致，便于复用 norm kernel）。

### #8 主 K 由 `main_kv_proj` 从 5120 维投到 3×512，写入各草稿层自己的 SWA 缓存
- 证据：`mtp/dflash.py:118-123`
  > `main_kv_weights = [_clone_linear_weight(module.block.self_attn.Wkv.A.weight) ...]`
  > `self.main_kv_proj = ReplicatedLinear(cfg.hidden_size, MTP_num_layers * qk_head_dim, weight=merge_linear_weights(main_kv_weights))`
  `mtp/dflash.py:219-228` `main_hiddens = projection_norm(projection(dflash_input))` → `main_k = main_kv_proj(...)` → 逐层 `update_main_cache`
- 等级：A
- 意味着什么：草稿层不复用主模型 MLA 的 latent KV，而是**自己维护一份 512 维 MQA 窗口缓存**（内容来自目标层特征），所以主模型的 KV 压缩收益不会自动传导到草稿侧；同时注释（`dflash.py:117`）说明 KV 权重被刻意复制了两份以换取连续布局，显存略增。
- 疑点：主 K 写入位置 `slot_mapping_store_swa` 与主模型 SWA 写入是同一套 slot 映射，`dflash.py:206-208` 注释解释被拒绝的 draft 位置随后会被 draft key 覆盖，故无需按接受结果屏蔽——这是一个"写后写"依赖，读代码时要留意。

### #9 草稿层有自己的 KV 缓存，层号 40/41/42，窗口必须与主模型一致
- 证据：`mtp/dflash.py:125-136` `DFlashDecoderLayer(..., num_main_layers + i, ...)`；`dflash.py:143-144` `assert self.window_size == cfg.window_size[0]`（注释："不然无法使用相同的 swa cache 布局"）；`config.py:136-143` `_dflash_spec` 只给 `sink_mode="swa"` 与 `attn_sink_init_value`（不含压缩 KV 规格）；`config.py:146-155` `attn_module_list` = 40 层 + 1 个 lm head 占位 + 3 个 dflash spec；`worker/cache_engine.py:874-877` 按 `attn_module_list` 统计 `num_layers`
- 等级：A
- 意味着什么：开 MTP 时 KV cache 引擎多分配 3 层窗口缓存（每层只有 128 槽 SWA，不含压缩 KV），这是投机解码的隐性显存成本。
- 疑点：草稿层缓存的每 token 体积（512 维 latent）与主模型块状 KV 的关系，留给 R02a 核对。

### #10 草稿 token 的生成：块并行前向 + Markov 链式采样（rank 256）
- 证据：`mtp/dflash.py:308-334`
  > `for step_i in range(self.max_MTP_value): logits = logits_next[step_i :: self.max_MTP_value]`
  > `w1 = self.markov_w1(draft_token_ids); logits = self.markov_w2(w1, out_acc=logits)`
  > `_, token_index_next, draft_token_ids, ... = process_sampling(logits, ...)`
  权重：`mtp.2.markov_head.embed.weight [129280, 256]` + `mtp.2.markov_head.head.weight [129280, 256]`（代码属性名 `markov_w1`/`markov_w2`，`dflash.py:164-176`）
- 等级：A
- 意味着什么：一次前向得到 5 个位置的 logits，然后沿 5 步做**只经过 256 维 Markov 头**的自回归修正，代替 5 次完整前向——这是"5 个草稿只花 3 层前向"的关键；代价是块内真实依赖被低秩近似。
- 疑点：`process_sampling(..., tag="mtp")` 里温度/种子如何与主模型共享（`sampling_input.kernel_inputs.seq_seeds`、`sample_positions_table`），需 R05/R07 复核。

### #11 置信度头定义在 `mtp/dflash.py:46-79`，权重来自最后一个 MTP 块
- 证据：`mtp/dflash.py:46-79`
  > `class ConfidenceHead`：`self.proj = ReplicatedLinear(input_size=n_embed + markov_rank, output_size=1, weight=weight)`
  > `confidence_head_fused(hidden, weight=self.proj.weight, temperature=self.temperature[step_i], markov_emb=markov_emb, out=out)`
  挂载：`dflash.py:180-186`（`weight=modules[-1].confidence_head.proj.weight.to(torch.bfloat16)`）；参数容器：`v4_modules/heads.py:37-42` `AcceptRatePredictor`、`heads.py:102-105`（只在最后一个 shift 建 `shared_head` + `confidence_head`）
- 等级：A
- 意味着什么：置信度头是"隐藏态 ⊕ Markov 嵌入"上的单标量线性层（输入 5120+256=5376），fp32 输出；它跟输出头共享 Markov 嵌入，属于极轻量附加参数（5377 个）。
- 疑点：`dflash.py:159` 注释写死 `enable_confidence = True`（"HF DFlash 固定包含 confidence head"），没有配置开关。

### #12 置信度 = 带温度的 sigmoid，温度按 step 可分别配置
- 证据：`tilelang_ops/dflash.py:10-48`（docstring：`sigmoid(cat([hidden, markov_emb], -1) @ weight / temperature)`，输出 fp32 `[n]`）；`environ.py:135-143` `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 默认 `1.0`，允许逗号分隔的 `nextn` 个值
- 等级：A
- 意味着什么：每个草稿位置有独立温度，用于校准"预测接受率"与"真实接受率"的偏差；这是可调推理参数，但对生成质量无直接约束（只影响调度分配）。
- 疑点：默认 1.0 是否为训练时的校准值，仓库与模型卡都没写（不确定）。

### #13 置信度的实际用途：累乘后做全 batch 全局 top-k，决定每请求验证长度
- 证据：`tilelang_ops/schedule_mtp_varlen.py:11-33` + ref `35-100`
  > `cum_rates = torch.cumprod(gathered_rates, dim=1)`
  > `extra_budget = total_q - num_seqs`；`ranks = argsort(flat_rates, descending=True)`；`seqlens_q = 1 + wins`
- 等级：A
- 意味着什么：不是"置信度低于阈值就丢草稿"，而是把 `total_q` 这个全局 query 预算按累计接受率排序分配；高置信请求多验几个 token，低置信请求只验 1 个。这也解释了为什么 `MTPPredictInfo.confidence_rates` 要搬到 CPU（`sequence.py:82`、`sampler.py:262-290`）。
- 疑点：`total_q` 由谁定（scheduler 的 decode token 预算）需 R05 调度章复核；`environ.py` 里 `DEEPINFER_ENABLE_MTP_VARLEN` 控制是否走这条变长路径（`EnvVar.bool()` 无默认值，未确认默认开关）。

### #14 校验是拒绝采样；置信度另有一个"校准误差"监控量
- 证据：`layers/sampler.py:411-470` `_verification(...)` 调 `process_sampling(..., is_verify=is_verify, last_nextn_token_logprobs=..., draft_token_ids_pool=...)`，再 `get_accept_result(...)` 取前缀接受；`tilelang_ops/sampler.py:584-589`
  > `return torch.where(verify_masks, torch.abs(accept_probs - confidence), 0.0).sum()`
  `sampler.py:288` 输出 `batch_confidence_error`
- 等级：A
- 意味着什么：`accept_probs = 1 - residual.sum()`（`tilelang_ops/fused_sample.py:233`）是拒绝采样接受率；`|accept_probs - confidence|` 是置信度头的在线校准指标，可观测但不参与判定。
- 疑点：`batch_confidence_error` 落到哪个 metric（`stats_logger.py`）未逐行核对。

### #15 视觉编码器：32 层、1024 维、16 头、patch 14、MLP 2816，全部硬编码
- 证据：`vision_encoder/vit.py:26-34`
  > `PATCH_SIZE = 14; EMBED_DIM = 1024; DEPTH = 32; NUM_HEADS = 16; MLP_RATIO = 2.7; MULTIPLE_OF = 128; MAX_GRID_SIZE = 4096`
  `vit.py:174` 注释："cfg.vision 余下的键不参与构造：结构全由本文件常量固定"；`config.py:217-234` `_build_vision` 只传 `params.recomputing_every_layer=2`
- 等级：A
- 意味着什么：换任何 `vision_config` 都不会改变 ViT 结构；`config.py` 里那份 vision cfg 只是"校验 + 传一个梯度重算粒度"的壳，读者不要以为它是可配的。
- 疑点：`recomputing_every_layer` 在推理里只影响 residual 物化位置（`vit.py:189`），名字是训练期遗留。

### #16 ViT 单块结构：pre-norm + 融合 QKV 双向注意力（2D RoPE）+ SwiGLU
- 证据：`vit.py:90-123`（`qkv = nn.Linear(dim, dim*3)`，`_flash_attn_fwd(..., causal=False)`）、`vit.py:126-136`（`linear1: 1024→5632`，`swiglu_forward`，`linear2: 2816→1024`）、`vit.py:139-165`（Block = norm1→attn→norm2→mlp，residual 延迟融合）、`vit.py:200-203`（row/col 两路 RoPE 表 gather）
- 参数：每块 = 3072×1024 + 3072 + 1024×1024 + 1024 + 5632×1024 + 1024×2816 + 4×1024 = **12.85M**；32 块 = 411.30M，加 `patch_embed` 0.60M 与末尾 `fc_norm` 1K，视觉侧合计 **411.9M**
- 等级：A（结构）/ E（参数求和，形状见权重 `vision.blocks.N.attn.wqkv.weight [3072,1024]` 等）
- 意味着什么：ViT 是**双向（非因果）注意力**，每张图独立（`cu_seqlens_vit` 变长打包），因此图像编码可以整图并行、和文本解码完全解耦。
- 疑点：`_flash_attn_fwd` 来自 `flash_attn.cute`，NPU 走 host list 分支（`vit.py:212-225`），两条路径数值是否 bit 一致未验证。

### #17 下采样投影层：3×3 unfold → 9216 维，接 `aligner.w1`
- 证据：`vl_aligner.py:69-76`
  > `images_embeds = batch_downsample(images_features, images_spatial_crop, downsample_ratio=self.aligner.cfg.downsample_ratio, mode=self.aligner.cfg.projector_type, ...)`
  > `images_embeds = self.aligner.layers(images_embeds)`
  `image_embed_ops.py:578-579` `permute(2,0,1)` + `F.unfold(kernel_size=3, stride=3)`（行主序 3×3 展平）；`mlp_projector.py:17-22`
  > `modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio ** 2, cfg.hidden_size)]` + `GELU` + `Linear(hidden, hidden)`
- 参数：5120×9216 + 5120 + 5120×5120 + 5120 = **73.41M**（权重 `aligner.w1.weight [5120, 9216]`、`aligner.w2.weight [5120, 5120]`）
- 位置：ViT `fc_norm` 之后、写入文本 embedding 之前，即"ViT 输出 patch 特征 → 3×3 空间合并（通道拼 9 份）→ 两层 MLP 投到 5120"
- 等级：A（结构/位置）/ E（参数求和）
- 意味着什么：下采样不是 pooling，而是**空间转通道**（空间 9 倍降、通道 9 倍升），所以 aligner 首层是 9216×5120 的大矩阵（47.2M，占整个视觉侧 73.4M 的 64%），这是视觉部分唯一"重"的 GEMM。
- 疑点：`batch_downsample` 对不整除的图先零填充再 unfold（`image_embed_ops.py:571-579`），零填充块会进入 MLP 但不参与 loss；推理侧只影响边界 patch 的数值。

### #18 图像 token 几何：每图 ≤1024 个文本 token，段长 = h(w+1)+2
- 证据：`ve/processor/image_processor.py:76-78`
  > `return n_llm_h * (n_llm_w + 1) + 2`（注释：IMAGE_START + 每行 `n_llm_w` 个 IMAGE + 1 个 NEW_LINE + IMAGE_END）
  `image_processor.py:90-92` `n_llm = ceil(n_vit / 3)`；`image_processor.py:95-127` 在 `h*(w+1) = max_n_token-2` 约束下解最大网格；`config.json` `vision_config: max_image_tokens=1024, min_pixels=295936, patch_size=14, downsample_ratio=3`
- 等级：A（代码）/ B（config 值）
- 意味着什么：一张 1024 token 上限的图，ViT 侧最多 `1024*9 ≈ 9216` 个 patch（约 1344×1344 像素），注意力部分的计算量按 patch 数平方增长；`min_pixels=295936 = 544²` 是小图放大下限。
- 疑点：`max_wh_ratio=null`，超宽图没有裁剪上限，长截图会产生很高的 `n_vit_h`（受 1024 token 约束间接限制）。

### #19 视觉编码器的位置：VL 模型在 embedding 阶段就地替换 IMAGE token
- 证据：`models/deepseek_vl.py:26-39`（`DeepseekVLForCausalLM(DeepseekForCausalLM)` 内建 `vision_language_tuner = VisionLanguageInstructTuner(cfg, model.module.module[0], vision_source=model.vision_pipeline.aligner)`）；`deepseek_vl.py:201-218`（先 `embed_tokens(input_ids)`，若有图则调 `vision_language_tuner(input_embeds=hidden_states, ...)`）；`vl_aligner.py:118-139`（`rgb_pad_to_patches` → ViT → downsample → aligner → `replace_input_embeds`）
- 等级：A
- 意味着什么：ViT 不占独立推理阶段，它嵌在 prefill 的 embedding 步骤里、和 LLM 同进程同卡；因此多模态请求的 TTFT 包含"ViT + 3 层 MLP"的串行时间。图像编码能否跳过取决于 **token 前缀命中**：整段图落在 cached prefix 内时，crop 补零、raw 字节为 0，ViT 被 `seqlens_vit != 0` 过滤掉（`deepseek_vl.py:131-137`、`149-155`），并不存在按图粒度的 embedding 缓存。
- 疑点：`deepseek_vl.py:95-110` 有"被 KV cache 完全覆盖的图片替换为等长 dummy"的逻辑，具体 chunk 切片与 prefix 命中判定条件需 R09 prefix caching 章复核。

### #20 MoE 门控对图像 token 用另一套路由偏置 `bias_vl`
- 证据：权重 `layers.N.ffn.gate.bias` 与 `layers.N.ffn.gate.bias_vl` 各 `[384]`（F32）；`v4_modules/mlp.py:232-238`
  > `bias_vl = moe_module.gate.e_score_correction_bias_vl`；`self.e_score_correction_bias_vl = ...`
  `mlp.py:388-390` `bias=..., image_bias=self.e_score_correction_bias_vl if images_seq_mask is not None else None, image_token_mask=images_seq_mask`
- 等级：A
- 意味着什么：视觉 token 与文本 token 走同一组专家池，但用不同的 noaux_tc 偏置做负载/选择校正——这是"多模态不影响专家并行拓扑、只影响选路"的实现方式；对推理的影响是 gate kernel 多一路 mask 输入。
- 疑点：MTP 层也有 `mtp.N.ffn.gate.bias_vl [128]`，但草稿是纯文本（HF 参考实现 `inference/model.py:1129` 注释 "drafts are text: no VL bias"），说明这路参数在草稿里恒不生效。

## 3. 必答问题逐条回答

**Q1 全局拓扑：总参数 / 激活参数 / 层数 / 专家数 / 每 token 激活专家数 / 词表**

| 项 | 值 | 依据 |
|---|---|---|
| 总参数 | **763.2B**（其中 Engram 查表 196.9B、路由专家 543.6B） | #1 |
| 每 token 激活参数 | **≈16.4B**（含 embed+lm_head）；15.1B 不含词表 | #2 |
| 层数 | **40**（全部 MoE，无 dense 前缀） | #3 |
| 路由专家 | **384 / 层**（EPLB 另有物理副本，逻辑数 384） | #3 |
| 共享专家 | **1 / 层**，`moe_intermediate_size=2304` | #3 |
| 每 token 激活专家 | **6 路由 + 1 共享** | #3 |
| 词表 | **129,280**（image token id = 129,264；noise token = 128,799） | #3 |
| 上下文 | 1,048,576（`max_position_embeddings`），SWA 窗口 128 | #3 |
| MTP 部分 | 3 层 × 128 专家 / topk 3 = 14.23B | #1 #4 #5 |

**Q2 MTP / nextn / DFlash：草稿怎么来、置信度头在哪、与主模型什么关系**

1. **草稿 token 生成**（#5–#10）：
   - 主模型 decode 一步时，第 37/38/39 层的 attention 输入残差被写进 `dflash_input`（`decoder_layer.py:145`），主模型 forward 末尾 `update_main_cache` 把它投影成 512 维 K 写进 3 个草稿层各自的 SWA 缓存（`dflash.py:219-228`，`attn.py:946-961`）。
   - 采样出本步的 golden token 后，草稿层取"最后接受位置 +1 … +5"这 5 个位置（`get_kv_index.py:169-174`），位置 0 填 golden token、其余填 noise token 128799（`dflash.py:230-239`）。
   - 3 个草稿层（layer 40/41/42）依次对同一 5 位置块做前向，每层读自己的 128 槽窗口缓存（`dflash.py:241-269`、`decoder_layer.py:224-239`）。
   - 最后一层输出 → `post_norm` → `lm_head` 得 5 组 logits；再沿 5 步用 Markov 头（rank 256）逐步修正并采样，得到 5 个草稿 token 及其 logprobs（`dflash.py:288-336`）。
   - 草稿结果存进 `mtp_info_pool`，下一步 `Sampler._verification` 用它们做拒绝采样验证（`sampler.py:411-470`）。
2. **置信度头位置**：`deepinfer/model_executor/models/mtp/dflash.py:46-79`（`class ConfidenceHead`），参数来自 `mtp.2.confidence_head.proj.weight`（`dflash.py:180-186`），参数容器定义在 `v4_modules/heads.py:37-42`（`AcceptRatePredictor`），计算内核在 `layers/tilelang_ops/dflash.py:10-48`。输出是每个草稿位置的标量接受率（fp32），由 `compute_seqlens_q_by_confidence_rates` 用于分配验证预算（#13）。
3. **与主模型的拓扑关系**（#8 #9）：
   - 附着关系：草稿层是"追加在 40 层之后"的 3 层，层号 40/41/42，KV cache 由同一 `attn_module_list` 统一编排（`config.py:146-155`、`cache_engine.py:874-877`）。
   - 结构关系：与主模型同构的 MLA（64 头 / head_dim 512 / rope 64 / q_lora 1280 / o_lora 1024 / o_groups 8），但专家数 384→128、topk 6→3，FFN 宽度不变（2304）。
   - 数据关系：**单向**——主模型特征 → 草稿（通过 projection + main_kv_proj），草稿不修改主模型任何状态；唯一共享的是 SWA 缓存的 slot 布局（`dflash.py:143-144`）。
   - 参数关系：`main_kv_proj` 复制了一份各层 `Wkv.A` 权重（`dflash.py:117-119`，注释说为了连续布局，显存略增）。

**Q3 视觉编码器：ViT 结构、patch 分辨率、下采样投影层参数与位置**

- ViT：32 个 Block，`hidden=1024`，16 头（head_dim 64），patch 14×14×3，MLP 隐层 2816（`vit.py:24-34`），双向注意力 + 行/列 2D RoPE（`vit.py:200-203`），pre-norm + 延迟残差融合（`vit.py:189`）。总参数 411.9M（32×12.85M + patch_embed 0.60M），存 bf16（#15 #16）。
- patch 分辨率：输入按 `patch_size=14` 切块，图像先对齐到 14 的倍数（`image_processor.py:178-179`），ViT 侧网格 `n_vit = (h/14, w/14)`，RoPE 表按最大网格 4096 预生成（`vit.py:34`、`199-201`）。
- 下采样投影：`projector_type=downsample_mlp_gelu`，`downsample_ratio=3`，`depth=2`，`input_dim=1024`，`hidden_size=5120`（`config.py:225-233`）；实现为 3×3 unfold 通道拼（`image_embed_ops.py:578-579`）→ `Linear(9216, 5120)` → GELU → `Linear(5120, 5120)`（`mlp_projector.py:17-22`），**73.41M 参数**，权重名 `aligner.w1/w2`。
- 位置：ViT `fc_norm` 之后（`vit.py:230`）、`aligner.layers` 之前（`vl_aligner.py:69-76`），输出直接替换文本序列里的 IMAGE token 嵌入（`vl_aligner.py:86-93`）；整条视觉链路嵌在 prefill 的 embedding 阶段（#19）。
- 单图 token：`ceil(h/14/3) × ceil(w/14/3) ≤ 1024`，段长再加每行一个 newline 与首尾标记（`image_processor.py:76-78`，#18）。

## 4. 新发现的问题

1. **权重 metadata 与"参数量"口径不一致的风险**：HF API 报 `safetensors.total=763.2B`，但 FP4 专家的 `I8` 张量是打包存储（元素数 = 参数数/2）。若引用方直接按"763B 参数 × 1 字节"估显存会低估到 510GB 的一半。书中必须写清"763B 参数 / 510GB 权重"。
2. **`dflash_block_size` 由 env 覆盖且无断言**（#5 疑点）：`DEEPINFER_USE_NEXTN_DECODE` 改成非 5 时，位置索引、`sample_positions_table`、`confidence_rates` 形状都会跟着变，但权重是按 block=5 训练的。这是配置陷阱，值得在部署章提醒。
3. **模型卡里的 Tech Report 链接 404**：`README.md` 指向 `DeepSeek_V41_Tech_Report.pdf`，但 HF 仓库文件列表里没有该文件（2026-10-23 核对）。所以"官方总参数/激活参数"没有可引的一手 PDF，只能用权重元数据推导（本笔记做法）。
4. **Engram 表的参数归属**：196.9B 查表占总量 25.8%，但它们是 token 级 sparse gather；如果书里沿用"总参数 763B / 激活 16B"的表述，读者会误以为 763B 都要过矩阵乘。建议书中单列一行"查表参数"。
5. **草稿块的"块内非因果"是否是训练同构**：引擎侧确定（#6），但训练是否也用同一 noise-token 布局无法从仓库确认；若不同构，Markov 头就是补偿项而非等价项（E）。
6. **MTP 层也带 `gate.bias_vl`**（#20 疑点）：草稿不走多模态路径，这份参数恒不生效，属于"多加载 128 个 float"的噪音，可忽略但会在权重表里造成疑问。

## 5. 可用于图表的材料

**表 A：参数分项（饼图/堆叠条形）** — 见 #1 表格（543.58 / 196.93 / 14.23 / 5.13 / 1.42 / 1.32 / 0.41 / 0.19 B）。

**表 B：单层激活参数构成（每 token，单位 M）**

| 项 | 参数 | 占比 |
|---|---|---|
| 注意力 MLA（wq_a/wq_b/wkv/wo_a/wo_b + norms） | 126.62 | 33.6% |
| 6 个路由专家（6×35.39） | 212.34 | 56.3% |
| 1 个共享专家 | 35.39 | 9.4% |
| MoE gate | 1.97 | 0.5% |
| HC（`hc_*_fn [24,20480]`） | 0.98 | 0.3% |
| 合计/层 | 377.30 | 100% |

**表 C：DFlash decode 时序（流程图素材）**

| # | 动作 | 代码位置 |
|---|---|---|
| 1 | 主模型 40 层前向，第 37/38/39 层把 attention 输入残差写入 `dflash_input` | `decoder_layer.py:144-145` |
| 2 | 主模型 forward 末尾把 `dflash_input` 投成 512 维主 K，写 3 个草稿层 SWA 缓存 | `dflash.py:219-228` |
| 3 | Sampler 采样/验证，得 golden token 与 `last_valid_offset`（接受数） | `sampler.py:411-470` |
| 4 | 构造 5 位置草稿块索引（窗口 128 + 块内 5） | `get_kv_index.py:169-197` |
| 5 | 3 个草稿层对 5 位置块前向（同构 MLA + 128 专家/topk3） | `dflash.py:241-269` |
| 6 | `post_norm` → `lm_head` 得 5×logits | `dflash.py:288-289` |
| 7 | 5 步 Markov 修正 + 采样 + 置信度头并行输出 | `dflash.py:308-334` |
| 8 | 草稿写入 `mtp_info_pool`，置信度进 CPU 调度 | `dflash.py:339-345`、`sequence.py:82` |
| 9 | 下一轮按置信度累乘的全局 top-k 分配 `seqlens_q` | `schedule_mtp_varlen.py:11-33` |

**表 D：ViT 单块参数（12.85M）**

| 子层 | 形状 | 参数 |
|---|---|---|
| attn.wqkv | [3072, 1024] + bias 3072 | 3.15 M |
| attn.wo | [1024, 1024] + bias | 1.05 M |
| mlp.w1（gate+up 融合） | [5632, 1024] | 5.77 M |
| mlp.w2 | [1024, 2816] | 2.88 M |
| 2×RMSNorm | [1024]×2 | 0.002 M |

**图 E：图像 token 几何**：`n_vit = (⌈H/14⌉, ⌈W/14⌉)` → `n_llm = (⌈n_vit_h/3⌉, ⌈n_vit_w/3⌉)` → 文本段 = `n_llm_h × (n_llm_w + 1) + 2`，上限 1024（约束式 `h(w+1) = max_n_token − 2`，`image_processor.py:103-104`）。

**可复现脚本（参数分项求和，E 级推导过程）**：对 48 个 shard 用 HTTP Range 读 safetensors 头部 JSON，按名字归类；`I8` 且名含 `.ffn.experts.` 的按 2×numel 计参数，`.scale`（`F8_E8M0`）不计参数；求和得 763.21B，与 HF `safetensors.total` 一致。
