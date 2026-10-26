# R06 分页 KV Cache、StateBlock 槽位布局、128 对齐与显存规划

## 0. 元信息

- 使用的代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（本地 `research/sources/DeepInfer`，`git log -1` 为 2026-10-21）
- 查阅的主要来源（一手，A 级）：
  - `deepinfer/worker/cache_engine.py`（`KVSTORE_GRAIN`、`KVCache`、`StateBlock`、`KVCacheManager`、`GPUCacheInterface`、`CacheEngine.allocate_gpu_cache`）
  - `deepinfer/model_executor/layers/tilelang_ops/cache_common.py`（`KVCacheFormat`、`V41_FORMAT`、`V41_FP4_FORMAT`、`FP4_INDEX_KV_FORMAT`、`alloc_kv_cache`）
  - `deepinfer/model_executor/layers/tilelang_ops/{store_kv_cache.py,load_paged_kv_cache.py,state_cache.py,cuda/state_cache.py,get_kv_index.py,swa_index.py}`
  - `deepinfer/model_executor/layers/tilelang_ops/{prepare_v4_decode_metadata.py,asc/kv_quant.py}`
  - `csrc/engine/block_manager.hpp`、`csrc/engine/sequence.hpp`、`csrc/engine/scheduler.hpp`
  - `csrc/engine/input_metadata/{prepare_tensor_input_base.cpp,prepare_tensor_input_v4.cpp}`
  - `deepinfer/worker/{base.py,model_runner.py}`、`deepinfer/engine/llm_engine.py`、`deepinfer/config.py`、`deepinfer/engine/arg_utils.py`
  - `deepinfer/model_executor/model_metadata_utils.py`、`deepinfer/model_executor/input_metadata/input_metadata.py`、`deepinfer/model_executor/layer_utils.py`
  - `tests/worker/test_state_cache_ops.py`
  - `doc/inference_deployment_qa.md`（"KV cache storage" 一节，官方文档，B 级）
- 说明：本笔记区分"代码字面语义"（A）与"官方 QA 文档描述"（B）。凡文档写了但仓库里没有调用者的路径，一律标注，不当成"代码里有"。没有运行任何东西；仓库也没有可用的 V4.1 checkpoint（`examples/_config.py:38` 指向 HF `deepseek-ai/DeepSeek-V4.1-Flash-1017`，本地无缓存）。引用块（`>`）是原文摘录；两行的小函数用 ` / ` 连接或在括号里注明所属函数，不代表源码是单行。
- 与 R07 的分工：SWA 有界重放 / YOCO 截断 / `KEEP_TOKENS` 归 R07；本笔记只写"落盘与槽位"这一面，环尺寸推导直接引用 R07 #2/#3。
- 引用缩写约定：正文为省篇幅，同一事实条目内后续引用只写文件名。所有裸文件名对应的完整路径如下（行号均指该 commit 的本地文件）：
  - `cache_engine.py` → `deepinfer/worker/cache_engine.py`
  - `base.py` / `worker/base.py` → `deepinfer/worker/base.py`
  - `model_runner.py` → `deepinfer/worker/model_runner.py`
  - `llm_engine.py` → `deepinfer/engine/llm_engine.py`；`arg_utils.py` → `deepinfer/engine/arg_utils.py`；`engine_type.py` → `deepinfer/engine/engine_type.py`
  - `environ.py` → `deepinfer/environ.py`；`config.py` → `deepinfer/config.py`
  - `cache_common.py` / `store_kv_cache.py` / `get_kv_index.py` / `state_cache.py` / `prepare_v4_decode_metadata.py` / `swa_index.py` → `deepinfer/model_executor/layers/tilelang_ops/<同名>`
  - `cuda/state_cache.py` → `deepinfer/model_executor/layers/tilelang_ops/cuda/state_cache.py`；`asc/compress.py` → `.../asc/compress.py`；`asc/kv_quant.py` → `.../asc/kv_quant.py`
  - `compress.py`（仅 #14 出现）→ `deepinfer/model_executor/models/v4_modules/compress.py`；`attn.py` / `v4_modules/attn.py` → `deepinfer/model_executor/models/v4_modules/attn.py`
  - `input_metadata.py` → `deepinfer/model_executor/input_metadata/input_metadata.py`；`prepare_tensor_input_zos.py` → `deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py`
  - `model_metadata_utils.py` → `deepinfer/model_executor/model_metadata_utils.py`；`layer_utils.py` → `deepinfer/model_executor/layer_utils.py`
  - `block_manager.hpp` / `sequence.hpp` / `scheduler.hpp` / `pybind_sequence.cpp` / `prepare_tensor_input_base.cpp` / `prepare_tensor_input_v4.cpp` → `csrc/engine/...`（`prepare_tensor_input_*` 在 `csrc/engine/input_metadata/` 下）
  - `test_state_cache_ops.py` → `tests/worker/test_state_cache_ops.py`；`benchmark.py` → `examples/benchmark.py`；`_config.py` → `examples/_config.py`

## 1. 三个最重要的发现

1. **代码里有三个不同的"128"，书里必须分开写**：① `block_size = 128`（每个 KV block 表示 128 个压缩前 token，`model_metadata_utils.py:134-135`）；② `KVSTORE_GRAIN = 128`（state 只在 token 数为 128 倍数处落盘，`cache_engine.py:32-33`）；③ 128 **字节**（KV 层偏移与 `block_stride` 的对齐要求，`model_metadata_utils.py:138-140`、`cache_engine.py:481-484`），外加 `kv_ca_index` 列数 pad 到 128 的整数倍（`input_metadata.py:362`）。混用会让"128 对齐"这句话失去意义。
2. **一个 state 槽（`StateBlock`）是"两个区 + 两个环"**：`[stored region: SWA][dropped region: YOCO consumer SWA | compression state]`（`cache_engine.py:229-235`）。落盘只取 stored 区，`state_size_to_disk = 20 × 134 × 528 = 1,415,040 B/seq`（`cache_engine.py:306` + `doc/inference_deployment_qa.md:169`）。环形缓冲的 134 = 128（窗口）+ 6（`MAX_BACKTRACKING_TOKENS`），而 **kvstore 侧环是 262 = 134 + 128**（`PREFILL_STORE_EXTRA_SIZE`）；两者不是同一个数，默认配置下（`EXTRA_STATE=False`）才相等。
3. **开源版没有任何自动前缀缓存**：`BlockManager` 只是"FIFO 空闲链表 + ref_count"（`block_manager.hpp:27-117`），没有 hash 链、没有 radix tree、没有 LRU。仓库里唯一的前缀共享原语是显式的 `Scheduler.fork_seq`（ref_count pin 住整张 block table + clone state，`scheduler.hpp:1082-1121`）；落盘/命中相关的一切（`copy_state`、`select_state_slice_to_store`、`load_state_from_store_to_prefill_cache`、`do_save_kvcache`）在开源树里要么是 `None`，要么只有测试调用。

## 2. 事实条目

### #1 三个"128"：block 大小、落盘粒度、字节对齐
- 证据：
  - `deepinfer/model_executor/model_metadata_utils.py:134-135`
    > `def resolved_tokens_per_block(self) -> int:` / `return 128`
  - `deepinfer/worker/cache_engine.py:32-33`
    > `KVSTORE_GRAIN = 128` / `"""kvstore 落盘粒度：只在 token 数为 128 的倍数处落盘"""`
  - `deepinfer/model_executor/model_metadata_utils.py:138-140`
    > `"""MXFP8 / sparse FP4 cache requires 128-byte block alignment."""` / `return 128`
- 等级：A
- 意味着什么：`block_size`（token 计数）与 128 B（字节对齐）无关；文档里"128 token 对齐"指第 ②③ 类，不是 block 大小。写书时第一次出现就要点名是哪一个。
- 疑点：`block_size` 与 `KVSTORE_GRAIN` 都取 128 只是巧合还是有意（让"一个 block = 一个落盘粒度"成立）？代码没有断言二者相等。

### #2 block 大小硬编码 128 token，与压缩率无关；压缩层每 block 只存 128/ratio 个 token
- 证据：`deepinfer/worker/cache_engine.py:452-453`
  > `token_per_block = block_size // ratio` / `delta = token_per_block * self._bytes_per_token`
  `deepinfer/worker/cache_engine.py:433-434`（index 段同样按 `block_size // ratio` 切）
  > `token_per_block = block_size // ratio if ratio > 0 else block_size` / `delta = token_per_block * self._qk_topk_bytes`
  以及 `deepinfer/engine/arg_utils.py:84-86, 93-97`（`resolved_tokens_per_block` 先于 `CacheConfig` 求出，与用户传入的 `tokens_per_block` 只做相等性断言，然后作为 `CacheConfig.block_size` 传入）
- 等级：A
- 意味着什么：C1 层一个 block 存 128 个 KV token，C2 层同一个 block 只存 64 个（压缩后）。"一个 block 装多少 token"按层不同，只有"压缩前 128 token"对所有层一致。
- 疑点：无。

### #3 一个 block 是一整段 uint8 字节，按 runtime 层序切段，每段 32 B 对齐，整块再按 `lcm(128, bytes_per_token)` 对齐
- 证据：`deepinfer/worker/cache_engine.py:479-484`
  > `# DeepGEMM index requires block_stride to be a multiple of 128 bytes;`
  > `# main attention requires block_stride to be a multiple of _bytes_per_token.`
  > `block_alignment = math.lcm(self._align_to, self._bytes_per_token)`
  > `self.kv_total_in_bytes = _align_to_multiple(self._kv_offset, 1, block_alignment)`
  每层段内 pad：`deepinfer/worker/cache_engine.py:437-439, 459-461`（`padded_delta = _align_to_multiple(delta, 1, 32)`）
- 等级：A
- 意味着什么：block 不是"每层一个 tensor"，而是**一条扁平的字节带**：`[层0 indexK | 层0 mainKV | 层1 ...]`，各层用 `(layer_id, offset, length)` 三元组寻址（`cache_engine.py:604` 的 `layout`）。算出来的对齐值：`lcm(128,288)=1152`、`lcm(128,528)=4224`。
- 疑点：`layout` 里的 `length` **已包含层内 32 B pad**（`cache_engine.py:931` 注释），所以 `padding_bytes = block_shape[0] - Σlength` 只是**尾部对齐余量**（补齐到 `lcm(128, bytes_per_token)` 的部分）。日志里把它叫 `padding_bytes` 并给出百分比（`:932-934`），会让读者低估实际浪费——层内 pad 没算进去。

### #4 每 token 字节数：main KV 288(FP4)/528(MXFP8)，index K 68(FP4)/132(MXFP8)，SWA 永远 528
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cache_common.py:109-155`
  > `V41_FORMAT = KVCacheFormat(quantized_dtype="e4m3", quantized_elements=512, elements_per_sf=32, ..., sf_format="ue8m0", token_interleaved=True)`
  > `V41_FP4_FORMAT = ... "e2m1", 512, elements_per_sf=16, sf_format="e4m3"`
  > `FP4_INDEX_KV_FORMAT = ... "e2m1", 128, 32, sf_format="ue8m0"`（无 `token_interleaved`）
  `bytes_per_token = quantized_bytes + sf_bytes`（`cache_common.py:99-101`）
- 等级：A
- 意味着什么：`512/2+512/16=288`、`512+512/32=528`、`128/2+128/32=68`、`128+128/32=132`。这四个数就是全书所有 KV 容量估算的原子。`use_fp4_sparse_kv` 默认 `True`（`deepinfer/hugging_face/config.py:303`），所以开源默认模型的主 KV 是 288 B/token 的 FP4。
- 疑点：**分配侧把 index K 写死成 FP4**：`_get_qk_topk_bytes` 直接算 `index_head_dim//2 + index_head_dim//32`（`cache_engine.py:76-78`），完全不看 `use_fp4_sparse_kv`；而 kernel 侧同时支持 132 B 的 `MXFP8_INDEX_KV_FORMAT` / `FP8_FLOAT32_SF_INDEX_KV_FORMAT`（`cache_common.py:138-155`）。若某个 checkpoint 的 indexer 用 MXFP8，`KVCacheManager` 会按 68 B 切段 → 欠分配。仓库里没有断言把两者钉在一起。

### #5 层按三种 attention 形态分类，决定它占哪几类缓存
- 证据：`deepinfer/worker/cache_engine.py:683-699`（三行 return 原文）
  > `return cfg.cfg.get("sink_mode", "") == "swa"`（`_has_SWA`；`cfg is None` 时先返回 `False`）
  > `return cfg.cfg.get("index_topk", 0) > 0`（`_is_sparse`）
  > `return cfg.cfg.get("kv_compress_ratio", 0) == 0 and cfg.cfg.get("sink_mode", "") != "swa"`（`_is_full`）
- 等级：A
- 意味着什么：一个层最多同时拥有 `swa_cache` + `ica_cache` + `ca_cache`（+ 两个 state）。`_is_full` 的层直接 `AssertionError`（`cache_engine.py:466-468`），即 V4.1 没有"全 attention"层，`KVCacheManager` 不支持。
- 疑点：`_has_SWA` 的注释说"`use_swa=False` 的层（packed-KV/YoCO consumer）也占一个从不读写的 SWA slot"（`cache_engine.py:684`），即有些槽位是**纯占位**，会白白消耗 state 内存。

### #6 block 表在 C++ 侧是每序列一个 `vector<int>`，按 token 顺序映射逻辑块→物理块
- 证据：`csrc/engine/sequence.hpp:200-208`
  > `struct KvCacheState { int block_size{0}; ... int hit_tokens{0}; int swa_valid_start{0}; std::vector<int> block_table; int state_id{-1}; int last_copy_state_token_idx{0}; }`
- 等级：A
- 意味着什么：`block_table[i]` = 第 i 个逻辑块（覆盖 token `[i*128, (i+1)*128)`）的物理块号。`state_id` 与 `block_table` 并列——**KV 分页和 state 槽位是两套独立资源**，各自一套 `BlockManager`（见 #11）。
- 疑点：无。

### #7 kernel 侧不看 `vector<int>`，看每 token 一份的稠密副本 `[total_q, max_num_blocks]`
- 证据：`csrc/engine/input_metadata/prepare_tensor_input_base.cpp:207-229`
  > `// 每个 query token 复制一份所属序列的 block table，右侧留未初始化的 pad`
  > `auto block_tables = get_or_create_tensor(result, has_out, "block_tables", {total_q, max_num_blocks}, torch::kInt32);`
  > `std::copy(p_seq->block_table.begin(), p_seq->block_table.begin() + num_blocks, p_block_tables + i_q * max_num_blocks);`
  另一条路径（ZOS decode）在 GPU 上展开：`prepare_v4_decode_metadata.py:73, 96`（`copy_block_tables_tl` + `copy_kernel(...)`）
  行宽上界：`deepinfer/worker/model_runner.py:114-115`
  > `# 这里的 64 是对齐要求` / `self.max_num_blocks = ((runtime_max_model_len + block_size - 1) // block_size + 63) // 64 * 64`
- 等级：A
- 意味着什么：这是 decode 每步都在的"块表副本区"：`total_q × max_num_blocks × 4 B`。128k 上下文、block=128 → 1024 块；`max_num_blocks` 再向上取 64 的倍数。
- 疑点：`num_blocks` 用 `seqlen_k / compress_ratio` 折算（`prepare_tensor_input_base.cpp:219-222`），所以同一个 `block_tables` 张量对不同 ratio 的层复用不同前缀长度，读越界只能靠 kernel 侧 `klen` 约束。

### #8 槽位公式：`slot = block_id * tokens_per_block + offset`，`-1` 表示不写
- 证据：`deepinfer/model_executor/layers/tilelang_ops/store_kv_cache.py:48` 与 `:106-107`
  > `slot_mapping: [n], int32; -1 denotes padding and is not stored.`
  > `safe_slots = slots.clamp_min(0).long()` / `block, offset = safe_slots // block_size, safe_slots % block_size`
  decode 读侧同一公式：`get_kv_index.py:51-56`
  > `block_tables.flatten().index_select(0, (pos // token_per_block)...) * token_per_block + pos % token_per_block`
- 等级：A
- 意味着什么：整个 paged KV 只有"一维 slot 编号"这一种寻址，block 号与块内偏移在 kernel 里现算。prefill 写新 KV、decode 写 1 个 token，走的是同一个 `store_kv_cache`。
- 疑点：无。

### #9 物理块 0 永远不用；空闲块是 FIFO 双链表 + `ref_count`
- 证据：`csrc/engine/block_manager.hpp:27-35`
  > `struct Blocks { // NOTE: gpu blocks 0 是不用的，所以这里 0 代表空`
  > `Blocks(int _n) : _blocks(_n), _head(1), _tail(_n - 1), _size(_n - 1) { TORCH_CHECK(_n > 1, "GPU blocks 太少了: ", _n); ...`
  分配/释放：`block_manager.hpp:82-103`（`allocate()` 从 head 弹出并 `ref_count += 1`；`free()` 减到 0 时 `_push_back` 到 tail）
- 等级：A
- 意味着什么：① 实际可用块是 `num_gpu_blocks - 1`；② 复用的是"最早被释放"的块，不是 LRU——因为不存在"再次使用"的概念，没有前缀命中；③ `ref_count` 只被 fork 用（#24）。
- 疑点：`_blocks[_n-1].next = 0` 写在 for 循环之后（`block_manager.hpp:30-34`），依赖循环只到 `_n-2` 才正确；读起来容易误判，但逻辑是对的。

### #10 分配只发生在三个时刻：admission 批量、decode 每步 ±1、free 整表
- 证据：`csrc/engine/scheduler.hpp:426`（admission：`auto new_blocks = block_manager->allocate_nolock(to_allocate_blocks);`）、`:816-828`（`_append_slot`）
  > `int num_blocks = seq.get_num_blocks();`
  > `if (... block_table.size() < num_blocks) block_table.emplace_back(block_manager->allocate_single_nolock());`
  > `else if (... > num_blocks) { ... free_blocks_nolock(redundant_blocks); block_table.erase(...); }`
  块数定义：`csrc/engine/sequence.hpp:291-293`
  > `return (get_len() + schedule.total_inflight_tokens + get_nextn_len() + cache.block_size - 1) / cache.block_size;`
- 等级：A
- 意味着什么：MTP 的 draft token 会**提前把块算进去**（`get_nextn_len()`），draft 被拒后 `_append_slot` 再把多出来的块 free 掉——块表随投机结果伸缩。
- 疑点：`total_inflight_tokens` 是 ZOS 异步调度在飞的 token，说明块数按"最坏情况"预留；这与 ZOS 的语义耦合，见 R05。

### #11 两套 `BlockManager`：KV 块一套，state 槽一套
- 证据：`deepinfer/engine/llm_engine.py:265-266`
  > `self.block_manager = BlockManager(cache_config.num_gpu_blocks)`
  > `self.state_manager = BlockManager(cache_config.num_state_seqs) if cache_config.bytes_of_state_per_seqs > 0 else None`
  联合准入：`csrc/engine/scheduler.hpp:194-201`
  > `/// block + state 的联合分配检查：state 开不出来时，视 block manager 也为 LATER。`
- 等级：A
- 意味着什么：`BlockManager` 被复用成通用"整数槽分配器"（`Blocks` 只认 id），state 槽与 KV 块共用同一份 FIFO 逻辑，但**容量与释放时机独立**；任何一边不够，请求都进不来。
- 疑点：类名叫 `BlockManager` 而实际管两种资源，`pin_block` / `free_blocks` 在 state 上用单元素列表传参（`scheduler.hpp:741`），读起来容易误解。

### #12 `StateBlock` 槽位布局：stored（落盘）在前，dropped（不落盘）在后
- 证据：`deepinfer/worker/cache_engine.py:221-241`
  > `具体布局（从低地址到高地址）：`
  > `    [stored region: SWA]`
  > `    [dropped region: YOCO consumer SWA | compression state]`
  > `state_size_to_disk 仅包含需落盘 SWA 的 inference 窗口。`
  区域切分：`cache_engine.py:348-362`
  > `self._swa_blocks = state_tensor[:, :swa_total]` / `self._state_res = state_tensor[:, swa_total:]`
- 等级：A
- 意味着什么：落盘 = 取槽的前 `state_size_to_disk` 字节（前缀截断即可排除整个 dropped 区），PE→CE 传输 = 取前 `total_bytes_inference` 字节（含 dropped）。**同一块 buffer 靠长度区分两种用途**，这是整个设计的支点。
- 疑点：文档 `doc/inference_deployment_qa.md:163` 用散文写清了 dropped 区内容（consumer SWA + `ca_state`/`ica_state`），但没给区内**顺序**；代码里的顺序是 `[consumer SWA][逐层 ica_state, ca_state]`（#13），两者不矛盾但也没有互相校验。

### #13 dropped 区内部的分配顺序：先整块 consumer SWA，再按层交错 `ica_state` → `ca_state`
- 证据：`deepinfer/worker/cache_engine.py:352-354`
  > `# _state_res contains dropped SWA followed by compression state.`
  > `self._dropped_state_offset = self._swa_drop_offset + self._state_block.swa_drop_total_bytes_kvstore_pad`
  逐层分配：`cache_engine.py:441-443`（先 index state）与 `:463-465`（后 main state），都调 `_alloc_state`（`:367-374`，纯指针推进）
- 等级：A
- 意味着什么：`ica_state` 与 `ca_state` 是**按层交替**的，不是"所有 index state 一段、所有 main state 一段"。`finalize()` 用断言校验这条累加路径与 `StateBlock` 的预计算不漂移（`cache_engine.py:485-492`）。
- 疑点：`_alloc_state` 用 `view(-1, num_state_token, 2, head_dim*4).view(torch.float32)` 把 uint8 槽切成 fp32（`cache_engine.py:374`），要求 `head_dim*4` 与 stride 严格匹配；`_state_res` 是切片后的非连续视图，这里能成立只因为 `state_tensor` 的 stride(0) 被 `total_bytes_*` 对齐保证。

### #14 compression state 的形状与语义：`[slots, ring, 2, dim]` fp32，`2` = (score, latent)
- 证据：`deepinfer/worker/cache_engine.py:205-215`
  > `"""index attn state: [num_state_seqs, ring_size, 2, index_dim], fp32."""`
  > `"""compressed attn state: [num_state_seqs, ring_size, 2, dim], fp32."""`
  两个通道的写入：`deepinfer/model_executor/layers/tilelang_ops/asc/compress.py:195-198`
  > `T.copy(score_cache_ub, state_cache[state_idx, cache_idx, 0, 0])` / `T.copy(latent_cache_ub, state_cache[state_idx, cache_idx, 1, 0])`
  来源：`deepinfer/model_executor/models/v4_modules/compress.py:95-96`
  > `assert projected_kv.dtype == torch.float32 and state_cache is not None` / `score, latent = projected_kv.chunk(2, dim=-1)`
- 等级：A
- 意味着什么：C2 层把投影输出拆成 score/latent 两半，逐 token 累加到环里，攒够 2 个 token 才吐出一个压缩 KV。C1 层没有 state（`compress.py:89-93` 直接断言 `state_cache is None`）。
- 疑点：`_state_bytes(head_dim, n) = n * 2 * head_dim * 4`（`cache_engine.py:41-43`）里 `2` 是硬编码的通道数，和 `view(-1, n, 2, head_dim*4)` 两处独立写死，没有共享常量。

### #15 压缩环大小 = `ratio + (B-1)`，即 C1→6、C2→7
- 证据：`deepinfer/worker/cache_engine.py:130-133`
  > `@lru_cache(8)`
  > `return compress_ratio + max(0, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS - 1)`
- 等级：A
- 意味着什么：环要多存 5 个"半压缩"边界，供 MTP 回退时重算。注意 SWA 环加的是 `B`（6），压缩环加的是 `B-1`（5）——同一个 `B` 的两种用法。
- 疑点：为什么是 `B-1` 而不是 `B`，代码与文档都没有解释（E：回退 n 个 token 只需重放 n-1 个压缩边界，因为边界本身由 token 计数决定）。

### #16 `134 = 128 + 6` 与 `262 = 134 + 128` 是两个不同的环
- 证据：`deepinfer/worker/cache_engine.py:88-110`
  > `return (window_size + envs.DEEPINFER_MAX_BACKTRACKING_TOKENS) if window_size > 0 else 0`
  > `return get_inference_swa_block_size(window_size) + ((envs.DEEPINFER_PREFILL_STORE_EXTRA_SIZE + get_yoco_store_extra_tokens()) if envs.DEEPINFER_PREFILL_STORE_EXTRA_STATE else 0)`
  默认值：`deepinfer/environ.py:128-132, 163-164`（`USE_NEXTN_DECODE=5`、`MAX_BACKTRACKING_TOKENS=max(v, 5+1)`、`PREFILL_STORE_EXTRA_STATE=False`、`PREFILL_STORE_EXTRA_SIZE=128`）
- 等级：A
- 意味着什么：默认（`EXTRA_STATE=False`）槽内环就是 134，落盘时无需重排；打开 `EXTRA_STATE` 后槽内环 262，落盘前要把 262 的环"卷"成 134 的连续窗口（#19）。**书里写"134 = 128 + 6"时必须补一句"kvstore 视角还有 +128 的余量"**。仓库里显式打开它的地方只有两处：`examples/benchmark.py:156-157`（YoCo 截断模式非 off 时无条件开）与 `examples/yoco/yoco_equiv.py:474`（compaction 实现强制要求）；没有生产部署脚本——"生产 PD 会开"属于 E 级推断。
- 疑点：`PREFILL_STORE_EXTRA_SIZE=128` 与 block 大小同为 128，按 #36 的推导它正好是"落盘边界最多落后一整段"的余量（134+127 → 262）；`cache_engine.py:104` 只说"多存一段才能保证多轮命中"，没把 127 写出来——这一步是 E 级。

### #17 落盘只在 128 的倍数处，且位置语义是"以该位置为结尾（不含）的窗口"
- 证据：`deepinfer/worker/cache_engine.py:771-774, 785-787`
  > `假设当前算到了 pos=128+127，则需要落盘 [-x-127, -127] 这一段的值，x 是 SWA 窗口大小`
  > `NOTE state_tensor 也是一个滚动数组`
  > `if not TO_CE: assert last_hit_position % KVSTORE_GRAIN == 0, f"{last_hit_position=} 必须是 {KVSTORE_GRAIN=} 的倍数"`
  测试侧同款：`tests/worker/test_state_cache_ops.py:66-68`
  > `"""kvstore hit positions: the processed position is rounded down to KVSTORE_GRAIN."""` / `return last_process_position // KVSTORE_GRAIN * KVSTORE_GRAIN`
- 等级：A
- 意味着什么：`last_hit_position` 是**开区间右端**，窗口是 `[pos-134, pos)`。因为窗口比 128 大 6，任意 128 边界处的窗口都能完整取出，不依赖当前 decode 位置。
- 疑点：`TO_CE=True` 时不做 128 断言（注释写"传给 CE 时发送完整 state，无对齐要求"），意味着 PE→CE 的 state 传输可以发生在任意位置；CE 侧如何解释这个非对齐窗口，开源树里没有代码。

### #18 ring→窗口的线性化由 tilelang kernel 做，不是整段 copy
- 证据：`deepinfer/model_executor/layers/tilelang_ops/cuda/state_cache.py:55-67`
  > `start_pos = last_hit_position - block_size_inference`
  > `source_token_idx = (start_pos + actual_token_idx + block_size_kvstore) % block_size_kvstore`
  > `target_token_idx = (start_pos + actual_token_idx + block_size_inference) % block_size_inference`
  > `state_tensor[actual_layer, target_token_idx, feat_idx] = state_cache[actual_layer, source_token_idx, feat_idx]`
  回写：`cuda/state_cache.py:132-134`（`source_token = (last_hit_position + token) % block_size_inference`，`target_token = (last_hit_position + token - block_size_inference) % block_size_kvstore`）
- 等级：A
- 意味着什么：`state_tensor`（落盘/staging 目标）本身也是一个长度 134 的滚动数组，写入位置用**绝对位置对 134 取模**，所以连续多次落盘会覆盖同一块内存——落盘方必须在下次覆盖前把数据取走（这就是 `copy_state` 存在的原因，#22）。
- 疑点：参考实现 `state_cache.py:44-45` 写成 `state_tensor[:, index % block_size_inference]`（`index` 是绝对位置），与 kernel 的 `(start_pos+i+W)%W` 数学上等价，但可读性差，容易让人以为在写"绝对位置槽"。

### #19 无 `EXTRA_STATE` 时直接整段 `copy_`，有则逐层重排
- 证据：`deepinfer/worker/cache_engine.py:788-797`
  > `# 无 prefill-extra 时 state 不需要重排，直接整段读出 / copy`
  > `state_size = cls.total_bytes_inference if TO_CE else cls.state_size_to_disk`
  > `src = state_cache[state_id, :state_size]` … `state_tensor.copy_(src)`
- 等级：A
- 意味着什么：默认配置下"落盘"就是一次连续 D2H 的源切片，零 kernel；只有打开 EXTRA_STATE 才付重排代价（`select_state_slice` 逐层、逐 token 的 gather）。
- 疑点：`total_bytes_inference`（PE→CE 用）与 `state_size_to_disk`（落盘用）共用同一个函数分支，`TO_CE` 只是一个 bool；参数语义靠调用方自律。

### #20 state 槽字节数按 `lcm(32, 528) = 1056` 对齐（有 SWA 时），否则按 128
- 证据：`deepinfer/worker/cache_engine.py:301-305`
  > `if self.swa_num_layer:`
  > `    # swa_cache.stride(0) inherits the entire state slot, including compression state.`
  > `    align_to = math.lcm(32, swa_size_per_token)`
  > `self.total_bytes_kvstore = _align_to_multiple(self.swa_total_bytes_kvstore_pad + compress_bytes, 1, align_to)`
  校验：`tests/worker/test_state_cache_ops.py:466-468`（`state_align_to = math.lcm(32, 528)` 并传给 `_assert_layout_self_consistent`）
- 等级：A
- 意味着什么：`swa_cache` 用 `as_strided` 让 stride(0) 等于整个槽长（`cache_engine.py:420-422`），所以槽长必须是 528 与 32 的公倍数，否则单层视图会跨槽。`1056 = 2 × 528`。
- 疑点：`compress_bytes` 是 fp32（4 的倍数），`align_to=1056` 对它没有整除保证，靠的是"整块对齐"而不是"每段对齐"；`finalize()` 的断言能发现漂移但发现不了浪费。

### #21 `state_size_to_disk` 只含 stored 区：20 × 134 × 528 = 1,415,040 B/seq
- 证据：`deepinfer/worker/cache_engine.py:306`
  > `self.state_size_to_disk = self.swa_store_total_bytes_inference_pad`
  `doc/inference_deployment_qa.md:169`
  > `Only state_size_to_disk = 20 layers × 134 tokens × 528 B = 1,415,040 B per sequence goes into the store.`
- 等级：A（代码）+ B（文档数值）；两者一致（20×134×528=1,415,040，已核）
- 意味着什么：落盘量与上下文长度 L 无关，是常量——这是"SWA 状态可以按 slot 整段搬运"的前提。134×528=70,752 B 本身已是 32 的倍数，所以每层无额外 pad。
- 疑点：`swa_store_num_layer` 由 `n_prefill_layer` 决定（`cache_engine.py:113-116, 266-270`），文档说"V4.1 是前 20 层"（`doc:163`），但仓库里没有 V4.1 的 config，**20 这个数字无法在代码侧复核**。

### #22 decode 期间的 state 副本区 `copy_state`：基类只有钩子，没有分配
- 证据：`deepinfer/engine/llm_engine.py:284-285`
  > `self.copy_state: torch.Tensor | None = None` / `self.copy_state_event: torch.cuda.Event | None = None`
  `:787-798`
  > `"""把给定 seq 当前的 GPU state 拷贝到 copy_state buffer，并更新 last_copy_state_token_idx。该方法仅下发任务到 GPU，不等待完成，消费方需自行同步。"""`
  > `self.copy_state[state_id].copy_(self.driver_worker.cache_engine.gpu_states[state_id])` / `seq.last_copy_state_token_idx = seq.num_cached_tokens`
  `:811-813`（`def should_copy_state(self, seq) -> bool: """虚函数…""" return False`）、`:930-937`（decode 每步先 `copy_state_event.synchronize()` 再 `do_save_kvcache`）
  语义（B 级）：`doc/inference_deployment_qa.md:165`
  > `During the decode stage, a separate replica region is allocated in device memory; every time the request crosses a 128-token boundary, its current window is copied into the replica region on-device. ... The copy position can overshoot the boundary by at most 5 tokens, which is absorbed by the 6 extra positions the slot's ring buffer has beyond the window.`
- 等级：A（钩子与拷贝语句）+ B（"每跨 128 边界拷一次、结束时取最近一次完整副本"的完整语义）
- 意味着什么：副本区不是"每 token 一份"，而是**整个 state 槽的第二份**（`copy_state[state_id]` 与 `gpu_states[state_id]` 同形），由 decode 引擎在跨 128 边界时整槽覆盖，从而把"环被继续写"与"落盘需要稳定快照"解耦。显存侧的对价写在 `worker/base.py:243`：`state_size *= 2  # state store & copy buffer, 所以要开两倍大小`。
- 疑点：① 开源树里 `copy_state` **从未被分配**（只有 `= None` 与判空），`should_copy_state` 恒 `False`；真正分配它的子类不在仓库里。② 文档说"最多越过边界 5 个 token"，而环多出的是 6 个位置，多出的 1 个是给 `nextn+1` 的最后一个 accept 位——这是 E 级推断。

### #23 decode 期间的块表副本区：`[total_q, max_num_blocks]` int32，每步重填
- 证据：`prepare_tensor_input_base.cpp:207-229`（见 #7）；ZOS 路径的 buffer 复用：`deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py:177`
  > `block_tables=get_or_create(args.out, "block_tables", [total_q, args.max_num_blocks], torch.int32)`
- 等级：A
- 意味着什么：这是**第二块"副本区"**，容易被和 #22 混为一谈：块表副本位于 input metadata（每步重建，`total_q×max_num_blocks×4 B`），state 副本位于引擎状态（每跨 128 边界重建，`num_state_slots×slot_bytes`）。
- 疑点：`get_or_create` 带 `args.out` 时是持久 workspace（ZOS），不带时每步新分配；两条路径的显存账在 profiling 里如何计，未见说明。

### #24 前缀共享只有显式 `fork_seq`：ref_count pin 整表 + COW 最后一块 + clone state
- 证据：`csrc/engine/scheduler.hpp:1082-1121`
  > `init.cache.hit_tokens = seq->cache.hit_tokens;`
  > `block_manager->pin_blocks(fork_seq->cache.block_table);`
  > `fork_seq->cache.state_id = seq->cache.state_id; state_manager->pin_block(seq->cache.state_id);`
  写入时的 COW：`scheduler.hpp:602-617`
  > `_clone_state(*seq); int new_block = block_manager->allocate_single_nolock(); ... clone_blocks.emplace_back(seq->cache.block_table[num_blocks - 1]); clone_blocks.emplace_back(new_block);`
  使用者：`examples/benchmark.py:331-353`（`fork_request_batch` 把一个已 prefill 的前缀 fork 成一批）
- 等级：A
- 意味着什么：开源版能做到"同一前缀被 N 个请求共享"，但**必须由调用方显式 fork**，且共享的是"最后一个不完整块之前的全部块 + state 槽"；被 fork 的序列在第一次 decode 前会把最后一块复制出来（COW），避免写坏兄弟。
- 疑点：`clone_states` / `clone_blocks` 的配对语义（`(old,new)` 交替）由 kernel 侧解释，开源树里只有 C++ 侧记录、没有消费代码（`scheduler.hpp:130-132` 只是 reserve）。

### #25 没有 hash / radix / LRU：`BlockManager` 不具备前缀缓存能力
- 证据：`csrc/engine/block_manager.hpp:119-180`（全部接口：`reset / can_allocate_nolock / allocate_nolock / free_blocks / pin_blocks / allocate_single_nolock / get_num_free_blocks`）
  `deepinfer/engine/llm_engine.py:265`（构造时只传块数）
- 等级：A
- 意味着什么：块的内容没有身份（无 hash key），因此"相同前缀"无法被自动识别；释放即入 free list，内容被下一个请求覆盖。**开源版的 prefix caching = 不存在**。
- 疑点：无（这是"没有"，不是"没找到"：`csrc/` 全树 grep `prefix|radix|hash` 无命中于 block manager）。

### #26 前缀命中的注入端口存在，但开源树里没有调用者
- 证据：`csrc/engine/pybind_sequence.cpp:118, 127`
  > `bind_sequence_property(sequence, "cached_hit_tokens", [](Sequence &seq) -> auto & { return seq.cache.hit_tokens; });`
  > `[](Sequence &seq, int value) { seq.cache.set_num_tokens(value); }, "Number of tokens with KVCache ready."`
  `csrc/engine/sequence.hpp:210-214`
  > `/// 外部注入命中数时同步建立增量基线。` / `void set_num_tokens(int value) { num_tokens = value; previous_num_tokens = value; }`
- 等级：A
- 意味着什么：设计上是"Entrypoint 在路由前算出命中长度 → 填 `block_table` + `cached_hit_tokens` + `num_cached_tokens` → 引擎只跑增量"。仓库里 `cached_hit_tokens` 在 Python 侧零引用（`grep` 仅 pybind 与 C++ 内部），所以这条链在开源版是断的。
- 疑点：`hit_tokens` 与 `num_cached_tokens` 两个字段语义重叠（前者"命中"，后者"已就绪"），`fork_seq` 只继承前者、`set_num_tokens` 设后者；两者何时必须相等没有断言。

### #27 落盘/装回的 API 只被测试调用
- 证据：`deepinfer/worker/cache_engine.py:762-818`（`select_state_slice_to_store`）、`:820-848`（`load_state_from_store_to_prefill_cache`）
  调用者：`grep -rn "select_state_slice_to_store|load_state_from_store_to_prefill_cache"` 仅命中 `tests/worker/test_state_cache_ops.py:138, 156, 175, 181, 196, 203, 220, 235, 265, 278, ...`
- 等级：A
- 意味着什么：这两个函数是**为 PE/CE 与 kvcache-store 预留的公开接口**，开源树里没有生产调用者。写书时不能说"开源版会把 SWA state 落盘"，只能说"预留了落盘接口，并有完整单测"。
- 疑点：函数是 `@classmethod` 且依赖 `GPUCacheInterface` 的类级布局字段（`cache_engine.py:494-537` 在 `_kvcache_layout` 里被填充），意味着**必须先跑过一次布局计算**才能调用，否则 `assert cls.cache_info_ready` 失败（`:719`）。

### #28 PD 分离只有枚举与虚函数，引擎子类不在仓库里
- 证据：`deepinfer/engine/engine_type.py:7-17`（`NORMAL / PROMPT / COMPLETION`）、`deepinfer/engine/llm_engine.py:801-822`（`init_pd_split`、`do_save_kvcache`、`pe_prepare_save_kvcache`、`pe_do_save_kvcache` 全是 `"""虚函数"""`）
  `grep -rn "LLMEngine)" --include=*.py deepinfer/ examples/` → 无结果
- 等级：A
- 意味着什么：官网/文档描述的 P/D 分离、kvcache-store 命中、3FS 落盘，在开源树里只有**接口形状**：`EngineType` 枚举 + C++ scheduler 分支 + 一组空实现的虚函数。这是"文档有、代码没有"的典型，必须标明。
- 疑点：`init_pd_split()` 在 `_init_cache` 里被无条件调用（`llm_engine.py:454-455`，仅非 NORMAL 时），基类实现是空函数，所以 `EngineType.PROMPT/COMPLETION` 在开源版能起服务但不会真的做 PD 传输。

### #29 显存规划：峰值实测 + `gpu_memory_utilization` − 2×state，再除以 block 字节数
- 证据：`deepinfer/worker/base.py:217-246`
  > `peak_allocated_memory = peak_torch_allocated_memory + max(non_torch_allocated_memory, 0)`
  > `available_kv_cache_memory = total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_allocated_memory`
  > `state_size = math.prod(CacheEngine.get_state_shape(...)); state_size *= 2  # state store & copy buffer`
  > `available_kv_cache_memory -= state_size`
  > `num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)`
  跨 rank 归并：`base.py:264-267`
  > `dist.all_reduce(num_gpu_blocks, op=dist.ReduceOp.SUM, group=PG.world_group())` / `num_gpu_blocks = num_gpu_blocks.item() // PG.world_group().size() // 2 * 2`
- 等级：A
- 意味着什么：`cache_block_size = CacheEngine.get_cache_block_size(...) = block_shape[0]`（`cache_engine.py:968-978`），即#3 的整块字节数。`state_size` 按 `[slots, slot_bytes]` 的**元素数**乘出来（uint8，所以元素数=字节数）。四舍五入到偶数块。
- 疑点：`llm_engine.py:426-428` 的 docstring 写 "we take the minimum number of blocks across all workers"，实现是 `SUM/size` = **算术平均**。异构显存下平均值可能大于最小可用值，这是文档与实现的实质不一致（A 级事实，影响正确性）。另外 `// 2 * 2` 只保证偶数，不保证能被 DP/EP 组整除。

### #30 分配走 `cuMemAlloc`，不计入 torch 统计；profile 用的临时 cache 也走同一条路
- 证据：`deepinfer/worker/cache_engine.py:910-916`
  > `meta = torch.empty(size=shape, dtype=torch.uint8, device="meta")` / `raw = cuda_utils.alloc_tensor(meta.numel() * meta.element_size())` / `return raw.view(meta.dtype).view_as(meta)`
  `deepinfer/worker/model_runner.py:498-507`
  > `# 分配走 cuMemAlloc，不计入 torch 显存统计，且随本函数返回即释放，因此不影响 profile_num_available_blocks 的计账。`
- 等级：A
- 意味着什么：KV/state 是裸设备内存，torch 的 `max_memory_allocated` 看不到它；所以 `profile_num_available_blocks` 必须在**分配 KV 之前**测峰值，且 `non_torch_allocated_memory` 要把 NCCL/驱动占用算进去。用 `device="meta"` 造 shape 再按字节数分配，是为了绕开 torch 的 allocator。
- 疑点：既然 KV 不在 torch 统计里，`torch.cuda.mem_get_info()` 的差值就是唯一真相；多进程/多引擎同卡时这个差值会包含别人的占用，代码没有防护。

### #31 启动期硬校验：`max_model_len` 不得超过总 token 容量的 80%
- 证据：`deepinfer/engine/llm_engine.py:441-451`
  > `max_seq_len = int(self.cache_config.block_size * num_gpu_blocks * 0.8)`
  > `if self.model_config.runtime_max_model_len > max_seq_len: raise ValueError(...)`
- 等级：A
- 意味着什么：即使只有一条请求，也不允许"单条请求就能吃满全部块"。这是 0.8 的经验余量，和调度水位 0.9 是两回事。
- 疑点：0.8 的来源没有注释；它与 `GPU_CACHE_PREFILL_WATERMARK = 0.9`（`scheduler.hpp:80`）叠加后，实际可用于单序列的比例被压到 ~72%。

### #32 调度侧水位：KV 使用率 ≤ 90% 才允许排 prefill
- 证据：`csrc/engine/scheduler.hpp:80`（`static constexpr float GPU_CACHE_PREFILL_WATERMARK = 0.9f`）、`:313`
  > `bool want = (get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK) && _has_prefill_waiting();`
  使用率定义：`scheduler.hpp:830-835`（已用 = 总数 − 空闲 + 预留）
- 等级：A
- 意味着什么：decode 与 prefill 抢同一池块，prefill 在 90% 处让步，避免 decode 因拿不到块被抢占。
- 疑点：水位只在 `is_any_point / is_all_point` 分支起作用（`:310-316`），普通路径的准入靠 `_can_allocate_with_state` 逐请求判断（#11），两套逻辑并存。

### #33 抢占默认"杀请求"，不 swap、不 offload
- 证据：`csrc/engine/scheduler.hpp:766-790`
  > `if (stop_preempted_request || so_blocks_recompute) { ... seq->schedule.status = SequenceStatus::FINISHED_PREEMPTED; ... }`
  > `else { ... seq->reset_for_recompute(); waiting.push_front(seq); ... }`
  默认值：`deepinfer/engine/arg_utils.py:40`（`stop_preempted_request: bool = True`）
- 等级：A
- 意味着什么：开源版没有 swap-out 到 CPU（`CacheEngine` 也只分配 GPU：`allocate_gpu_cache` 返回 `(gpu_cache, gpu_blocks, gpu_states)`，`cache_engine.py:918-966`）。块不够时默认直接失败/丢弃，这决定了容量规划必须留余量。
- 疑点：`cache_engine.py:851-857` 的类 docstring 还写着 "managing the GPU and CPU KV caches … swapping and copying"——vLLM 遗留文案，与实现不符。

### #34 MTP/nextn 层也是同一个 block 里的一层，`lm_head` 占位不算层
- 证据：`deepinfer/model_executor/layer_utils.py:10-16`
  > `"""遍历 attention list。注意 lm head 也会出现在 attention list 中，特判掉这一层，不计入 layer id。"""`
  > `runtime_layer_id = config_idx if config_idx < n_main_layers else config_idx - 1`
  `deepinfer/worker/cache_engine.py:875-882`
  > `self.num_layers = cfg.num_hidden_layers + int(envs.DEEPINFER_USE_NEXTN_DECODE)`
- 等级：A
- 意味着什么：`attn_module_list` 的长度是 `num_hidden_layers + 1（lm_head 占位） + n_extra`，`n_extra` 是 MTP/DFlash 额外的 attn 模块数；`KVCacheManager` 会给这些模块也切一份 `ca_cache/ica_cache/swa_cache/state`——**MTP 的 KV 不是免费蹭主模型的**，它按层计入 block 与 state。
- 疑点：`self.num_layers` 用 `+ int(envs.DEEPINFER_USE_NEXTN_DECODE)` 估算层数（= +1，与 nextn 深度 5 无关），暗示"只有一个 MTP 层"；这和 `iter_runtime_attn_modules` 的"任意个额外模块"能力不完全对齐。另外 `self.num_layers` 算出来只出现在 `cache_engine.py:936` 的注释里，**是死变量**；那条注释 `[num_gpu_blocks, num_layers, block_size, kv, num_heads, head_size]` 也已过时（真实形状是 `[num_gpu_blocks, kv_total_in_bytes]`）。

### #35 持久层不是一块盘：block KV 进 3FS，SWA state 留在主机内存
- 证据：`doc/inference_deployment_qa.md:131-135`（官方表格，原文）
  > `| In-flight block KV cache + state KV cache | Inference GPU memory | — | Prefill / decode instances |`
  > `| SWA KV cache | Local host memory | Single copy; nothing on 3FS | Prefill / decode instances; cross-machine hits pull over a 400G VLAN |`
  > `| Paged block KV cache | Dedicated 3FS storage cluster | No application-level replication ... | kvcache-store |`
  另见 `:169`（"The SWA KV cache does not enter the persistent cache; it stays in the memory of the machine that produced it."）
- 等级：B（官方文档；仓库里没有 kvcache-store 与主机内存层的代码，无法 A 级复核）
- 意味着什么：书里讲"KV 存哪"要分三层：**HBM（在飞）→ 主机内存（SWA state）→ 3FS（分页 block KV）**。SWA state 是"单副本、跟着产生它的机器走"，跨机命中要拉 400G VLAN；block KV 才是共享存储集群。把两者都写成"落盘"会错。
- 疑点：表里第一行的 "state KV cache"（HBM）与第二行的 "SWA KV cache"（主机内存）是同一对象的不同副本吗？从 `:163`/`:169` 看是（state cache 的 GPU 版 vs 主机版），但表格用词区分度低，容易让读者以为是两种缓存。

### #36 落盘位置 = `floor((L-1)/128) × 128`，且 chunked prefill 只有最后一个 chunk 落盘
- 证据：`doc/inference_deployment_qa.md:165`
  > `After the prefill stage: the store position is obtained by reserving one token at the end of the sequence and rounding down to a multiple of 128; if that leaves less than one full 128-token unit, nothing is stored this round. Nothing is stored part-way through a chunked prefill — only the last chunk stores.`
- 等级：B
- 意味着什么：这条补上了开源代码里只有 `assert last_hit_position % 128 == 0`（#17）而没有的"pos 怎么算"。`L=128` 时 `g=0`（本轮不落盘），`L=129` 时 `g=128`——**第一个 128 token 要等到第 129 个 token 处理完才落盘**。更关键的是：`g` 最多比当前处理位置 `L-1` 落后 127 个 token，而落盘窗口的右端固定在 `g` 上。这正是 #16 两个环尺寸的来源——默认环 134 只够"在边界处立刻复制"（复制最多迟 5 个 token，被 +6 吸收）；环 262 才够"边界落后一整段"时不丢数据（134 + 127，上取整到 262）。
- 疑点：公式里的"reserve one token"原因未写（E：保证窗口右端不越过已完成位置，chunked prefill 中途不产生半截窗口）。开源树里没有任何函数实现这个公式，它是 Entrypoint / kvcache-store 侧的约定；上面 262 的推导也是 E 级（数值吻合，但没有注释背书）。

## 3. 必答问题逐条回答

### Q1 块大小、块表结构、分配与释放方式（与 C++ `BlockManager` 协作）

- **块大小**：`block_size = 128` 个"压缩前 token"，硬编码在 `ModelMetadata.resolved_tokens_per_block`（`model_metadata_utils.py:134-135`），经 `arg_utils.py:84-96` 进入 `CacheConfig.block_size`。压缩层一个 block 实际装 `128/ratio` 个 token（C1=128，C2=64，`cache_engine.py:433, 452`）。块的字节数不是常数，而是按层累加：`Σ_layers pad32((128/ratio_l) × bytes_per_token_l)`，最后整块再对齐到 `lcm(128, bytes_per_token)`（`cache_engine.py:479-484`），得到 `kv_total_in_bytes` 作为每块的 stride（`cache_engine.py:889-893, 937-938`）。→ #1 #2 #3 #4
- **块表结构**：双层表示。C++ 侧每序列 `std::vector<int> block_table`（`sequence.hpp:206`），逻辑块 → 物理块；kernel 侧是 `[total_q, max_num_blocks]` int32 的**每 token 稠密副本**（`prepare_tensor_input_base.cpp:207-229`，ZOS 路径由 `copy_block_tables_tl` 在 GPU 上展开，`prepare_v4_decode_metadata.py:73,96`），`max_num_blocks` 由 `ceil(runtime_max_model_len/128)` 再向上取 64 的倍数（`model_runner.py:114-115`）。寻址统一为 `slot = block_id*tokens_per_block + offset_in_block`（`store_kv_cache.py:106-107`，`get_kv_index.py:51-56`）。→ #6 #7 #8
- **分配**：`BlockManager` 只有"整数槽池"一个能力：`allocate_nolock(n)` 从空闲链表头弹 n 个（`block_manager.hpp:138-145`），`allocate_single_nolock()` 用于 decode 每步（`scheduler.hpp:816-828`）。admission 时按 `get_num_blocks() = ceil((len + inflight + nextn)/block_size)` 一次性要够（`sequence.hpp:291-293`，`scheduler.hpp:426`）；块表容量另外预留 `+1` 块（`scheduler.hpp:190-192, 1106`）。KV 块与 state 槽是**两个** `BlockManager`，准入时联合检查（`llm_engine.py:265-266`，`scheduler.hpp:194-201`）。→ #9 #10 #11
- **释放**：`free_blocks_nolock` 逆序归还（`block_manager.hpp:147-149`），`ref_count` 归零才真正入 free list（`:95-103`）；序列结束/抢占时 `free_seq_nolock` 同时释放 `state_id` 与整张 block table（`scheduler.hpp:738-745`）。故意不用 0 号块（`block_manager.hpp:27`），所以实际容量是 `num_gpu_blocks - 1`。**没有 swap/offload**：抢占默认直接结束请求（`stop_preempted_request=True`），否则重算（`scheduler.hpp:766-790`）。→ #9 #33
- **与 C++ 协作的边界**：Python 侧只做"布局 + 一次性分配 + 视图切分"（`CacheEngine.allocate_gpu_cache` → `GPUCacheInterface` → `KVCacheManager.process`），物理块号的分配/回收全在 C++ scheduler；Python 拿到的 `block_table` 是只读的 `vector<int>`，经 `prepare_tensor_input` 变成稠密副本后喂给 kernel。→ #7 #11

### Q2 一共有几类缓存？各自形状、格式、存放位置

代码里 `KVCache` 只有 5 个字段（`cache_engine.py:205-218`），但外加"MTP 层的同构副本"与"引擎级副本区"，一共 5 类 + 2 类辅助：

| # | 名称 | 形状（代码原文） | dtype / 格式 | 存放位置 | 生命周期 |
|---|---|---|---|---|---|
| 1 | `ca_cache`（压缩主 KV，块状） | `[num_blocks, block_size//ratio, 1, bytes_per_token]` | uint8；FP4 288 B/token 或 MXFP8 528 B/token，`[data|scale]` per token | `gpu_blocks`（`[num_blocks, kv_total_bytes]`）的层内偏移段 | 随 block 分配/释放 |
| 2 | `ica_cache`（indexer K，块状） | `[num_blocks, block_size//ratio, 1, 68]` | uint8；FP4 e2m1 + ue8m0，**块内 data 段/scale 段分离** | 同上，紧邻 `ca_cache` 之前 | 同上 |
| 3 | `swa_cache`（滑窗，按序列槽） | `[num_state_seqs, swa_block_size, 1, 528]` | uint8；固定 MXFP8（`use_fp4=False` 写死，`cache_engine.py:261-263`） | `gpu_states` 的最前面 `swa_store_total_bytes_*` 段 | 随 state 槽分配/释放 |
| 4 | `ca_state`（C2 压缩边界累加器） | `[num_state_seqs, ring(6/7), 2, compress_dim]` fp32 | fp32，通道 0=score、1=latent | `gpu_states` 的 dropped 区，逐层交错 | 同上 |
| 5 | `ica_state`（indexer 压缩累加器） | `[num_state_seqs, ring(6/7), 2, index_head_dim]` fp32 | 同上 | 同上，排在 `ca_state` 前 | 同上 |
| 辅助 A | `copy_state`（decode 副本区） | 与 `gpu_states` 同形（`[num_state_seqs, slot_bytes]`） | uint8 | 独立 device buffer，`llm_engine.py:284` | 开源版恒 `None`（#22） |
| 辅助 B | `block_tables` 副本 | `[total_q, max_num_blocks]` int32 | int32 | input metadata workspace | 每步重建（#23） |

- 数量上：块状 KV 有 2 类（主 + index），按序列的 state 有 3 类（SWA + ca_state + ica_state），共 5 类；`KVCache` 一个实例就是"某层在这 5 类里占的那几段视图"，层数 = `num_hidden_layers` + 额外 MTP/DFlash attn 模块（`lm_head` 占位不计，#34）。
- 证据：`cache_engine.py:205-218`（形状注释）、`:407-422`（SWA 视图 `as_strided`）、`:428-443`（index）、`:447-465`（main）、`:367-374`（state 视图）；格式定义 `cache_common.py:109-155`；"index 缓存块内 data/scale 分段"见 `cache_common.py:31-32`（`Index caches keep block-wide data/scale sections.`）与 `:272`（非 interleaved 时 `storage[:, :block_bytes].view(...)`）。
- 上表是 **HBM 内的运行时副本**；持久层另算：分页 block KV 进 3FS 专用集群，SWA state 留在产生它的机器的主机内存（单副本、不进 3FS）——见 #35。
- → #4 #5 #12 #14 #34 #35

### Q3 `StateBlock` 槽位布局：存储区、丢弃区、环多出的 6 个位置

- **布局（从低地址到高地址）**：`[stored region: 前 swa_store_num_layer 层的 SWA 环] [dropped region: YOCO consumer SWA 环 | 逐层的 ica_state, ca_state]`。文档字符串原文见 `cache_engine.py:229-235`；代码切分见 `:348-362`（`_swa_blocks = state_tensor[:, :swa_total]`，`_state_res = state_tensor[:, swa_total:]`）；dropped 区内部偏移 `:352-354`。
- **stored 区**：每层 `swa_bytes_per_layer_kvstore_pad = align32(swa_block_size_kvstore × 528)`（`cache_engine.py:274-277`），只有 `layer_id < n_prefill_layer` 的层进来（`_is_dropped_swa_layer`，`:113-116`）；落盘时换成 `swa_bytes_per_layer_inference_pad = align32(134×528)=70,752`，所以 `state_size_to_disk = 20×70,752 = 1,415,040`（`:306`，文档 `:169` 一致）。
- **dropped 区**：`consumer SWA`（`layer_id >= n_prefill_layer` 的 SWA 层，`:411-413`）+ 每个 C2 层的 `ica_state` 与 `ca_state`（`:441-443, 463-465`），每层 2 个 state、`ica` 在前。`dropped_bytes_inference = swa_drop_total_bytes_inference_pad + compress_bytes`（`:300`）。
- **多出的 6 个位置**：`swa_block_size_inference = window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS`（`cache_engine.py:88-93`），而 `MAX_BACKTRACKING_TOKENS = max(用户值, USE_NEXTN_DECODE + 1)`（`environ.py:128-132`），默认 `5+1 = 6`。所以 `134 = 128 + 6`。这 6 个位置有两个用途，代码/文档各说了一半：① MTP 回退（`nextn` 个 draft 被拒要回退，`get_inference_swa_block_size` 的 docstring 写"nextn 需要回退 n 个 token"，`cache_engine.py:89-92`）；② 吸收"副本区越过 128 边界"的 5 个 token（`doc:165`）。→ #15 #16 #22
- **注意**：如果开了 `DEEPINFER_PREFILL_STORE_EXTRA_STATE`，槽内环是 `134 + 128 = 262`（`cache_engine.py:101-110`），多出的 128 不是"给回退的"，而是"给 128 落盘粒度留的一整段余量"；落盘前由 `select_state_slice` 把 262 环卷成 134 的连续窗口（`cuda/state_cache.py:55-67`）。默认 `EXTRA_STATE=False`（`environ.py:163`），此时 262 不出现。→ #16 #18 #19

### Q4 128 token 对齐存储、decode 期间的副本区在代码里怎么实现

分四件事，别混：

1. **落盘位置对齐**：`KVSTORE_GRAIN=128`（`cache_engine.py:32-33`），`select_state_slice_to_store` 在非 `TO_CE` 时断言 `last_hit_position % 128 == 0`（`:785-787`）。语义是"以该位置为结尾、不含该位置的窗口"，即 `[pos-134, pos)`。上游怎么算这个 pos（"保留末尾 1 个 token 再向下取整到 128"）不在开源树里，只有 `doc:165` 的描述（B 级）。
2. **让窗口在落盘时刻仍然完整**：环长 = 窗口 128 + 6。写入位置是 `pos % 134`，若一个请求连续写了 134 个 token，环刚好容纳 `[pos-134, pos)`；多出的 6 个位置的用途是**容忍落盘滞后**——落盘在边界 `g` 发生，而当前处理位置可能已到 `g+5`（`doc:165` 的"最多越过 5 个 token"），此时 `[g-134, g)` 仍在环内。MTP 回退（位置倒退）也靠这 6 个位置（#16）。`state_tensor` 侧同样是长度 134 的滚动数组，用绝对位置 mod 134 定位（`cuda/state_cache.py:66`）。
3. **从环到连续窗口的"重排"**：`select_state_slice`（`cuda/state_cache.py:16-69`）逐层逐字节做 `(start_pos+i) mod K → (start_pos+i) mod W` 的 gather；不需要重排时（`EXTRA_STATE=False`）直接 `state_cache[state_id, :state_size].copy_()`（`cache_engine.py:788-797`）。装回方向对称：`load_state_to_cache`（`cuda/state_cache.py:80-136`）+ `load_state_from_store_to_prefill_cache`（`cache_engine.py:820-848`）。
4. **decode 副本区**：`copy_state` 是与 `gpu_states` 同形的第二份 state（`llm_engine.py:284-285`），跨 128 边界时 `copy_state[state_id].copy_(gpu_states[state_id])` 整槽覆盖并记录 `last_copy_state_token_idx`（`:787-798`），事件 `copy_state_event` 在下一步 pre-sampler 前同步（`:930-937`）。请求结束时"从最近一次完整副本落盘"（`doc:165`，B 级）。显存代价直接体现在 profile：`state_size *= 2  # state store & copy buffer`（`worker/base.py:243`）。
   - 关键限定：**开源版只到钩子为止**。`copy_state` 从未被分配、`should_copy_state` 恒 `False`、`do_save_kvcache` 是空虚函数（`llm_engine.py:811-822`）。真正实现它的 decode 引擎不在仓库里（#28）。
   - 另一块容易混的"副本区"是块表副本 `[total_q, max_num_blocks]` int32（#7 #23），它在 input metadata 里，和 state 无关。

- → #17 #18 #19 #22 #23

### Q5 开源版有没有前缀缓存？有的话显存内怎么做；没有的话入口在哪

- **没有自动前缀缓存**。`BlockManager` 的全部能力是"分配/释放/引用计数"，没有 key、没有 hash、没有 LRU、没有淘汰策略（`block_manager.hpp:119-180`）；释放即入 FIFO 空闲链，内容会被下一次分配直接覆盖（`:95-103`）。`csrc/` 全树没有 radix/hash 结构（#25）。
- **显存内唯一能共享前缀的机制是显式 `fork_seq`**：把已 prefill 的序列 fork 成 N 条，`pin_blocks` 把整张 block table 的 `ref_count` 加一、`state_id` 也 pin（`scheduler.hpp:1108-1114`）；被 fork 的序列第一次 decode 时，`_clone_state` 分配新 state 槽、`_clone_blocks` 复制最后一个不完整块（COW），其余块继续共享（`:602-617`）。用法见 `examples/benchmark.py:331-353`。这**不是** prefix cache：没有命中查询、没有跨请求复用、进程退出即丢。
- **预留的入口（按成熟度排序）**：
  1. `Sequence.cached_hit_tokens` / `num_cached_tokens` 的 pybind setter（`pybind_sequence.cpp:118,127`）+ `KvCacheState.set_num_tokens`（`sequence.hpp:210-214`）——外部算出命中长度后注入，引擎只跑增量。Python 侧零调用者（#26）。
  2. `Scheduler.fork_seq` 的 `blocks` 参数允许直接塞入一张现成 block table（`scheduler.hpp:1087-1094`），即"外部把磁盘上的块搬进显存后重建序列"这条路是通的。
  3. state 侧的 `select_state_slice_to_store` / `load_state_from_store_to_prefill_cache`（`cache_engine.py:762-848`）+ `KVSTORE_GRAIN` + `total_bytes_inference/state_size_to_disk` 的区分——完整的"落盘/装回"接口，只有单测（#27）。
  4. 引擎侧虚函数 `init_pd_split` / `do_save_kvcache` / `pe_prepare_save_kvcache` / `pe_do_save_kvcache` / `should_copy_state` / `copy_state`（`llm_engine.py:284-285, 801-822`）——PD 分离与落盘调度的挂点，基类全空（#28）。
  5. `EngineType.PROMPT / COMPLETION` 枚举与 C++ scheduler 的对应分支（`engine_type.py:7-17`，`scheduler.hpp` 中 `EngineType` 分支）——部署形态的挂点，子类缺失。
- 一句话：**入口齐全、实现缺席**。书里可以写"开源版把前缀缓存做成了可插拔的外部组件（Entrypoint + kvcache-store），引擎只保留注入端口"，但不能写"开源版支持前缀缓存"。
- 生产形态（B 级，官方文档）：**匹配在 Entrypoint 做，是对 metadata 的 hash-chain 前缀匹配**；命中长度必须在路由前算出，所以这段开销藏不掉。**取 KV** 由独立进程异步从 3FS 搬到主机共享内存；**swap-in** 在引擎进程内的线程里做，目标是让逐层 H2D 与前向重叠，且因为 CUDA copy engine 处理细碎 KV 块性能差，swap-in 用 GDR（RDMA NIC 打满 PCIe，不占 SM）。原文 `doc/inference_deployment_qa.md:157`；另见 `:141`（专用 3FS 集群的好处：任何 prefill 机器都能命中任意前缀，避免"前缀亲和 vs 负载均衡"冲突）、`:153`（3FS 侧无 TTL 逻辑，回收靠 3FS GC；agent 流量 3 天、chatbot 7 天）。
- 与 DeepInfer 开源版的对应关系：`cached_hit_tokens` / `num_cached_tokens` 就是 Entrypoint 把"hash-chain 匹配结果"注入引擎的字段（#26）；`block_table` 由外部重建（`fork_seq(blocks=...)`，`scheduler.hpp:1087-1094`）；state 由 `load_state_from_store_to_prefill_cache` 装回（#27）。三条链路在开源树里都在，只是没有上游服务。

### Q6 显存容量规划：开源版如何决定分配多少块

- **输入**：`total_gpu_memory`（`torch.cuda.mem_get_info`）、`gpu_memory_utilization`（默认 0.90，`arg_utils.py:34`）、profile run 的峰值、`state_shape`、`cache_block_size`。
- **公式**（`worker/base.py:217-246`）：
  ```
  peak = torch.max_memory_allocated() + max(total - free - torch_allocated, 0)
  avail = total × gpu_memory_utilization − peak
  state_bytes = num_state_slots × slot_bytes          # get_state_shape 的乘积
  avail −= 2 × state_bytes                            # state store & copy buffer
  num_gpu_blocks = floor(avail / block_bytes)         # block_bytes = kv_total_in_bytes
  ```
  其中 `num_state_slots = int(max_num_seqs × 1.2) + 1`（`cache_engine.py:201-202`，默认 `DEEPINFER_NUM_STATE_SLOTS_RATIO=1.2`，`environ.py:162`），`max_num_seqs` 默认 256（`arg_utils.py:36`）→ **308 槽**；`slot_bytes` 由#12/#20 的布局算出。
- **跨 rank 归并**：`all_reduce(SUM) / world_size`，再 `//2*2`（`base.py:264-267`）。注意 docstring 说"取最小"，实现是取平均（#29 疑点）。
- **启动期校验**：`num_gpu_blocks ≤ 0` 直接报错；`runtime_max_model_len > block_size × num_gpu_blocks × 0.8` 也报错（`llm_engine.py:441-451`）。
- **运行期**：块不够时先按 90% 水位拒 prefill（`scheduler.hpp:80,313`），再按 `_can_allocate_with_state` 逐请求 `LATER`，最后 `_preempt`（默认杀请求，`scheduler.hpp:766-790`）。没有 swap、没有 CPU KV（#33）。
- **分配实现**：`_alloc_gpu_uint8` 造 meta 张量取 shape，再用 `cuda_utils.alloc_tensor(bytes)` 走 `cuMemAlloc`，绕过 torch allocator（`cache_engine.py:910-916`）；profile 用的临时 cache 也走这条路且随即释放，因此不影响计账（`model_runner.py:498-507`）。
- **实操口径**：`num_gpu_blocks` 是全局常量，所有 rank 相同；state 槽数与 `max_num_seqs` 绑定，与块数无关——**加 `max_num_seqs` 会同时吃 state 内存（×2）和调度并发，但不会减少块数以外的 KV 容量**。
- → #29 #30 #31 #32 #33 #20 #21

## 4. 新发现的问题

1. **`num_gpu_blocks` 的跨 rank 归并是平均值而不是最小值**（`worker/base.py:264-267` vs `llm_engine.py:426-428` 的 docstring）。异构显存/不同 DP rank 上残留不同时，平均值可能大于某些 rank 的实际可用量 → 那些 rank 会在 `allocate_gpu_cache` 时 OOM。值得在书里作为"开源实现与注释不一致"的例子，或建议主编向作者确认。
2. **`copy_state` 的分配者缺失**：`llm_engine.py:284` 只把它设为 `None`，`should_copy_state` 恒 `False`。这意味着开源版 **decode 期间不会产生 SWA 落盘快照**，`do_save_kvcache` 也永不执行。文档描述的"decode 也写盘以提高命中率"（`doc:149`）在开源树里无法复现。
3. **`page` 与 `state` 的容量耦合没有校验**：`num_state_slots = int(max_num_seqs×1.2)+1` 是个 magic ratio，没有任何断言保证"同时在跑的请求数 ≤ 槽数"。`_can_allocate_with_state` 只在准入时检查，长尾请求 + fork 可能把 state 槽先耗尽，而块还富余。
4. **`_has_SWA` 的占位槽浪费**：`use_swa=False` 的层也占一个 SWA slot（`cache_engine.py:684` 注释），这些层被划进 dropped 区。按每层 `134 × 528 = 70,752 B`、308 槽算，每个占位层 ≈ 21.8 MB（E 级估算；若开 `EXTRA_STATE` 则按 262 环算，翻倍）。是否可以用 `swa_store_num_layer` 之外的字段区分"占位"与"实用"，代码里没有。
5. **`state_cache.py` 顶部 `import pytest`**（`deepinfer/model_executor/layers/tilelang_ops/state_cache.py:6`），而 `cache_engine.py:23-26` 在生产路径 import 它。等于引擎运行时依赖 pytest。这是把测试辅助代码放进 runtime 模块的痕迹，值得作为"工程质量"脚注。
6. **`CacheEngine` 的类 docstring 与实现不符**：`cache_engine.py:851-857` 写 "GPU and CPU KV caches … swapping and copying"，实际只分配 GPU，也没有 swap 函数。vLLM 遗留文案。
7. **`self.num_layers` 是死变量**（`cache_engine.py:875-882` 计算，仅出现在 `:936` 的过时注释里）；`:936` 注释的形状 `[num_gpu_blocks, num_layers, block_size, kv, num_heads, head_size]` 与真实的 `[num_gpu_blocks, kv_total_in_bytes]` 不符。书里若引用这行注释会写错。
8. **`lru_cache(8)` 与 env 的耦合**（`cache_engine.py:130-138`）：`get_inference_compress_block_size` 缓存了环境变量结果，改 `DEEPINFER_MAX_BACKTRACKING_TOKENS` 后必须显式 `clear_compress_block_size_cache()`，测试里专门处理了（`test_state_cache_ops.py:102-107`）。生产里若在运行中改 env，环大小会不一致。
9. **`last_hit_position` 在 `TO_CE=True` 时不做 128 断言**（`cache_engine.py:785-787`），但下游 `select_state_slice` 仍按 134 窗口取数；非对齐位置的窗口语义没有文档，CE 侧如何消费未知。
10. **`max_num_blocks` 向上取 64 的倍数**（`model_runner.py:115`）会让 `block_tables` 副本比实际需要大最多 63 列；对 128k/128 是 1024（正好整除），对更短上下文（如 8k → 64）刚好，但对 100k → 782→832，多 6%。这是纯显存浪费，可写进"工程细节"。

## 5. 可用于图表的材料

### 5.1 一个 KV block 的字节布局（示意，按 runtime 层序）
```
block_id = 7, 128 个压缩前 token
┌──────────────────────────────────────────────────────────────────────┐
│ layer 0: ica_cache  [64 × 68 B = 4,352 B]  ← pad32 后 4,352 B        │
│ layer 0: ca_cache   [64 × 288 B = 18,432 B]（ratio=2）               │
│ layer 1: ica_cache  [128 × 68 B = 8,704 B]                           │
│ layer 1: ca_cache   [128 × 288 B = 36,864 B]（ratio=1）              │
│ ...                                                                  │
├──────────────────────────────────────────────────────────────────────┤
│ 尾部对齐 pad，使整块 = lcm(128, 288) = 1,152 B 的整数倍              │
└──────────────────────────────────────────────────────────────────────┘
layout 三元组 = (layer_id, kv_offset, kv_length)，kv_offset % 32 == 0
```
数据点（已核算）：`128×288=36,864`、`64×288=18,432`、`128×68=8,704`、`64×68=4,352`；`lcm(128,288)=1,152`、`lcm(128,528)=4,224`、`lcm(32,528)=1,056`。

### 5.2 `StateBlock` 一个槽的布局（`EXTRA_STATE=True` 时）
```
state_tensor[state_id]  （长度 = total_bytes_kvstore，对齐 1,056）
┌───────────────────────────────┬───────────────────────────────┬─────────────────────────────┐
│ stored SWA（前 20 层）        │ dropped SWA（consumer 层）    │ compression state           │
│ 每层 262 × 528 = 138,336 B    │ 每层 262 × 528 B              │ 逐层 [ica_state, ca_state]  │
│ 合计 2,766,720 B（E，按 20 层）│                              │ ring 6/7 × 2 × dim × 4 B    │
└───────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
 ↑ 落盘只取前 state_size_to_disk = 20 × 134 × 528 = 1,415,040 B（inference 环，重排后）
 ↑ PE→CE 取前 total_bytes_inference（含 dropped 区）
```

### 5.3 五类缓存对照表
| 类别 | 是否分页 | 粒度 | 每单位字节 | 谁分配 |
|---|---|---|---|---|
| `ca_cache` | 是（block 池） | 每 token | 288 / 528 | 块号：`BlockManager`；字节段：`KVCacheManager.process` |
| `ica_cache` | 是（block 池） | 每 token | 68 / 132 | 同上，同一块内的前一段 |
| `swa_cache` | 否（序列槽） | 每 token（环） | 528 | 槽号：`BlockManager`（state 槽）；字节段：`process` |
| `ca_state` | 否（序列槽） | 每 ring 项 | `2×compress_dim×4` | 同上（`_alloc_state`） |
| `ica_state` | 否（序列槽） | 每 ring 项 | `2×index_head_dim×4` | 同上（`_alloc_state`） |

### 5.4 显存规划伪代码（可直接画流程图）
```
profile_run(dummy seqs, 临时 CacheEngine)
  → peak = torch_peak + non_torch
  → avail = total × 0.90 − peak
  → state_bytes = (int(max_num_seqs×1.2)+1) × slot_bytes
  → avail −= 2 × state_bytes
  → blocks = floor(avail / block_bytes)
  → all_reduce(SUM)/world_size, //2*2
  → assert blocks > 0
  → assert runtime_max_model_len ≤ 128 × blocks × 0.8
```

### 5.5 decode 单步的 KV 相关 GPU 工作（时序素材）
1. C++/GPU 生成 `slot_mapping_store_swa`（`state_id×K + pos%K`，`prepare_v4_decode_metadata.py:217`）。
2. 生成 `block_tables` 副本 `[total_q, max_num_blocks]`（`prepare_tensor_input_base.cpp:217-227` 或 `copy_block_tables_tl`）。
3. 生成 `kv_ca_index`（`[total_q, pad128(max_model_len/ratio)]`）与 `kv_swa_index`（`[total_q, window]`），`-1` 为无效。
4. 写新 KV：`store_kv_cache(ca_cache, slot_mapping_store_ca)` + `store_kv_cache(swa_cache, slot_mapping_store_swa)`。
5. 读历史：SWA 用环索引、CA 用 paged 索引，`decode_mla_mqa_sparse` 直接吃 `kv_cache.swa_cache` + `kv_ca_index`（`v4_modules/attn.py:640-653`）。
6. 若跨 128 边界且副本区可用：`copy_state[state_id].copy_(gpu_states[state_id])` + `copy_state_event.record()`（`llm_engine.py:787-798, 935-937`）。

### 5.6 关键数字（可直接进表格）
- 每 token：main KV `288`（FP4）/`528`（MXFP8）；index K `68`（FP4）/`132`（MXFP8）；SWA 固定 `528`。
- 每 block：128 个压缩前 token；C1 层 128 token、C2 层 64 token。
- 环：SWA 推理 `134 = 128+6`，kvstore `262 = 134+128`；压缩 state `6`（C1）/`7`（C2）。
- 落盘：`1,415,040 B/seq = 20 × 134 × 528`。
- 槽数：`308 = int(256×1.2)+1`（`max_num_seqs=256`）。
- 对齐：`32 B`（层偏移）、`1,152 B`（block，FP4 main）、`4,224 B`（block，MXFP8 main）、`1,056 B`（state 槽）。

### 5.7 三层存储（画"KV 存哪"用，B 级）
| 层 | 内容 | 介质 | 副本策略 | 谁管 |
|---|---|---|---|---|
| L1（在飞） | 分页 block KV + state KV | HBM | — | prefill / decode 实例 |
| L2（主机） | SWA state（`state_size_to_disk` 那 1.4 MB） | 本机主机内存 | 单副本，不进 3FS；跨机命中走 400G VLAN | prefill / decode 实例 |
| L3（持久） | 分页 block KV（main + indexer K） | 专用 3FS 集群 | 不做应用级复制，靠 3FS 自身冗余 | kvcache-store |

数据点：3FS 侧约 100 PiB SSD，TTL agent 3 天 / chatbot 7 天；3FS 侧无应用层过期逻辑，回收靠 3FS GC（`doc/inference_deployment_qa.md:153`）。
