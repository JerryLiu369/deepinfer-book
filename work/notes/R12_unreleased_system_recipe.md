# R12 未开源部分的系统设计与 deepseek-recipe

## 0. 元信息

- 使用的代码版本：
  - `deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21，本地 `work/sources/DeepInfer`）
  - `deepseek-ai/deepseek-recipe` @ `935817c66f5d6e64838f419692545f06787e485f`（v0.2.0，2026-10-20，本地 `work/sources/deepseek-recipe`）
- 查阅的主要来源（均为 2026-10-23 访问）：
  - `doc/inference_deployment_qa.md`（312 行，官方英文 Q&A，commit 内文件，B 级）
  - 知乎《DeepSeek V4.1 Flash 推理系统概览》 https://zhuanlan.zhihu.com/p/2095560295930508953 （2026-10-20 07:30 编辑，中文；本次重抓全文 537 行快照，B 级）
  - `doc/images/architecture_overview.png`（官方架构图，用视觉读取，B 级）
  - DeepInfer `README.md`、`deepinfer/engine/*`、`csrc/engine/scheduler.hpp`、`deepinfer/entrypoints/`、`deepinfer/config.py`、`deepinfer/utils.py`
  - deepseek-recipe `README.md`、`docs/tokenizer.md`、`docs/streaming.md`、`CHANGELOG.md`、`deepseek-recipe-*/src/*.rs`、各 `Cargo.toml`
  - DualPath 论文：SIGCOMM '26（DOI 10.1145/3789240.3829159），arXiv:2602.21548v2 全文（B 级，与 DeepInfer 同作者群、同用 3FS）
- 说明：知乎全文与 Q&A 同源（Q&A 第 21 行自述 "This Q&A builds on that blog post"）。凡两处数字冲突，本笔记并列并标注。

## 1. 三个最重要的发现

1. **仓库里没有 P/D 传输和 KV 存储的实现，只有"插槽"。** 开源代码留了 `EngineType` 三态、7 个空实现的虚函数、C++ 注释里 "prefill_engine … will be transferred to complete_engine"、以及 `copy_state` 缓冲区与 `should_copy_state()`（恒返回 `False`）；README 那句 "most required engine functionality is already implemented" 说的是这些插槽与调度/状态复制机制，不是跨机传输本身。
2. **"四级负载均衡"不是四个平级入口组件。** 架构图显示 Entrypoint 后面只有 3 个路由 LB（Prefill / Decode / VE），EPLB 画在 Prefill Service 与 Decode Service **内部**；而"命中长度必须在路由前算出"这一条，决定了 Entrypoint 必须先做前缀匹配，这是整套 KV 存储设计的性能约束源头。
3. **两份官方文本对 `cost(q, c)` 的 `q` 定义不一致**：QA 写 `q` = 序列 token 数，知乎写 `q` = 序列中**未命中**的 token 数。开源代码里没有任何代价模型（`compute_quota` grep 0 命中），书里写这个公式必须注明取自哪一份，否则会写出自相矛盾的模型。

## 2. 事实条目

### A. 系统总览与部署形态

#### #1 整体形态：E/P/D 分离 + 专用存储节点
- 证据：`doc/inference_deployment_qa.md:41-43`
  > - What is the overall shape of the inference system?
  >     - A disaggregated E/P/D architecture plus dedicated storage nodes.
- 等级：B
- 意味着什么：书里讲"服务架构"要先画 E/P/D 三段，而不是 P/D 两段；VE 是独立一段，有自己的负载均衡器。
- 疑点：无。

#### #2 并行配置：Prefill 小 EP/DP，Decode Wide-EP，VE 单卡，不用 TP/CP
- 证据：`doc/inference_deployment_qa.md:45-53`
  > - Prefill: EP2–8 / DP2–8; use the smallest EP/DP that HBM allows.
  > - Decode: EP64–128 / DP64–128 (Wide-EP); use the largest EP/DP that the available GPUs allow.
  > - VE: a single GPU per instance is generally enough, and cheaper GPUs are recommended.
  > - TP and CP are not used for now.
- 等级：B
- 意味着什么："P 用普通 8 卡机、D 用 64–128 卡超节点"是硬约束，不是调优偏好；也因此 P 与 D 的实例粒度差 8–16 倍，P/D 配比只能按实例数而非卡数对齐。
- 疑点：知乎同一处写 `Prefill: EP2～8/DP2～8`、`Decode: EP64~128/DP64~128`，与 QA 一致（知乎快照 94-96 行）。

#### #3 Wide-EP 的收益口径
- 证据：`doc/inference_deployment_qa.md:55-59`
  > Wide-EP substantially reduces per-step memory traffic in the MoE expert part and, with good EPLB support, effectively raises TPS.
- 等级：B
- 意味着什么："Wide-EP 提吞吐"的机制是**每步 MoE 权重流量下降**，前提是 EPLB 足够好；两个条件缺一不可，不能单独把 EP64 当银弹写。
- 疑点：随文配图 `images/wide_ep_decode_pareto.png` 的具体数据点留给 R16。

#### #4 最小部署单元：1P+1D 可跑，实用最小单元 64P+64D
- 证据：`doc/inference_deployment_qa.md:65-67`
  > 1P + 1D is deployable. In practice the GPU ratio between P and D is roughly 1:1. Since the recommended decode configuration is EP64 or above, the smallest efficient deployment unit is 64 GPUs of prefill plus 64 GPUs of decode.
- 等级：B
- 意味着什么：这是全书唯一给出的"多大规模才像个服务"的门槛数字：128 卡。1P+1D 只是能跑通，不是能服务。
- 疑点："roughly 1:1" 是官方线上口径，与 #55 的自动扩缩容（P 按未命中 token 数、D 按在算请求数）并不等价，书里不要把 1:1 当恒定值。

#### #5 跨数据中心：技术上可行，每 8 卡约 400 Gbps，实际部署在相邻机房
- 证据：`doc/inference_deployment_qa.md:71-73`
  > splitting P and D across data centers is technically feasible (at some cost in reliability …). Budget roughly 400 Gbps of interconnect bandwidth per 8 GPUs. In practice, DeepSeek's inference nodes are deployed in different but adjacent data centers.
- 等级：B
- 意味着什么：3FS 集中式存储 + 跨机房 P/D 的前提是带宽预算可算：400 Gbps/8 卡 ≈ 50 Gbps/卡。书里可以把这条当成"存储集中化的网络代价"。
- 疑点："at some cost in reliability" 没有量化；光纤被施工挖断是原文举的例子，属于运维风险不是设计指标。

#### #6 三个设计目标与唯一的营收口径
- 证据：`doc/inference_deployment_qa.md:7-17`
  > 1. Support accelerators from multiple vendors …
  > 3. **Retain KV cache for more than 72 hours** …
  > A single cluster currently serves 155 trillion tokens per day, with overall revenue of US$38 per GPU-hour (at valley pricing).
- 等级：B
- 意味着什么：`>72h` 的 KV 留存是**设计目标**，与后文 TTL 3/7 天（#46）是同一件事的两种说法；`155 万亿 tokens/天`、`$38/卡·小时（谷价）` 是全书唯一官方营收数字，引用必须带"单集群""谷价"两个限定。
- 疑点：`$38 per GPU-hour` 与知乎"整体收入是每卡每小时 38 美金（按谷价）"逐字对应（知乎快照 67 行），两处都不是利润。

#### #7 开源边界的官方原话
- 证据：`doc/inference_deployment_qa.md:21`
  > The KV cache management and load balancing components of the inference system are tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced.
  知乎同段（快照 71 行）：
  > 推理系统中的 KVCache 管理与负载均衡组件因与内部基础设施耦合较深，相关代码目前尚未完整开源。
- 等级：B
- 意味着什么：**未开源的是"KVCache 管理 + 负载均衡"两类组件**，理由是"与内部基础设施耦合"。书里写"DeepSeek 没开源核心"要落到这两个具体组件上，不要泛化成"引擎没开源"。
- 疑点：`not yet fully open-sourced`（尚未完整开源）留了后门，不等于永不开源。

#### #8 README 自己列的"缺件清单"
- 证据：`README.md:38-42`
  > Required for a complete inference service (not included in this release):
  > - A KV cache system
  > - VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)
  > - Load balancing across data-parallel replicas for both prefill and decode
- 等级：B（官方仓库文档）
- 意味着什么：这是"哪些有、哪些没有"的最权威清单。括号里那句 `most required engine functionality is already implemented` 极易被误读成"P/D 分离已实现"——实际指的是引擎侧为分离预留的机制（见 #66–#75）。
- 疑点：无。

### B. 集群与存储

#### #9 单台 KV 存储服务器配置
- 证据：`doc/inference_deployment_qa.md:75-77`
  > 20 × 30.73 TB SSDs per KV cache storage server, with each server providing at least 40 GB/s read and 5 GB/s write throughput and enough total capacity to retain 24 hours of data.
- 等级：B
- 意味着什么：单机裸容量 20×30.73 = 614.6 TB（计算值），读 40 GB/s / 写 5 GB/s —— 读:写 = 8:1，说明这套存储是**读多写少**的缓存盘，不是训练存储。
- 疑点：`retain 24 hours of data` 与 #46 的 TTL 3 天 / 7 天口径不一致，见 §4 问题 1。

#### #10 存储集群与在线 GPU 的配比：14.4 TB/GPU（逻辑）
- 证据：`doc/inference_deployment_qa.md:79-81`
  > In practice, about 14.4 TB per GPU measured as logical size; the physical capacity required is larger than that.
- 等级：B
- 意味着什么：可算容量题：64P+64D = 128 卡 → 128 × 14.4 = 1843.2 TB 逻辑，约 3 台 #9 的裸容量（614.6 TB/台）才够，物理还要更多（EC 与水位线）。书里做"一个集群要多少存储"的估算就靠这两个数。
- 疑点："logical size" 未定义是否含 EC 开销与副本；`physical capacity required is larger` 没给放大系数。

#### #11 3FS 的可靠性机制：10+2 EC、读 11 取 10、GC Worker 按 LRU
- 证据：`doc/inference_deployment_qa.md:83-91`
  > Data is stored on 3FS with 10+2 erasure coding; each EC group tolerates two simultaneous node failures.
  > A read issues requests to all 11 storage nodes in the group simultaneously; any 10 responses are enough to decode and return.
  > A GC Worker component reclaims expired cache: one instance is deployed per storage node and periodically evicts the least recently used KV cache under an LRU policy, keeping disk usage below a watermark.
- 等级：B
- 意味着什么：这是"KV 缓存不是普通文件"的关键证据：读放大到 11 台（尾延迟由最慢的那台决定），回收在存储侧用 LRU + 水位线，应用侧不参与。
- 疑点：读取时"任意 10 个响应即可解码"意味着每次读都打 11 台，带宽成本 ×1.1；书中若要解释 10 TiB/s 聚合读带宽的来源，需注意这是 EC 解码前的物理读还是逻辑读，原文没说。

#### #12 3FS 的容量、块粒度与带宽（知乎独有）
- 证据：知乎 §Block KVCache（快照 196 行）
  > 3FS 分布式文件系统为 KVCache 提供约 100 PiB 的 SSD 存储 …… 3FS 存储集群针对 KVCache block 粒度（约 112 KiB）的随机读写进行优化，峰值聚合读带宽达到 10 TiB/s，峰值聚合写带宽约 850 GiB/s。
- 等级：B
- 意味着什么：`112 KiB` 是后面 GDR 论证（#43）的物理依据：不是一个抽象的"小块"，而是可算的具体字节数。10 TiB/s 读 / 850 GiB/s 写的读写不对称（约 12:1）与 #9 的 8:1 同向。
- 疑点：`100 PiB` 是"为 KVCache 提供"的总容量，未说清是单集群还是全公司；10 TiB/s 是"峰值聚合"，未给集群规模，不能直接与单机 40 GB/s 做除法推机器数。

#### #13 TTL：agent 3 天、chatbot 7 天；3 天 TTL 让 chatbot 命中率掉 0.5%–1%
- 证据：`doc/inference_deployment_qa.md:151-153`
  > There is no expiry logic on the kvcache-store side; reclamation happens entirely in 3FS's own GC. … a TTL of 3 days for agent traffic and 7 days for chatbot traffic. … Empirically, a TTL of around 3 days costs chatbot traffic 0.5%–1% of KV cache hit rate.
- 等级：B
- 意味着什么："按流量类型给不同 TTL"是这套系统的成本杠杆：chatbot 长尾更重（少量 KV 被读很多次），所以给它 7 天；agent 3 天。0.5%–1% 是把 TTL 从 7 天砍到 3 天的**代价**，不是收益。
- 疑点：该 0.5%–1% 无统计窗口、无流量规模，属于经验值（原文 "Empirically"）。

### C. 四级负载均衡

#### #14 四个 LB 的清单与分工
- 证据：`doc/inference_deployment_qa.md:33`
  > … we built a **Prefill Load Balancer, Decode Load Balancer, Vision Encoder Load Balancer, and Expert-Parallel Load Balancer**.
  知乎 §负载均衡（快照 138 行）：
  > Prefill/Decode Load Balancer 通过请求路由和调度来分别控制 prefill 和 decode 阶段不同卡上注意力子层的负载均衡 …… Expert-Parallel Load Balancer 通过冗余专家和动态调整来控制 MoE 子层的负载均衡。
- 等级：B
- 意味着什么：四个 LB 的**作用对象不同**：前三个均衡"注意力子层"（按请求路由），EPLB 均衡"MoE 子层"（按专家搬家）。书里不该把它们并列成同一层的四个组件。
- 疑点：无。

#### #15 架构图上的真实拓扑：Entrypoint×3 → 3 个路由 LB；EPLB 在 Service 内部
- 证据：`doc/images/architecture_overview.png`（视觉读取，1727×911）
  - 左：`User Request (JSON)` → `DeepSeek Recipe`（绿）→ `Token List + Sampling Parameters`
  - 中：三个堆叠的 `Entrypoint` 方框（粉）
  - 中右：`Prefill Load Balancer` / `Decode Load Balancer` / `Vision Encoder Load Balancer`（蓝，三个平级）
  - 右：`Prefill Service`（黄，内含 `Expert-Parallel Load Balancer`）、`Decode Service`（橙，内含 `Expert-Parallel Load Balancer`）、`Vision Encoder Service`（紫）
  - 底部：`External KVCache Storage (3FS)`，与 Prefill/Decode Service 双向箭头
  - 图例：黑箭头 = Request，蓝箭头 = KVCache；Prefill Service → Decode Service 有一条蓝箭头
- 等级：B（官方图，读图结论）
- 意味着什么：这张图纠正了"四个 LB 都在入口"的直觉：入口层只有 3 个路由 LB，EPLB 是每个 Service 内部的 MoE 均衡器。另外图上根本没有单独的 `kvcache-store` 方框，只有 `External KVCache Storage (3FS)`——`kvcache-store` 是逻辑组件（#40）。
- 疑点：三个 Entrypoint 方框是"多副本"还是"三层"无法从图上判定（文字侧只支持"多台/按 key 分片"，见 #49）。

#### #16 Prefill LB 的四层结构
- 证据：`doc/inference_deployment_qa.md:107-115`
  > Bucketing (tiers): … 4 tiers — <16K / 16–64K / 64–256K / ≥256K. The number of engines assigned to each tier is redistributed dynamically according to that tier's share of compute-quota demand in the previous round, and each tier has its own per-batch compute-quota cap.
  > Across DP groups: engines pull work according to their remaining compute quota. Since any engine can hit any prefix, there is no prefix-affinity routing.
  > Within a DP group: … each request as it arrives goes to the DP rank with the lowest current quota load.
  > Within a single GPU: requests are dequeued FIFO, and the chunk size is not fixed but solved from a budget.
- 等级：B
- 意味着什么：这是书里最好画的一张"四级决策表"：桶级（分资源）→ 组间（拉取）→ 组内（最小配额）→ 单卡（FIFO + 反解 chunk）。注意"不做前缀亲和路由"是**因为**存储集中化（#41），两者是同一决策的两面。
- 疑点：四层里只有第四层是"每步"决策，前三层是"每轮/每请求"决策，频率不同，书里要标清楚。

#### #17 `cost(q, c)` 的输入、输出与求解（QA 版）
- 证据：`doc/inference_deployment_qa.md:115`
  > Each prefill step is given a time budget (`compute_quota_limit`, in nanoseconds), and a fitted quadratic cost model `cost(q, c)` predicts how long the step will take (`q` = number of sequence tokens, `c` = the prefix length already computed for that sequence). The scheduler solves that quadratic for `q`, taking the largest `q` that stays within the 32,768-token budget while exactly consuming the remaining time budget. The budget itself is further reduced by an all-reduce MIN within the DP group.
- 等级：B
- 意味着什么：模型是**每序列**的二次代价函数，输入 `(q, c)`，输出"这一步要花多久"（时间）；用途不是排序而是**反解 chunk 大小**：在时间预算内取最大 `q`，同时被 32,768 token 硬上限截断。因为 DP 组内同步，预算还要先做一次 all-reduce MIN——最穷的 rank 决定全组。
- 疑点：① 二次式的系数（拟合参数）未公布；② "solves that quadratic for q" 只说解析求根，未给判别式/取整规则；③ 单序列模型的输出如何合成整步代价（求和？取 max？）未说明——见 §4 问题 3。

#### #18 `q` 的定义在两份官方文本里冲突
- 证据：
  - `doc/inference_deployment_qa.md:115`：`q` = number of sequence tokens
  - 知乎 §Prefill Load Balancer（快照 142 行）：`q` = 序列中**未命中**的 token 数，`c` = 这条序列已经算完的前缀长度
- 等级：B（两处都是官方口径）
- 意味着什么：如果 `q` 是"序列总 token 数"，则 `c`（已算前缀）是它的子集，二者信息重复；如果 `q` 是"未命中 token 数"，`cost(q,c)` 才是一个真正的二维模型（要算的量 + 已有上下文）。**从建模合理性看知乎版更自洽**，但这是我的推断。
- 疑点：两条都无更早出处，无法判定哪个是笔误。书里建议：写"q 为待计算 token 数（知乎口径：未命中 token；QA 口径：序列 token 数）"并注明冲突。

#### #19 分桶与"每档独立 compute-quota 上限"（QA 独有）
- 证据：`doc/inference_deployment_qa.md:109`（`each tier has its own per-batch compute-quota cap`）
- 等级：B
- 意味着什么：长上下文桶（≥256K）不会被短请求桶的调度预算拖累，各档有自己的每批配额上限——这是"分桶"真正的意义，而不是简单按长度分组。
- 疑点：知乎对应段落（快照 144 行）只写"每档分到的引擎台数按该档上一轮的 compute-quota 需求占比动态重分"，**没有** per-batch cap；两处信息量不同。

#### #20 Decode LB：统一路由所有 decode 卡，两个维度，启发式
- 证据：`doc/inference_deployment_qa.md:117-119`
  > The Decode Load Balancer routes uniformly across all GPUs of all decode instances. The principle is to balance two dimensions at once: request count, which determines the per-step batch size, and total KV cache length, which determines the cost of the attention part. Heuristically, it sends short requests to GPUs with low concurrency but large total KV cache length, and long requests to GPUs with small total KV cache length. See DualPath's intra-group scheduling algorithm for details.
- 等级：B
- 意味着什么：Decode 侧的负载有两个不可互换的维度：**请求数**（决定 batch size / MoE 效率）与**KV 总长**（决定 attention 代价）。启发式是把二者当互补资源配对：短请求去"卡多但 KV 长"的卡，长请求去"KV 短"的卡。这是全书最能体现"decode 调度是二维装箱"的一段。
- 疑点：没有给出权重或阈值，只有方向性描述；"uniformly across all GPUs" 说明 decode LB 是全局视角，不是分组视角（与 Prefill 的组间/组内分层不同）。

#### #21 VE LB：机器粒度随机 + 机内最少 patch
- 证据：`doc/inference_deployment_qa.md:121-123`
  > Images are routed at machine granularity: a prefill machine picks at random one machine that hosts one vision encoder per GPU, and on that machine the image is handed to the GPU currently processing the fewest image patches — computed directly from the dimensions as `(width/patch_size) × (height/patch_size)`.
- 等级：B
- 意味着什么：VE 的负载单位是 **patch 数**而不是图片数，且路由是"随机选机 + 机内最小 patch"两级。这是"图片大小差异极大"场景下的必要设计。
- 疑点：随机选机不做机器级负载判断（原文 "picks at random"），意味着机器间不均衡靠实例数而非路由解决。

#### #22 EPLB 机制：冗余专家 + 统计窗 + 定期重排 + symm_mem 搬家
- 证据：`doc/inference_deployment_qa.md:95-97`
  > beyond the experts it is responsible for, each GPU holds a few redundant experts that host replicas of hot experts. The load of each expert over the last N steps is collected, and every M steps experts are re-assigned to GPUs based on those statistics; expert weights are shuffled within the group over NVLink `symm_mem`.
- 等级：B
- 意味着什么：EPLB 的三要素是"冗余副本、滑动窗口统计、周期重排"，搬家走 NVLink `symm_mem`（不是 RDMA），所以重排只发生在 scale-up 域内。
- 疑点：`symm_mem` 在代码里的实现见 R08；本笔记不重复。

#### #23 EPLB 的 Prefill 参数与 `moe_lb` 数字（QA 与知乎冲突）
- 证据：
  - `doc/inference_deployment_qa.md:99`：4 redundant experts/GPU 起步；100 步统计窗；每 1,000 步重排；`moe_lb` = **1.074 (EP4DP4) / 1.045 (EP2DP2)**
  - 知乎（快照 164 行）：每卡 4 个冗余专家起步；统计窗 100 步；每 1000 步重排；`moe_lb` = **1.0320（EP2/DP2）/ 1.0618（EP4/DP4）**
- 等级：B（两处都是官方）
- 意味着什么：参数完全一致，**四个数字没有一个对得上**，而且 EP2/EP4 的大小关系在两份文本里相反（QA: EP2 更好；知乎: EP4 更好）。
- 疑点：无法判定哪份是最终数据（Q&A 自述会持续更新）。书里若要写这张表，必须二选一并注明来源，不能混用。

#### #24 EPLB 的 Decode 参数与 `moe_lb`
- 证据：`doc/inference_deployment_qa.md:103`
  > start from 1 redundant expert per GPU … — at EP64, for example, this works out to 7 experts per GPU after rounding, excluding the shared expert; a 1,000-step statistics window and a re-assignment every 10,000 steps. Measured `moe_lb` = 1.090.
- 等级：B
- 意味着什么：Decode 的 EPLB 更"慢而稳"：统计窗 1,000 步、重排 10,000 步，是 Prefill 的 10 倍；因为 decode 每步都在跑，重排本身会打断服务。EP64 时每卡 7 个专家（含 1 冗余后取整）是可直接引用的算例。
- 疑点：`moe_lb = 1.090` 没给 EP/DP 配置，只说了"at EP64"作为冗余专家算例，二者是否同一配置不明。

#### #25 负载均衡的总体收益：达到完美均衡离线基准的约 80%
- 证据：`doc/inference_deployment_qa.md:125-127`
  > It can be compared against the offline benchmark: we reach roughly 80% of a perfectly balanced offline benchmark.
- 等级：B
- 意味着什么：这是"四个 LB 加起来值多少"的唯一官方数字：80%（相对完美均衡）。它同时说明**还有 20% 的失衡损失留在线上**，书里不要写成"解决了负载均衡问题"。
- 疑点：`offline benchmark` 的具体配置未给；80% 是吞吐还是有效吞吐未明说。

### D. KVCache 存储、Entrypoint 与 GDR

#### #26 三行存储表：谁在哪、几副本、谁管
- 证据：`doc/inference_deployment_qa.md:131-135`
  > | In-flight block KV cache + state KV cache | Inference GPU memory | — | Prefill / decode instances |
  > | SWA KV cache | Local host memory | Single copy; nothing on 3FS | Prefill / decode instances; cross-machine hits pull over a 400G VLAN |
  > | Paged block KV cache | Dedicated 3FS storage cluster | No application-level replication; relies on 3FS's own redundancy | kvcache-store |
- 等级：B
- 意味着什么：全章最重要的一张表。三个要点：① block KV 不在本机内存，直接进 3FS；② SWA KV 只有单副本、**不在 3FS**、跨机靠 400G VLAN 拉；③ 应用层不做副本，冗余完全交给 3FS。
- 疑点：知乎同表（快照 186-190 行）逐行一致，可互证。

#### #27 为什么用独立 3FS 集群而不是本机/分层存储
- 证据：`doc/inference_deployment_qa.md:141`
  > a dedicated storage cluster lets every prefill machine hit any prefix, avoiding a conflict between prefix affinity and the load-balancing routing policy. It also decouples autoscaling of storage nodes from compute nodes and keeps the KV cache unaffected by compute-node failures (compute nodes fail at a rate more than an order of magnitude higher than storage nodes).
- 等级：B
- 意味着什么：这句话是"四级负载均衡"的设计前提：**因为任何 prefill 都能命中任何前缀，所以 Prefill LB 才敢完全不做前缀亲和路由**（#16）。同时给出一个可引用的故障率比例："计算节点故障率比存储高一个数量级以上"。
- 疑点：`an order of magnitude higher` 无具体数字与统计窗口。

#### #28 命中路径三段，只有第一段在关键路径上
- 证据：`doc/inference_deployment_qa.md:157`
  > **Matching** is done by the Entrypoint against the local kvcache-store in the CHECK role, as a hash-chain prefix match over metadata only; but the hit length has to be computed before routing, so it cannot be hidden. **Fetching KV** happens asynchronously in a separate process, moving the KV cache from 3FS into host shared memory. **Swap-in** from shared memory to HBM runs in a thread inside the engine process …
- 等级：B
- 意味着什么：三段式（匹配/取 KV/换入）里，**匹配必然暴露在 TTFT 上**，因为路由要用命中长度；取 KV 与换入都能被掩盖。书里讲"KV 命中不是免费的"就靠这句。同时它明确了 `Entrypoint` 与 `kvcache-store` 的存在与角色。
- 疑点：`CHECK role` 只出现这一次，`kvcache-store` 还有哪些角色（写？GC？）未公开——见 §4 问题 4。

#### #29 Block KVCache 的块粒度与内容（知乎独有）
- 证据：知乎 §Block KVCache（快照 182 行）
  > 以 128 tokens 为一个 block 进行分页式管理，根据模型架构设计，存储 4 层的 main KV 和 indexer K。
- 等级：B
- 意味着什么：一个 block = 128 token 的**4 层** main KV + indexer K。这个 "4 层"与"112 KiB/block"（#12）一起才能算出每 token 的字节数，是书里做容量算术的入口。
- 疑点：QA 只说 `main KV + indexer K`（`doc/inference_deployment_qa.md:141`），**没有** "4 层"；"4 层"只此一处。为什么是 4 层（CED 架构的分层存储？）需要 R02b/R06 交叉验证。

#### #30 SWA KV 的存放结构：StateBlock 两区
- 证据：`doc/inference_deployment_qa.md:163`
  > it lives in the state cache, where each sequence occupies one fixed-size slot (`StateBlock`). Each slot is split into two regions by whether it needs to be stored: `[stored region: SWA] [dropped region: YOCO consumer SWA]`. … For V4.1, the stored layers are the first 20.
- 等级：B
- 意味着什么：SWA 不走 paged block，而是每序列一个固定槽；槽内按"要不要落盘"切成两区，dropped 区放 YOCO consumer 层与 `ca_state`/`ica_state`（压缩窗口边界天然对齐 128，所以无跨边界依赖）。V4.1 只存前 20 层。
- 疑点：`StateBlock` 在代码里的对应见 R06（本笔记不重复）。

#### #31 SWA 落盘位置规则：只在 128 的整数倍
- 证据：`doc/inference_deployment_qa.md:165`
  > The SWA KV cache is only stored at positions that are multiples of 128 tokens. After the prefill stage: the store position is obtained by reserving one token at the end of the sequence and rounding down to a multiple of 128 … Nothing is stored part-way through a chunked prefill — only the last chunk stores. During the decode stage, a separate replica region is allocated in device memory; every time the request crosses a 128-token boundary, its current window is copied into the replica region on-device. … The copy position can overshoot the boundary by at most 5 tokens, which is absorbed by the 6 extra positions the slot's ring buffer has beyond the window.
- 等级：B
- 意味着什么：三条易错的细节：① prefill 结束要**先留 1 个 token** 再向下对齐（因为最后一个 token 的 KV 还没算？）；② chunked prefill 中途不存，只有最后一个 chunk 存；③ decode 侧在显存里另开**副本区**，每跨 128 边界整窗拷一次，最多越过 5 个 token，靠 ring 多出的 6 个位置吸收（128 窗口 + 6 回退 = 134）。
- 疑点：`reserving one token` 的原因原文未解释；"6 extra positions" 与 `134 = 128 + 6` 自洽，可交叉验证 #32。

#### #32 SWA 的体量、格式与"留在产出的机器上"
- 证据：`doc/inference_deployment_qa.md:167-169`
  > Only `state_size_to_disk` = 20 layers × 134 tokens × 528 B = 1,415,040 B per sequence goes into the store. The 528 B/token is the MXFP8 SWA KV format. … An in-memory directory — logically unified, but physically partitioned by key across the Entrypoint machines — records which machine holds it …
- 等级：B
- 意味着什么：单序列 SWA 落盘量 = 1,415,040 B ≈ 1.35 MiB（计算值：20×134×528）。`Entrypoint` 在这里第二次出现：它维护一个**逻辑统一、物理按 key 分片**的内存目录。这就是"入口服务"这个名字背后的真实职责之一。
- 疑点：`logically unified, physically partitioned by key` 的 key 是什么（prefix hash？session id？）未说明；目录的一致性与失效协议未公开。

#### #33 SWA 的 P→D 传输与跨机命中
- 证据：`doc/inference_deployment_qa.md:171-177`
  > The SWA KV cache and the block KV cache share the same P-to-D transfer path.
  > Cross-machine hits rely on the Entrypoint in-memory directory together with a 400G VLAN pull-back.
- 等级：B
- 意味着什么：SWA 与 block KV **共用同一条 P→D 传输通道**（说明 P/D 之间有一条统一的 KV 传输面，不是两套）；跨机命中走 400G VLAN，而不是 3FS。
- 疑点：`same P-to-D transfer path` 具体是 RDMA 还是别的，原文未说；这也正是开源代码里缺失的那部分（#64）。

#### #34 SWA 的线上 TTL 与命中率（知乎独有）
- 证据：知乎 §SWA KVCache（快照 203 行）
  > 由于 SWA KV 只在会话内的分钟级窗口里被复用 …… 线上数据显示，其 TTL 约 180 秒，命中率在 98% 以上。
- 等级：B
- 意味着什么：`180s` 与 `98%` 解释了"为什么 SWA 不值得落盘"：分钟级生命期的数据，用 SSD 承载不划算（#35 的论证）。这是全书少见的"用线上统计支撑架构取舍"的例子。
- 疑点：无统计窗口、无流量类型；`TTL 约 180 秒` 是实测驻留时间还是配置值不明。

#### #35 命中策略：L−134 上限，block 定长度，SWA 定是否退一块
- 证据：`doc/inference_deployment_qa.md:189-191`
  > the maximum hit length for an input of length L is at most L − 134 (128-token window + 6 tokens of speculative backtracking). … the block KV cache is queried first, giving a hit length L1 (a multiple of the 128-token block size), and then the SWA KV cache is queried. If an SWA KV cache exists in the range L1 − 128 to L1, the hit length is L1; otherwise it is L1 − 128, where 128 is the recomputation window.
- 等级：B
- 意味着什么：命中的两段判定可以写成伪代码，是书里最值得画的决策树：`L1 = floor128(block_hit)`；`hit = SWA_hit_in(L1-128, L1) ? L1 : L1-128`。`134 = 128 + 6` 与 #31 的 ring 尺寸同源。
- 疑点：`L − 134` 的上限来自 YOCO 截断要求，和"投机回退 6"绑在一起；如果关闭投机解码（draft=0），这个 134 是否变成 128，原文没说。

#### #36 重算只花常数代价：SWA Bounded Replay 的两个用法
- 证据：`doc/inference_deployment_qa.md:181-187`
  > SWA bounded replay recomputes only the most recent 128 tokens and truncates the attention window … the query at position i is computed only against keys in the range max(s, i−127) to i … The cost of recomputation is thereby reduced to a constant, independent of context length and layer count.
  > … Decoder SWA bounded replay … reconstructing it exactly would require running the last 2,560 tokens through 20 decoder layers. Instead, each prefill recomputes only the last 128 tokens … nearly halving total prefill computation.
- 等级：B
- 意味着什么：`2560 → 128` 是 20 倍的重算削减；"总 prefill 计算量近乎减半"是这条机制的收益口径。机制细节归 R07，本笔记只保留与存储/命中耦合的部分。
- 疑点：`nearly halving` 的基线是"精确重建"而不是 V4 线上版本，书里引用要写清对比对象。

#### #37 decode 不做 KV offload，但 decode 产生的 KV 仍落盘
- 证据：`doc/inference_deployment_qa.md:147-149`
  > No. Decode is not bound by cache capacity, so Hi-Sparse is not used either. The KV cache produced during decode is still written to disk, in order to improve the cache hit rate.
- 等级：B
- 意味着什么：两个否定 + 一个肯定：不 offload、不用 Hi-Sparse，但**落盘**。这条把"落盘"和"offload"两件事分开了：落盘是为了下次命中，不是为了腾显存。落盘由 decode 实例做（知乎 §Block KVCache：「Block KVCache 由 decode 实例落盘存储，使得 prefill 和 decode 输出均被复用」）。
- 疑点：既然 decode 落盘，写带宽（#9 的 5 GB/s/台）就是 decode 侧的关键资源，文档没给写入量估算。

#### #38 prefill 也用 paged KV cache
- 证据：`doc/inference_deployment_qa.md:143-145`
  > - Does prefill use a paged KV cache? - Yes, the prefill side also uses a paged KV cache.
- 等级：B
- 意味着什么：block 粒度（128 token）在 P 和 D 两侧一致，否则 P→D 传输与 3FS 存取都要做重分页。这是"统一块格式"的最小证据。
- 疑点：无。

#### #39 GDR 换入：用 RDMA 网卡打满 PCIe，不占 SM
- 证据：`doc/inference_deployment_qa.md:157`（后半句）
  > Because the CUDA copy engine performs poorly on the fine-grained copies of KV cache blocks, swap-in is implemented with GDR: an RDMA NIC saturates PCIe bandwidth without consuming SMs.
  知乎同段（快照 199 行）：
  > **换入** 从 shm 到 HBM 在 engine 进程内的线程里做，目标是实现逐层 H2D 传输与前向计算重叠，由于 CUDA copy engine 对于 KVCache Block 细粒度的拷贝性能不佳，换入使用 GDR 实现，用 RDMA 网卡就能打满 PCIe 带宽，并且不占 SM。
- 等级：B
- 意味着什么：`GDR` 最合理的展开是 **GPUDirect RDMA**：数据从宿主 shm 直接进 HBM，由 RDMA 网卡发起，绕过 copy engine、也不占 SM。关键点不是"更快"，而是"**能按层细粒度地与计算重叠**"——换入线程每层搬一点，和前向计算并行。
- 疑点：`GDR` 这个缩写在两份文档里都没展开成全称，"GPUDirect RDMA" 是我的解释（E，见 #40 的判据）；它也可能是 DeepSeek 内部对某条 zero-copy H2D 路径的统称。

#### #40 "细粒度块拷贝不适合 CUDA copy engine"的定量依据
- 证据：DualPath（SIGCOMM '26，arXiv:2602.21548v2）§5.2 "CNIC-Assisted KV-Cache Copy"
  > We also observe that CNIC-assisted H2D and D2H outperform the CUDA copy engine when handling a large number of small data chunks. Our measurements show that submitting a single copy operation via `cudaMemcpyAsync` incurs a latency overhead of approximately 5-7 μs … In contrast, submitting one RDMA Write work request involves only a few mmio writes to NIC registers in user space and takes only around 1 μs. Furthermore, the RDMA work submission overhead can be significantly amortized by leveraging doorbell batching.
  §A.5：
  > Layerwise prefill reduces KV-Cache block size to 1/layer of the original, and makes the number of blocks larger to layer×, posing challenges to transfer and storage performance.
- 等级：B（同公司作者群的同行评审论文，实验全部用 3FS 做存储后端，见 arXiv §A.4）
- 意味着什么：可以算出"为什么细粒度不划算"：一个 112 KiB 的 block 在 40 GB/s 下单次搬运只需 2.87 μs（计算值：112×1024/40e9），而 copy engine 每次提交要 5–7 μs——**提交开销是搬运时间的约 2 倍**；RDMA Write 提交约 1 μs，还能靠 doorbell batching 摊销。层流水（layerwise）又把块切成 1/层数、块数 ×层数，进一步放大提交次数。这解释了 DeepInfer 为什么在 swap-in 上放弃 copy engine 改用 GDR。
- 疑点：① 5–7 μs 是 DualPath 在其测试平台（未标明 GPU 型号）上测的，不是 DeepInfer 线上；② 112 KiB 与 2.87 μs 的除法用的是 #9 的单机 40 GB/s 读带宽，真实单 block 带宽可能更低（多请求竞争），所以真实占比只会更差；③ 论文给 CNIC 路径的首要想法是**QoS 隔离**（KV 流量不得干扰 EP AllToAll 等延迟敏感集合通信），"提交开销"是第二理由；DeepInfer 文档只说 "performs poorly"，把两者合成因果链是我做的（E）；④ DualPath 的 CNIC = GPU 配对网卡，与 DeepInfer 的 "RDMA NIC" 是否同一硬件角色，是我按上下文对应（E）。

#### #41 命中长度计算为什么不能被掩盖
- 证据：`doc/inference_deployment_qa.md:157` + 知乎同段
  > the hit length has to be computed before routing, so it cannot be hidden
- 等级：B
- 意味着什么：这一句把"Entrypoint 前缀匹配"钉在关键路径上：路由依赖命中长度，命中长度依赖元数据哈希链匹配。因此匹配被设计成"只查元数据"（不读 KV 数据），这是它唯一能做的优化。
- 疑点：哈希链的 hash 函数、块大小（是否也是 128 token 一块）、元数据规模都未公开。

### E. 弹性与容错

#### #42 自动扩缩容：P 看未命中 token，D 看在算请求数，限速 + 上下界
- 证据：`doc/inference_deployment_qa.md:195-197`
  > Prefill determines the number of machines it needs from the number of non-cached tokens; decode adjusts based on the number of running requests. Adjustments are rate-limited and bounded above and below.
- 等级：B
- 意味着什么：扩缩容信号选得很讲究：P 的负载本质是"要算多少未命中 token"（命中越多越闲），D 的负载本质是"同时有多少请求在 decode"（与 KV 长度弱相关）。这解释了为什么 P/D 比例会随流量结构变化（#4 的 1:1 只是常态）。
- 疑点：`rate-limited` 的具体速率、上下界数值未给。

#### #43 缩容宽限期 1 分钟，未完成请求连同 KV 一次性转发
- 证据：`doc/inference_deployment_qa.md:199`
  > a decode engine (CE) enters a grace period (1 min) after receiving a shutdown notice; any requests still unfinished, along with their KV cache, are forwarded in one shot to other CEs to continue decoding.
- 等级：B
- 意味着什么：`CE` = completion engine，与代码里的 `EngineType.COMPLETION` 同名同义。缩容不是"等请求跑完"，而是"1 分钟后整体搬家"——这要求 KV 能被完整转移到另一个实例，即 P/D 之外还存在 **D→D 的 KV 迁移**能力。
- 疑点：D→D 迁移的实现文档没讲，开源代码里也没有（#78）；`in one shot` 是否包含 SWA KV 不明。

#### #44 冗余配额：ready 机器 + 利用率 0.8 以上禁止缩容
- 证据：`doc/inference_deployment_qa.md:201-203`
  > ready machines held unoccupied in the pool as scale-out inventory, and always keeping more instances online than the computed demand — concretely, scale-in is forbidden whenever quota utilization is above 0.8.
- 等级：B
- 意味着什么：容量规划的两个旋钮：库存机（cold standby）与 0.8 缩容门限（相当于常留 20% 余量）。这是"扩缩容"之外的第二层保护，书里容易漏。
- 疑点：`quota utilization` 的分子分母（卡数？显存？实例数？）未定义。

#### #45 重启分钟级；容错靠配额冗余 + 请求级重路由
- 证据：`doc/inference_deployment_qa.md:205-207`
  > Restarts are on the order of minutes. Fault tolerance comes from PE (prefill engine) / CE quota redundancy plus request-level rerouting, rather than from shortening the restart time of a single engine.
- 等级：B
- 意味着什么：这是全书对"弹性容错"最直白的一句设计哲学：**不优化恢复时间，优化"故障时还有余量"**。`PE` / `CE` 两个缩写再次确认引擎的 P/D 二分。
- 疑点：`request-level rerouting` 的粒度（整请求重跑 prefill？还是从断点续？）未说。

#### #46 EP failover：一卡故障整实例不可用，重启按健康卡数重定 EP
- 证据：`doc/inference_deployment_qa.md:209-211`（接 #45）
  > In a large-EP deployment, all GPUs of one instance form a single communication domain, so any single GPU failure makes the whole instance unavailable. … the platform re-determines the EP size from all currently healthy GPUs, and the instance runs with the number of GPUs it actually got.
- 等级：B
- 意味着什么：Wide-EP 的代价在这里：通信域 = 实例，故障域 = 实例。而恢复方式很"弹性"——**用剩多少卡就跑多大 EP**，不追求恢复到原配置。这是"弹性容错"四个字最具体的含义。
- 疑点：重定 EP size 后模型权重分片要重排（EPLB 重排？），文档没说这步耗时；这会直接影响"分钟级重启"是否包括重分片。

### F. 其它 Q&A（低精度、内核、投机、异步、结构化输出）

#### #47 FP4 的三处用法，三种粒度
- 证据：`doc/inference_deployment_qa.md:215-221`
  > - Main KV: per-16, e4m3 — like NVFP4 but without the global scale.
  > - Indexer KV: per-32, ue8m0, MXFP4.
  > - MoE routed-expert weights: per-32, ue8m0, MXFP4.
- 等级：B
- 意味着什么：同一套系统里三种量化粒度并存：KV 用 per-16 e4m3（更细），indexer KV 与 MoE 权重用 per-32 ue8m0（MXFP4）。书里讲"低精度"必须分开说，不能一句"FP4"盖过。
- 疑点：知乎全文 grep `MXFP4` / `e4m3` 均为 0 次（本次重抓的 537 行快照），这组数字**只有 QA 有**；R01 亦已记录该差异。

#### #48 内核：FlashMLA 少一个手调 SASS，decode kernel 慢 5%–10%，端到端最多 5%
- 证据：`doc/inference_deployment_qa.md:233-237`
  > They are broadly the same; the indexer and mega series differ somewhat. FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.
  > End-to-end performance is close — at most 5% apart.
- 等级：B
- 意味着什么：`5%–10%` 是 **kernel 级**数字，端到端被摊薄到 `at most 5%`（同段后文）。书里引用必须带"kernel 级 vs 端到端"的限定，否则会把开源版说慢一倍。
- 疑点：哪个 SASS 优化、为何不公开，原文未说。

#### #49 Mega MoE：scale-up 域内的通用融合，任何 EP/batch 都显著快于 unfused
- 证据：`doc/inference_deployment_qa.md:229-231`
  > Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE.
- 等级：B
- 意味着什么：这是"内部版内核比开源版强"的第二个例子（第一个是 #48），但**没有给数字**，只说 "significantly"。书里不能编比例。
- 疑点：`Mega MoE` 在知乎出现 0 次；开源仓库里是否有同名实现需 R10 核实，本笔记不下结论。

#### #50 DSpark 变长验证的完整数字
- 证据：`doc/inference_deployment_qa.md:245-256`
  > Dynamic-length verification cuts the number of draft tokens verified per step by about 30% (5.00 → 3.2–4.4) while leaving the average accepted length essentially unchanged (3.29 vs. 3.20). Computed from EP64/DP64 benchmark data, that works out to +4–7% throughput at the production operating point (~60 concurrent sequences per rank).
  表（`:249-256`，列：conc / fixed-5 平均验证长度 / 动态平均验证长度 / fixed-5 接受长度 / 动态接受长度 / fixed-5 估计吞吐 / 动态估计吞吐 / 增益）：
  | conc | 动态验证长度 | 接受长度 fixed/dyn | 吞吐 fixed/dyn (tok/s/rank) | 增益 |
  |---:|---:|---|---|---:|
  | 16 | 4.44 | 3.32 / 3.24 | 4891 / 4816 | −1.5% |
  | 32 | 3.65 | 3.25 / 3.17 | 8964 / 9111 | +1.6% |
  | 48 | 3.94 | 3.30 / 3.26 | 12032 / 12467 | +3.6% |
  | 60 | 3.58 | 3.30 / 3.21 | 13730 / 14466 | +5.4% |
  | 72 | 3.24 | 3.29 / 3.12 | 14962 / 15961 | +6.7% |
  | 96 | 3.24 | 3.30 / 3.15 | 16817 / 18660 | +11.0% |
- 等级：B
- 意味着什么：变长验证的收益不是"接受更多 token"（3.29 vs 3.20 只多 0.09），而是"**少验证约 30% 的 token**"；且收益随并发上升（16 并发时是 −1.5%），说明它只在生产负载点（~60）以上才划算。这是反直觉但关键的一点：省的是验证算力，不是提高接受率。
- 疑点：知乎同一处只给结论"吞吐提升 6%"（快照 231 行），是表格在 60 并发处 +5.4% 的口头化；书里用表，不要用 6%。

#### #51 异步调度：decode 第 N 步的空闲点与重叠方案
- 证据：`doc/inference_deployment_qa.md:268-274`
  > Decode step N runs `schedule(N) → forward(N) → sample(N) → post-process(N)`, where schedule and post-process are pure CPU logic, and a GPU-CPU synchronization is needed before `post-process(N)` … As a result, the GPU sits idle during `post-process(N)` and `schedule(N+1)`.
  > Following existing implementations (SGLang, NanoFlow) … while the GPU runs `forward(N) → sample(N)`, the CPU concurrently runs `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`.
- 等级：B
- 意味着什么：ZOS 的动机一句话说清：GPU 在等 CPU 做后处理与下一步调度。重叠方案是让 CPU 在 GPU 跑 forward(N)+sample(N) 时做 schedule(N+1)/launch/post-process(N-1)。注意它明确点名参考 **SGLang 与 NanoFlow**。
- 疑点：细节归 R05，本笔记只保留服务章节需要的动机句。

#### #52 结构化输出：编译与推理解耦，专用 SO 机器
- 证据：`doc/inference_deployment_qa.md:282-288` 与知乎 §结构化输出（快照 243-250 行）
  > Compilation and inference are decoupled: SO compilation runs asynchronously, and a request retrieves the compiled result before it enters the engine.
  > Compilation stage (on a dedicated SO machine): the format constraint is compiled into a grammar automaton.
  > Runtime: at each decode step, the automaton is advanced, the set of legal tokens is determined, and a mask is generated and applied to the sampling logic.
- 等级：B
- 意味着什么："独立 SO 机器"是架构上的一个额外角色（不在架构图里），说明语法编译是 CPU 重活且被池化了。请求进 engine 前必须已经拿到编译结果——这是又一处"入口侧同步点"（与 #41 的命中长度并列）。
- 疑点：SO 机器的池化/缓存策略、编译时延都未公开；架构图未画该角色。

### G. deepseek-recipe

#### #53 定位：库，不是服务
- 证据：`README.md:9-15`
  > deepseek-recipe is a collection of Rust libraries and Python bindings that uniformly convert API requests in different formats into the Conversation format … The caller must provide model inference, tool execution, and HTTP transport.
- 等级：B
- 意味着什么：recipe 不含 HTTP 服务、不含推理、不含工具执行——它只做"协议 ↔ token 序列"的编解码。书里不能把它写成"API 网关"。
- 疑点：无。

#### #54 支持的三套协议与能力面
- 证据：`README.md:23-26`
  > **Request/response formats:** Conversion of Messages, Chat Completions, and Responses requests, Streaming response, and complete responses. Supports text, images, thinking mode, client tool calls, and Chat Completions `prefix` continuation.
  目录结构印证：`deepseek-recipe/src/protocol/{anthropic/messages, openai/chat_completion, openai/responses}`
- 等级：B（README）+ A（目录）
- 意味着什么：三套协议 = Anthropic Messages、OpenAI Chat Completions、OpenAI Responses；四类内容 = 文本、图片、思考、客户端工具调用；外加 Chat Completions 的 `prefix` 续写。
- 疑点：`Messages` 是 Anthropic 协议名（Anthropic 官方叫 Messages API），README 未写 "Anthropic" 字样，靠目录 `protocol/anthropic/messages` 确认。

#### #55 实现语言：Rust 库 + PyO3 绑定（cdylib `_native`，abi3-py310）
- 证据：`deepseek-recipe-python/Cargo.toml:9-11,23`
  > crate-type = ["cdylib"]
  > name = "_native"
  > pyo3 = { version = "0.27", features = ["abi3-py310"] }
  `Cargo.toml:15`：`version = "0.2.0"`；`deepseek-recipe-python/src/lib.rs:48` 起为 `#[pymodule] fn _native`
- 等级：A
- 意味着什么：核心逻辑全在 Rust（`deepseek-recipe`、`-core`、`-encoding`、`-image`、`-macros` 五个 crate），Python 只是薄绑定。`abi3-py310` 意味着一个 wheel 覆盖 3.10+。
- 疑点：`edition = "2024"`（python crate 的 Cargo.toml）说明工具链很新；`rust-toolchain.toml` 锁了版本。

#### #56 防特殊 token 注入（输入侧）：编码时按"显式插入区间"走 added vocabulary，其余走基础词表
- 证据：`deepseek-recipe-core/src/prompt.rs:54-61`
  > pub fn push_special(&mut self, token: &str) { … let start = self.text.len(); self.text.push_str(token); self.special_ranges.push(start..self.text.len()); }
  `deepseek-recipe-encoding/src/tokenizer.rs:49-56,63-70`
  > let mut plain_tokenizer = tokenizer.clone(); plain_tokenizer.with_added_vocabulary(AddedVocabulary::new());
- 等级：A
- 意味着什么：机制是"**记录 + 分离**"：渲染时只有模板插入的 special token 会进 `special_ranges`；编码时这些区间直接映射成 added vocabulary 的 ID，其余文本用一个**剥掉 added vocabulary 的 tokenizer** 编码。因此用户写 `<｜User｜>` 只会被切成普通字符 token。
- 疑点：无（有测试，见 #57）。

#### #57 防注入的测试证据：用户文本里的 `<｜User｜>` 必须编码成普通 token 序列
- 证据：`deepseek-recipe-encoding/src/v4/mod.rs:587-641`（测试 `special_ranges_cover_explicit_special_tokens_only`）
  > // `<｜User｜>` starts each of the two user messages. `</think>` ends the reasoning content … 
  > assert_eq!(ids.iter().filter(|&&id| id == user_id).count(), 2);
  > assert!(contains(&plain_ids(&format!("x<｜User｜>{IMAGE}"))));  // 普通文本里的 <｜User｜> 按基础词表编码
  > assert!(contains(&plain_ids("</think>")));
- 等级：A
- 意味着什么：这是"注入防护"的可执行规格：整个对话里 `<｜User｜>` 只出现 2 次（两次消息边界），用户内容里的那一个必须能被基础词表的 token 序列匹配到。`</think>` 同理。
- 疑点：无。

#### #58 防注入（输出侧）：按 token ID 识别，不按文本拼写
- 证据：`deepseek-recipe/src/stream/decoder.rs:23-33,40-45`
  > - With `skip_special_tokens` set to `false`, `decode_ids` must keep special tokens in the text, because output parsing recognizes them.
  > - `is_special_token` must return `true` for every token ID of the added vocabulary, so that the stream processor decodes each special token as its own piece and output parsing matches a marker only against a token, not against text that spells the same.
  `deepseek-recipe/src/stream/output_parser.rs:83-94`（`ChunkKind::{Text, PlainTokens, SpecialToken}` 及其匹配规则注释）
- 等级：A
- 意味着什么：输出侧是对称的：模型若用普通 token 拼出 `<｜end▁of▁thinking｜>`，那只是内容，不能触发"思考结束"；只有真正的 special token ID 才算标记。`ChunkKind` 就是这条规则的载体：Text 按拼写匹配，PlainTokens 只匹配 plain 段，SpecialToken 才匹配 special 段。
- 疑点：`Text`（无 token 边界的纯文本输入）按拼写匹配，是给"上游只给文本"的退化路径用的——这条路径上注入防护不成立，文档没提醒，属潜在坑。

#### #59 分段并行编码：切段 + 一次 batch + 按序拼接
- 证据：`deepseek-recipe-encoding/src/tokenizer.rs:90-118`
  > let mut regions = Vec::with_capacity(special_ranges.len() + 1); // 每个 special 区间之前的普通段 + 最后一段
  > let encodings = self.plain_tokenizer.encode_batch_fast(regions, false) …;
  > for range in special_ranges { ids.extend_from_slice(encoding.get_ids()); ids.push(self.special_token_id(&text[range.clone()])?); }
  `docs/tokenizer.md:70-72`
  > `PromptTokenizer` encodes the plain regions as one batch, which the `tokenizers` crate can encode in parallel, depending on its `TOKENIZERS_PARALLELISM` setting.
- 等级：A（代码）+ B（官方文档）
- 意味着什么：并行不是自己开线程，而是把 N+1 个普通段作为**一个 batch** 交给 `tokenizers` 的 `encode_batch_fast`，由该 crate 内部的 rayon 池并行；special 段按 ID 直接塞回原位。因此"并行度"受 `TOKENIZERS_PARALLELISM` 控制，且总结果与逐段顺序编码一致（拼接顺序固定）。
- 疑点：`encode_batch_fast(regions, false)` 的第二个参数是 `add_special_tokens=false`，这点保证了不会给每段偷偷加 BOS。

#### #60 分段编码为什么成立（官方论证与我的补全）
- 证据：知乎 §Tokenizer 处理与加速（快照 126 行）
  > 构建 prompt 时，已经能够确定 special token 的位置和边界。根据 BPE tokenizer 的编码原理，可以发现 special token 之间的各段普通文本可独立编码。因此，我们在编码时首先跳过普通文本当中 special token 的匹配与切分，然后并行地编码各段普通文本，最后再按原顺序拼接结果。
- 等级：B（官方说法）+ E（机制补全）
- 意味着什么：官方只给了一句"根据 BPE 原理"。补全的推理是：special token 是 added vocabulary 里的**原子 token**，不参与普通文本的 BPE merge；把 added vocabulary 剥掉之后，普通段内部的 merge 完全由段内字符决定，而段边界处是一个"非普通词表"的原子，跨边界的 merge 不可能发生。因此"分段编码 + 顺序拼接"与"整段编码"等价。
- 疑点：这个等价性依赖一个未明说的前提——**跨边界的两个普通字符不会恰好构成一个 merge pair**。对当前 DeepSeek 词表成立（因为边界是 special token，不是普通字符），但如果有人把 special token 从 prompt 里删掉再编码，前提就破了。文档没有把这条前提写出来。

#### #61 流式解析状态机：11 个 Stage + 逐字节 KMP + 保留未完成进度
- 证据：`deepseek-recipe/src/stream/output_parser.rs:235-252`（`enum Stage`：`Content`、`Json`、`AfterJson`、`ToolCalls`、`ToolName`、`ToolCallArguments`、`ToolCallArgumentName`、`ToolCallArgumentStringAttribute`、`ToolCallArgumentValue`、`Reasoning`、`Finished`，共 11 个变体）
  `output_parser.rs:603-660`（`fn feed`：逐 byte 推进，`buffered_len` 记录已匹配长度，未匹配部分按 `action_on_unmatched` 切出）
  `output_parser.rs:704-764`（`kmp_table` / `kmp_next`）
- 等级：A
- 意味着什么：状态机是"**阶段 × 匹配分支**"的二维结构：每个 Stage 有自己的标记集合（`</think>`、`<｜DSML｜invoke name="` 等），每个标记用 KMP 增量匹配；`buffered_len` 保留"chunk 末尾只匹配了一半"的进度，下一个 chunk 接着喂，不回溯已输出文本。
- 疑点：`Stage::Finished` 之后的内容按 `InvalidTokenKind::ContentAfterFinished` 丢弃——这是一条"停止后仍收到 token"的容错策略，书里可作为一个细节。

#### #62 跨 chunk 的 UTF-8 处理：用 U+FFFD 判定并缓冲
- 证据：`deepseek-recipe/src/stream/decoder.rs:9-16`
  > Replacement character that decoding produces while a multi-token character is incomplete. Decoded text that ends with it stays buffered, so a token that decodes to U+FFFD itself is held until a later token ID completes the text, and is dropped at the end of input.
- 等级：A
- 意味着什么：一个汉字/emoji 可能跨 2–4 个 token，解码时末尾会出现 U+FFFD；实现不去猜字节，而是"**看到 U+FFFD 就缓冲 token ID，等下一个 token 补齐**"。这比"按字节重组"简单，但代价是"真的输出 U+FFFD 的 token 会被延迟到流结束"。
- 疑点：这是一个刻意接受的权衡（文档写明 "is dropped at the end of input"），书里可当作"工程妥协"的例子。

#### #63 工具参数的流式 JSON 生成
- 证据：知乎 §流式输出解析（快照 134 行）
  > 工具名解析完成后即可返回，后续根据 DSML 参数边界直接生成 JSON 片段。字符串值逐段做 JSON 转义，数组、对象等非字符串值按原始 JSON 文本输出，无需构造完整参数对象 … 即使参数较长，也可以在生成过程中逐段返回。
- 等级：B
- 意味着什么：不构造完整参数对象，直接拼 JSON 片段——这是"首 token 时延"在工具调用场景的具体优化点。代码侧对应 `StreamProcessor` 的 `ToolCallArgumentsDelta`。
- 疑点：非法 JSON（模型生成到一半断掉）时如何收尾，文档未说；代码里 `FinishReason::ToolCalls` 只在 `had_tool_calls_at_finish` 且正常 Stop 时给出（`processor.rs:206-212`）。

#### #64 图片组件：base64/URL，OpenCV 预处理，默认不限制目标地址
- 证据：`README.md:47-54`
  > Provided as base64 or external URLs, and preprocessed for DeepSeek V4.1 with OpenCV … The component's `ReqwestImageFetcher` connects to any address by default, so a service that fetches URLs from untrusted requests should install a `DestinationFilter`.
- 等级：B
- 意味着什么：这是一个明确的安全提示：默认配置下 URL 抓取可打内网（SSRF），生产必须装 `DestinationFilter`。书里讲"开源组件的边界"时值得引一句。
- 疑点：V4 不收图，转换时把图片替换成 `[Unsupported Image]` 文本——这条对多版本共存的服务有影响。

#### #65 其它编解码事实：max_tokens 默认值、模型版本解析、prefix 续写
- 证据：`README.md:27-38`
  > `resolve_max_tokens` … the `max_tokens` default when the request sets none: 8192 outside thinking mode, 65536 in thinking mode, and 131072 in thinking mode with `max` effort.
- 等级：B
- 意味着什么：三档默认值对应三种"思考预算"，是服务端行为的一部分；模型版本由 `model` 名解析（默认 `DeepSeek-V4-Pro`、`DeepSeek-V4.1-Flash`，可用 alias 注册），说明 recipe 是"按模型版本选模板"的设计。
- 疑点：`V4` 与 `V4.1` 的模板差异（system token、图片、`</think>`）散落在 `encoding/src/v4/{dsv4,dsv41}.rs`，细节归 R11/其它笔记。

### H. 开源代码里 P/D 分离与网络存储的痕迹（A 级）

#### #66 `EngineType` 三态：NORMAL / PROMPT / COMPLETION
- 证据：`deepinfer/engine/engine_type.py:11-13`
  > NORMAL = 0  # process prompts & completions
  > PROMPT = enum.auto()  # only process prompts
  > COMPLETION = enum.auto()  # only process completions
  `csrc/engine/scheduler.hpp:39-43` 同构（`NORMAL=0, PROMPT=1, COMPLETION=2`，注释 `only process prompts` / `only process completions`）
- 等级：A
- 意味着什么：引擎类型是**编译期之外的运行期一等参数**，Python 与 C++ 两侧各有一份且要求保持一致（`engine_type.py:8` 明确写"请和 csrc/engine/scheduler.hpp 中的 enum class EngineType 保持一致"）。P/D 分离在代码里首先是"同一份引擎代码的两种裁剪模式"。
- 疑点：无。

#### #67 `PROMPT` 引擎默认关闭零开销调度
- 证据：`deepinfer/engine/engine_type.py:15-17`
  > @property
  > def default_zero_overhead_scheduling(self) -> bool:
  >     return self in (EngineType.NORMAL, EngineType.COMPLETION)
  `deepinfer/engine/arg_utils.py:51-57`（据此设 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`）
- 等级：A
- 意味着什么：P 引擎默认不开 ZOS。与文档一致：prefill 的调度靠 `compute_quota_limit` 反解 chunk（#17），decode 才靠异步流水（#51）。这是"两种引擎调度目标不同"的代码级证据。
- 疑点：是否可手动开启（环境变量可覆盖）代码没禁止，说明只是默认值不是硬约束。

#### #68 `init_pd_split()` 只在非 NORMAL 引擎上调用
- 证据：`deepinfer/engine/llm_engine.py:453-455`
  > self._run_workers("init_cache_engine")
  > if self.engine_type != EngineType.NORMAL:
  >     self.init_pd_split()
- 等级：A
- 意味着什么：初始化路径里专门有一个"P/D 分离初始化"钩子，只有 P 或 D 引擎才走。这是"分离模式需要额外数据结构"的设计意图最直接的痕迹。
- 疑点：`init_pd_split` 的函数体是空的（#69），所以这条调用目前等于 no-op。

#### #69 P/D 分离的钩子方法：7 个无效实现 + 2 个带默认实现的钩子
- 证据：`deepinfer/engine/llm_engine.py:800-825`
  > # 这里开始都是虚函数
  > def init_pd_split(self) -> None: """虚函数，初始化 PD 分离需要的数据结构"""
  > def finalize_seq(self, seq): """虚函数，处理结束的请求""" self.scheduler.free_seq(seq)
  > def post_schedule(self, scheduler_outputs): """虚函数，处理每个 step scheduler 的结果"""
  > def should_copy_state(self, seq) -> bool: """虚函数，判断是否要保存 seq 当前状态""" return False
  > def do_save_kvcache(self, scheduled, finished): """虚函数，保存 kvcache, pre_sampler_sync 时调用"""
  > def pe_prepare_save_kvcache(...): """虚函数，pe 准备保存 kvcache"""
  > def pe_do_save_kvcache(...): """虚函数，pe 保存 kvcache"""
  > def process_output(self, output): """虚函数，处理输出一个 step 输出的 token …"""
- 等级：A
- 意味着什么：P/D 分离 + KV 落盘的接口面就是这 8 个方法（外加 `pre_sampler_sync` 这个调用者）。其中**7 个无实际实现**：`init_pd_split`、`post_schedule`、`do_save_kvcache`、`pe_prepare_save_kvcache`、`pe_do_save_kvcache`、`process_output` 为空，`should_copy_state` 恒 `False`；`finalize_seq` 有默认实现（`self.scheduler.free_seq(seq)`）。内部实现必然是 `LLMEngine` 的子类覆盖它们——这就是 README 说"大部分引擎功能已实现"的确切含义。`pe_` 前缀直接对应 prefill engine。
- 疑点：仓库内**没有任何子类覆盖**这些方法（grep 只有基类定义），所以开源版的行为是：不落盘、不传 KV、不做 P/D 交接。`pre_sampler_sync` 的调用分支见 #80。

#### #70 `send_processed_prompts()`：P 引擎每步结束后把处理完的 prompt 交给下游
- 证据：`deepinfer/engine/llm_engine.py:848-850`
  > def send_processed_prompts(self) -> None:
  > """把已经处理完的 prompt sequence 发送给 completion engine"""
  （函数体为空）
  `llm_engine.py:1009-1010`
  > if self.engine_type == EngineType.PROMPT:
  >     self.send_processed_prompts()
- 等级：A
- 意味着什么：这是 P→D 交接的**唯一命名明确的接口**：P 引擎在每步 `step()` 末尾调用它，语义就是"把算完的 prompt 发给 completion engine"。注释里 `completion engine` 与文档的 `CE`（#43）完全对应。
- 疑点：函数体为空，且没有任何调用方检查返回值——说明开源版这一步是 no-op，P 引擎算完的 KV 不会被送出。

#### #71 C++ 调度器里的移交注释：`prefill_engine … will be transferred to complete_engine`
- 证据：`csrc/engine/scheduler.hpp:941-943`
  > if (engine_type == EngineType::NORMAL) { /// prefill_engine won't change status
  >                                     /// because it will be transferred to
  >                                     /// complete_engine
- 等级：A
- 意味着什么：代码注释直接写明了 P→D 的状态语义：**prefill 引擎不改请求状态**，因为请求会被移交给 complete engine。这是"P/D 分离不是两套引擎，而是一个请求在两段流水上接力"的最硬证据，等级 A、且是英文注释（不是后加的翻译）。
- 疑点：`NORMAL` 引擎里才需要这段判断（因为 PROMPT 引擎根本不会走到这个状态迁移），注释与分支条件的对应关系略绕。

#### #72 COMPLETION 引擎没有 prefill 通道，必须配 `stop_preempted_request=true`
- 证据：`csrc/engine/scheduler.hpp:101-104`
  > // COMPLETION engine 没有 prefill 调度通道，无法重算被抢占的请求。启动时拦截，禁止该配置组合
  > TORCH_CHECK(engine_type != EngineType::COMPLETION || stop_preempted_request,
  >     "COMPLETION engine cannot recompute preempted requests (no prefill channel); "
  >     "it must be deployed with stop_preempted_request=true (--stop-preempted-request)");
- 等级：A
- 意味着什么：D 引擎被抢占时**不能重算**（重算需要 prefill 通道），只能直接终止请求。这是一个由 P/D 分离导出的硬约束，也解释了为什么 `EngineArgs.stop_preempted_request` 默认是 `True`（`arg_utils.py:39`）。书里讲"P/D 分离的代价"可以引这条。
- 疑点：被终止的请求由谁重试（Entrypoint 重路由？），代码里没有。

#### #73 `copy_state` 缓冲区 + `should_copy_state()` 恒 False：decode 侧状态复制的半成品
- 证据：`deepinfer/engine/llm_engine.py:284-285`
  > self.copy_state: torch.Tensor | None = None
  > self.copy_state_event: torch.cuda.Event | None = None
  `llm_engine.py:787-798`
  > """把给定 seq 当前的 GPU state 拷贝到 copy_state buffer，并更新 last_copy_state_token_idx。该方法仅下发任务到 GPU，不等待完成…"""
  > self.copy_state[state_id].copy_(self.driver_worker.cache_engine.gpu_states[state_id])
  > seq.last_copy_state_token_idx = seq.num_cached_tokens
  `llm_engine.py:930-937`
  > if not scheduler_outputs.prompt_run and self.copy_state is not None:  # 处理 kvcache 落盘相关
  >     to_copy_state_seqs = [seq for seq in … if self.should_copy_state(seq)]
- 等级：A
- 意味着什么：这段代码与知乎 §SWA KVCache 的描述**逐条对应**：decode 步（`not prompt_run`）里、按条件（`should_copy_state`）、把当前 state 拷到一块**独立的显存副本区**（`copy_state`）、并记录拷到了哪个 token 位置（`last_copy_state_token_idx`）。这正是"每跨一个 128 token 边界就把窗口拷入副本区，请求结束时从最近一次完成拷贝的位置存储"的机制骨架。
- 疑点：`should_copy_state` 基类恒返回 `False` → 开源版永不拷贝；`copy_state` 也从未被分配（只有 `= None`）。所以这是**半成品**：机制在，触发条件与存储端在内部子类里。

#### #74 `num_cached_tokens` 与 `state_id` 暴露给 Python：前缀"已算好"的接缝
- 证据：`csrc/engine/pybind_sequence.cpp:126,131,134`
  > "num_cached_tokens", [](const Sequence &seq) { return seq.cache.num_tokens; }
  > bind_sequence_property(sequence, "state_id", … seq.cache.state_id)
  > bind_sequence_property(sequence, "last_copy_state_token_idx", …)
- 等级：A
- 意味着什么：引擎的 `Sequence` 本来就带"已缓存多少 token"（`num_cached_tokens`）与"状态槽号"（`state_id`）两个字段。任何"从外部塞入一段已算好的前缀"的方案（KV 存储命中、P→D 传输）都只需要在构造 `Sequence` 时把 `num_tokens` 设成正数——**接缝在数据结构层面已经存在**，缺的只是搬运实现。
- 疑点：`num_cached_tokens` 只读暴露，外部无法直接改写（需构造时传入）；这条推断（E）需要读 `Sequence` 构造函数确认，本笔记未做。

#### #75 `engine_group_rank`：同一作业内多个独立 engine world 的端口隔离
- 证据：`deepinfer/utils.py:223-230`
  > # Ranks in one world share an offset; independent worlds get distinct
  > # ports via engine_group_rank.
  > return f"tcp://{ip}:{int(port) + parallel_config.engine_group_rank}"
  `deepinfer/engine/arg_utils.py:32-33`（`engine_rank`、`engine_group_rank` 两个入参）
- 等级：A
- 意味着什么：启动器支持"一个作业里起多个互不通信的 engine world"，各自用不同端口。P 组与 D 组正是这种关系（不共享 NCCL world，只通过 KV 传输面交互）。这是"分离部署需要多 world"的代码级准备。
- 疑点：`engine_group_rank` 也可能只是给 DP 副本用的；"P 组/D 组"是我的解释（E）。

#### #76 没有 Entrypoint / HTTP 服务
- 证据：`deepinfer/entrypoints/` 只有 `__init__.py`（0 字节）与 `llm.py`（8.6K，`LLM` 类，离线推理入口）；全仓 grep `FastAPI|uvicorn|aiohttp` 只命中 `engine/profile_server.py` 与 `sampling_params.py`，前者自述为
  > Engine 运行时 profiling 的 HTTP 接口，每个 engine 进程一个。POST /profile 采 num_steps 个 step 并落盘 … 默认关，设 `DEEPINFER_PROFILE_HTTP_ENABLE=1` 开启。
- 等级：A（负证据）
- 意味着什么：文档里的 `Entrypoint`（#28、#32）**在开源仓库里不存在对应实现**；仓库里唯一的 HTTP 服务是 perf profiling 端点（每 engine 一个、默认关闭），与请求路由无关。README 的缺件清单（#8）也没列 Entrypoint——因为它属于"KVCache 管理 + 负载均衡"那一类未开源组件。书里绝不能把 `deepinfer/entrypoints/llm.py` 当成文档里的 Entrypoint。
- 疑点：`profile_server.py` 是不是内部 Entrypoint 的一小块，无法确认（从注释看不是）。

#### #77 没有任何 cost model / compute_quota 代码
- 证据：全仓 `grep -rn "compute_quota\|cost_model"`（排除 vendor）**0 命中**；`32768` 只出现在 examples/eval 的参数默认值与 TileLang 内核的 benchmark 形状里，engine 内无 32,768 预算逻辑
- 等级：A（负证据）
- 意味着什么：Prefill LB 的二次代价模型（#17）与 32,768 token 预算（#17）**完全不在开源代码里**。开源调度器只有 `max_num_batched_tokens` 这个静态上限（`EngineArgs.max_num_batched_tokens`，`arg_utils.py:34`），没有"按时间预算反解 q"。
- 疑点：`max_num_batched_tokens` 默认 `None`（由配置推导），与文档的 32,768 是否同一个量，需读 `SchedulerConfig` 确认，本笔记未做。

#### #78 没有 3FS 客户端、KV 传输或 connector 代码
- 证据：全仓 grep `kv_transfer|kv_connector|nixl|mooncake|3fs|remote_kv|save_kvcache|load_kvcache`（排除 vendor）只命中 `doc/inference_deployment_qa.md` 的正文与 `pe_do_save_kvcache` 系列空函数名
- 等级：A（负证据）
- 意味着什么：3FS 存取、P→D RDMA 传输、D→D 迁移（#43）三件事在开源版**一行实现都没有**。`save_kvcache` 只以空方法名存在（#69）。这与 vLLM/SGLang 提供 `KVConnector` 抽象的做法形成对比（对比细节归 R13/R14）。
- 疑点：vendor 目录里的 DeepEP 提供的是 EP 通信（all-to-all），不是 KV 传输；不能拿它当 P/D 传输实现。

#### #79 调度器支持"整组同型"决策，为分离部署留了投票机制
- 证据：`csrc/engine/scheduler.hpp:295-300`
  > case EngineType::NORMAL: {
  > // NORMAL 引擎：决定这一步全组做 prompt 还是 decode。dp 内必须同型，靠投票统一。
- 等级：A
- 意味着什么：NORMAL 引擎（单机混跑）内部要保证 DP 组内"这一步大家都做 prefill 或都做 decode"，靠投票统一。分离部署把这个问题从"投票"变成"部署时决定"，本质是用资源冗余换掉同步开销——这是书里讲"为什么要 P/D 分离"的一个好角度。
- 疑点：投票的具体规则（多数？全体？）在后续行，本笔记未展开；细节归 R05。

#### #80 `pre_sampler_sync`：P 步只"准备"，D 步才真正落盘
- 证据：`deepinfer/engine/llm_engine.py:826-841`
  > def pre_sampler_sync(self, scheduler_outputs): 
  >     self.deferred_release.release()
  >     # save kvcache
  >     if not scheduler_outputs.prompt_run:
  >         if self.copy_state_event is not None: self.copy_state_event.synchronize()
  >         self.do_save_kvcache(...)
  >     else:
  >         self.pe_prepare_save_kvcache(...)
- 等级：A
- 意味着什么：这是落盘路径的**唯一调用点**，而且分支方向与直觉相反：**decode 步（`not prompt_run`）调 `do_save_kvcache` 真正保存，prefill 步只调 `pe_prepare_save_kvcache` 做准备**，P 分支里没有任何"保存"调用。这与知乎"Block KVCache 由 decode 实例落盘存储，使得 prefill 和 decode 输出均被复用"（#37）方向一致。
- 疑点：① `copy_state_event.synchronize()` 只在 decode 分支调用，说明 D 侧落盘前要等 state 拷贝完成；P 分支没有对应的等待，`pe_prepare_save_kvcache` 内部是否自己同步，代码里看不到。② "P 侧只准备、由 D 侧统一落盘"是我从"P 分支没有 save 调用"推出来的（E），另一种可能是 P 的 save 在 `send_processed_prompts` 里随交接一起做（#70），但那条路径也是空的。

## 3. 必答问题逐条回答

### 问题 1：把 Q&A 文档的每一个问答整理成事实条目

已按主题覆盖全部 45 个问答，映射如下（Q&A 行号 → 事实条目）：

| Q&A 行 | 问题 | 事实条目 |
|---|---|---|
| 41-43 | 整体形态 | #1 |
| 45-53 | 并行配置 | #2 |
| 55-59 | Wide-EP 收益 | #3 |
| 61-63 | 多模态部署 | #1、#21 |
| 65-67 | 最小部署单元 | #4 |
| 71-73 | 跨数据中心 | #5 |
| 75-77 | 存储服务器配置 | #9 |
| 79-81 | 存储/GPU 配比 | #10 |
| 83-91 | 3FS 特性 | #11、#12 |
| 95-99 | EPLB prefill | #22、#23 |
| 101-103 | EPLB decode | #22、#24 |
| 107-109 | Prefill LB 分桶 | #16、#19 |
| 111 | DP 组间 | #16 |
| 113 | DP 组内 | #16 |
| 115 | 单卡内 + cost(q,c) | #17、#18 |
| 117-119 | Decode LB | #20；另见 §3 问题 4 |
| 121-123 | VE LB | #21 |
| 125-127 | LB 总收益 | #25 |
| 131-135 | 存储三行表 | #26 |
| 139-141 | block KV 介质与理由 | #27、#29 |
| 143-145 | prefill 是否 paged | #38 |
| 147-149 | decode 是否 offload | #37 |
| 151-153 | TTL | #13 |
| 155-157 | 缓存开销如何掩盖 | #28、#39、#40、#41 |
| 161-163 | SWA 如何存 | #30 |
| 165 | SWA 落盘位置 | #31 |
| 167-169 | 存什么、存哪 | #32 |
| 171-173 | P→D 传输 | #33 |
| 175-177 | 跨机命中 | #33 |
| 181-187 | 重算 | #36 |
| 189-191 | 命中策略 | #35 |
| 195-199 | 自动扩缩容 | #42、#43 |
| 201-203 | 冗余配额 | #44 |
| 205-207 | 重启时间 | #45 |
| 209-211 | EP failover | #46 |
| 215-221 | 低精度 | #47 |
| 229-231 | Mega MoE | #49 |
| 233-237 | 内核库 | #48 |
| 245-256 | 投机解码 | #50 |
| 268-274 | 异步调度 | #51 |
| 282-288 | 结构化输出 | #52 |
| 292-296 | recipe 多协议 | #53、#54 |
| 298-304 | tokenize 加速 | #56、#59、#60 |
| 306-312 | 流式解析 | #61、#62、#63 |

（说明：Q&A 第 61-63 行的"多模态部署"与第 215 行后的低精度/内核/投机/异步/SO 各节，本笔记只保留服务章节需要的口径与数字；机制细节已在 R05/R07/R09/R10/R11 中展开，不重复。）

### 问题 2：知乎额外描述的架构（入口服务、kvcache-store、四个 LB、3FS）

- **入口服务 = Entrypoint**，文中不是一个独立小节，而是通过三处职责拼出来的：① 在**本机 CHECK 角色的 kvcache-store** 上做哈希链前缀匹配，只查元数据（#28）；② 维护 SWA KV 的**逻辑统一、物理按 key 分片**的内存目录（#32、#33）；③ 架构图显示它在 `DeepSeek Recipe` 之后、三个路由 LB 之前，且画了 3 个方框（#15）。文档未给 Entrypoint 的副本数、一致性协议与 QPS 能力，见 §4 问题 5。
- **kvcache-store** 只在两处出现：存储表里作为"分页 Block KVCache 的管理方"（#26），以及命中路径里作为 Entrypoint 查询的"CHECK 角色"实例（#28）。它是一个**管理 3FS 上 block KV 元数据的组件**，不是存储介质本身；其他角色（写、GC 协调）未公开，见 §4 问题 4。
- **四个 LB**：Prefill / Decode / VE 三个是**路由型**（在 Entrypoint 之后、Service 之前），EPLB 是**服务内型**（画在 Prefill Service 与 Decode Service 内部）。知乎对四者的分工有一句总述（#14），架构图给出拓扑（#15）。各级策略见 #16（Prefill 四层）、#20（Decode 二维）、#21（VE patch 数）、#22–#24（EPLB 冗余专家）。
- **3FS**：`约 100 PiB SSD`、`block 粒度约 112 KiB`、`峰值聚合读 10 TiB/s`、`峰值聚合写约 850 GiB/s`（#12）；`10+2 EC`、读 11 取 10、GC Worker 按 LRU + 水位线（#11）；`TTL agent 3 天 / chatbot 7 天`，3 天 TTL 让 chatbot 命中率掉 0.5%–1%（#13）。另有容量口径 `14.4 TB/GPU（逻辑）`（#10）与单机 `20×30.73 TB`（#9）。

### 问题 3：Prefill 的二次代价模型 `cost(q, c)`

- **输入**：`q` 与 `c`。`c` = 该序列**已经算完的前缀长度**（两份文档一致）。`q` 的定义两份文档冲突：QA 说"序列 token 数"，知乎说"序列中未命中的 token 数"（#18）。
- **输出**：预测"这一步要花多久"的时间（配合 `compute_quota_limit`，单位纳秒），即把 `(q,c)` 映射为代价，用于**反解**而不是排序（#17）。
- **求解方式**：给定本步剩余时间预算，对二次式 `cost(q,c) = T` 解出 `q`，取"正好用完预算的最大 `q`"，并受 `32,768` token 上限截断；预算在 DP 组内先做一次 `all-reduce MIN`，取最紧的那个（#17）。二次式系数由离线拟合得到，未公布；单序列代价如何合成整步代价（求和/取 max）未说明（§4 问题 3）。
- **为什么开源代码没有这个组件**：官方理由是"KVCache 管理与负载均衡组件与内部基础设施耦合较深，尚未完整开源"（#7）。代码层面可验证：全仓 `compute_quota` / `cost_model` 0 命中，engine 内没有 32,768 预算逻辑（#77）；P/D 分离与落盘只留了 7 个空虚函数（#69）。我补充的推断（E）：这个模型的系数是**从生产流量 profile 出来的**，脱离 DeepSeek 的机型、模型版本与流量分布就没有意义，因此不适合随代码发布。

### 问题 4：Decode 负载均衡引用的 "DualPath" 是什么

- 出处：知乎参考文献 [1]（快照 254 行）：
  > Y. Wu, S. Chen, R. Huang, Y. Tan, Y. Zhong, M. Zhang, X. Jin, and P. Huang. **DualPath: Accelerating Agentic LLM Inference by Harvesting Disaggregated KV-Cache Storage I/O.** In Proceedings of the ACM SIGCOMM 2026 Conference, SIGCOMM '26, Denver, CO, USA, 2026. ACM. ISBN 979-8-4007-2467-1.
- 论文本题（arXiv:2602.21548v2，标题为 *DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference*）：在传统 `storage→prefill` 路径之外，增加一条 `storage→decode→(RDMA)→prefill` 的第二条加载路径，把空闲的 decode 侧存储网卡带宽也用起来，并用一个中心 Request Scheduler 动态分配两条路径的流量；报告 offline 吞吐最高 1.87×、online 平均 1.96×（#40 引用了它的 5–7 µs 提交开销测量）。
- Decode LB 借的是它的**组内调度**：论文 §6.1 把调度分两级——inter-engine（跨引擎，按 token 数均衡 + 读队列阈值 α + 未完成 token 上限 β 分三类）与 intra-engine/within-group（按组内剩余 HBM 估算可调度集合 R，再用 `Z = 1.05 ×` 组内均值作高 token 阈值，防止长请求独占）。知乎那句"具体可参考 DualPath 的组内调度算法"对应的就是后者（#20）。
- 等级：B（论文原文可查）+ E（"知乎指的正是 §6.1 Phase 2"是我的对应，论文没有逐字写"intra-group scheduling algorithm"这个短语）。
- 疑点：论文的组内算法以 **token 数**为代理指标，而知乎的 Decode LB 强调"请求数 + KV 总长"两个维度；二者不完全等价，见 §4 问题 8。

### 问题 5：GDR 换入是什么，为什么细粒度块拷贝不适合 CUDA copy engine

- **GDR 换入** = 从宿主共享内存（shm）把 KV 直接搬进 HBM 的换入路径，由 RDMA 网卡发起（GPUDirect RDMA），而不是用 CUDA 的 `cudaMemcpy`/copy engine。文档给的两条特征是"能打满 PCIe 带宽"和"不占 SM"（#39）。它跑在 engine 进程内的一个线程里，目的是**逐层 H2D 与前向计算重叠**（#28）。
- **为什么不用 copy engine**：文档只说 "performs poorly on the fine-grained copies of KV cache blocks"（#39）。可量化的原因是 DualPath 测到的：`cudaMemcpyAsync` 单次**提交**要 5–7 µs，而 RDMA Write 的 WR 提交约 1 µs、还能靠 doorbell batching 摊销；同时 layerwise prefill 把 block 切成 `1/层数`、块数 `×层数`（#40）。代入 DeepInfer 的数：112 KiB block 在 40 GB/s 下搬运本身只要 2.87 µs（计算值），提交 5–7 µs 就是搬运时间的 2 倍左右——**开销在提交，不在带宽**。这就是"细粒度"的要害：块越小、次数越多，copy engine 的固定提交开销占比越高。
- 等级：B（两条都是官方/论文原文）+ E（把 112 KiB、40 GB/s、5–7 µs 三者相除得到"2 倍"的算术与因果链是我做的，论文没有替 DeepInfer 做这个计算）。
- 疑点：`GDR` 全称两份文档都没展开；5–7 µs 是 DualPath 测试平台的数据，不是 DeepInfer 线上实测。

### 问题 6：deepseek-recipe 的协议、防注入、分段并行、流式状态机、实现语言

- **协议**：`Messages`（Anthropic）、`Chat Completions`、`Responses`（后两个 OpenAI）；支持流式与完整响应、文本/图片/思考/客户端工具调用、Chat Completions 的 `prefix` 续写（#54）。它只是库，不含 HTTP 与推理（#53）。
- **防注入**：输入侧"记录 + 分离"——渲染时只把模板插入的 special token 记进 `special_ranges`，编码时这些区间走 added vocabulary 的 ID，其余文本用**剥掉 added vocabulary 的 tokenizer** 编码（#56），因此用户写的 `<｜User｜>` 只能变成普通字符 token（有测试，`v4/mod.rs:587-641`，#57）。输出侧按 **token ID** 识别 `</think>` 与 `｜DSML｜`，模型用普通 token 拼出的同样文本不算标记（`ChunkKind` 区分 Text/PlainTokens/SpecialToken，#58）。
- **分段并行编码为什么成立**：官方说法是"special token 之间的各段普通文本可独立编码"（知乎 §Tokenizer 处理与加速）；实现是把 special 区间之间的 N+1 个普通段作为**一个 batch** 交给 `encode_batch_fast`（由 `tokenizers` crate 内部并行，受 `TOKENIZERS_PARALLELISM` 控制），special 段按 ID 直接插回（#59）。机制上的前提是 special token 为原子 added token、不参与普通文本 BPE merge，因此跨边界不会产生新的 merge（#60，E）。官方没写出这条前提，是我补的推断。
- **流式状态机**：`Stage` 共 11 个变体（Content/Json/AfterJson/ToolCalls/ToolName/ToolCallArguments/ToolCallArgumentName/ToolCallArgumentStringAttribute/ToolCallArgumentValue/Reasoning/Finished），逐 **byte** feed；每个阶段用 KMP 增量匹配标记，`buffered_len` 保留"chunk 末尾只匹配一半"的进度，下一 chunk 接着喂，不回溯已输出文本；跨 chunk 的不完整 UTF-8 用 U+FFFD 判定并缓冲 token ID（#61、#62）。工具参数不构造完整对象，按 DSML 边界直接产出 JSON 片段（#63）。
- **实现语言**：Rust（5 个 crate）+ PyO3 绑定（`cdylib`，模块名 `_native`，`abi3-py310`，pyo3 0.27，版本 0.2.0）（#55）。

### 问题 7：开源仓库里 P/D 分离与网络存储引擎的预留接口与代码痕迹

- **接口形状**：`EngineType` 三态（#66）、`init_pd_split()` 只在非 NORMAL 引擎调用（#68）、7 个无实现的钩子 `init_pd_split` / `post_schedule` / `should_copy_state`(恒 False) / `do_save_kvcache` / `pe_prepare_save_kvcache` / `pe_do_save_kvcache` / `process_output`（#69）、`send_processed_prompts()` 空实现且 P 引擎每步调用（#70）、`pre_sampler_sync` 是落盘路径唯一调用点（P 准备 / D 保存）（#80）。
- **语义痕迹**：C++ 注释 `prefill_engine won't change status … transferred to complete_engine`（#71）；`COMPLETION engine cannot recompute preempted requests (no prefill channel)` 的启动拦截（#72）；调度器里 NORMAL 引擎"DP 组内必须同型、靠投票统一"（#79）。
- **状态复制机制**：`copy_state` 显存副本区 + `copy_state_event` + `do_copy_state()` + `last_copy_state_token_idx`，挂在 decode 步（`not prompt_run`）上，与知乎"每跨 128 token 边界拷一次"逐条对应；但 `should_copy_state()` 基类恒 `False`、`copy_state` 从未分配 → **半成品**（#73）。
- **数据结构接缝**：`num_cached_tokens`（= `seq.cache.num_tokens`）与 `state_id` 已暴露，说明"前缀已算好"可以由外部构造（#74）；`engine_group_rank` 支持同作业多 engine world 的端口隔离（#75）。
- **明确的空白**：没有 Entrypoint/HTTP 服务（#76）、没有 cost model/compute_quota（#77）、没有 3FS 客户端/ KV 传输 / connector（#78）。README 的缺件清单（#8）与此一致。
- **结论**：开源版能做到的 P/D 相关能力 = "同一份引擎可以被配置成只做 prefill 或只做 decode，并按各自目标调度"；做不到的是把 KV 从 P 搬到 D、写进 3FS、从 3FS 读回。书里必须把这两层分开写。

## 4. 新发现的问题

1. **同一份文档里"24 小时"与"3 天/7 天"口径冲突**：`doc/inference_deployment_qa.md:77` 说存储服务器容量按"retain 24 hours of data"配，`:153` 说 TTL 是 agent 3 天、chatbot 7 天。是"容量规划按 24h 算、TTL 只是失效上界"，还是两处不同时期的口径？需要向官方确认或在书里并列。
2. **`q` 的定义冲突**（#18）：QA 的"序列 token 数"会让 `c`（已算前缀）成为冗余信息，建模上不自洽；知乎的"未命中 token 数"更合理。书里必须选一个并注明。
3. **`cost(q,c)` 是单序列模型，但调度决策是整步的**：文档没说多个序列的代价如何合成（求和？取 max？按 chunk 分配后取最大？）。缺这一步，读者无法复现"反解 q"。
4. **`kvcache-store` 的角色没有枚举**：只出现 "CHECK role"（`:157`）。是否有 WRITE / GC 协调 / 目录同步角色？block KV 由 decode 落盘（知乎 §Block KVCache），那写入路径经过哪个角色？完全空白。
5. **Entrypoint 的一致性与容量**：SWA 目录"逻辑统一、物理按 key 分片"（`:169`），但 key 是什么、目录如何更新/失效、3 个副本是否强一致，全未公开。若书里要画"入口服务"，只能画职责不能画协议。
6. **路由是 push 还是 pull**：QA 说"engines pull work according to their remaining compute quota"（`:111`），暗示**引擎主动拉**；但同一段又说 Entrypoint 在路由前算命中长度（`:157`），暗示**入口推**。两者如何衔接（Entrypoint 算完长度后放进队列、引擎来拉？）没有说明。这是理解 Prefill LB 的关键，建议列入 open questions。
7. **开源版 decode 实际不会落盘**：`should_copy_state()` 恒 `False`、`copy_state` 从未分配（#73），所以开源版跑完一个请求，KV 既不落盘也不传给任何下游。README 的"KV cache system 未包含"（#8）在代码上表现为"所有落盘钩子都是空函数"。书里若要演示"KV 复用"，必须明确说开源版做不到。
8. **DualPath 的组内算法与知乎描述不完全对应**：论文以 token 数为代理（跨组按 token 数、组内按 HBM 可调度集合 + 1.05× 均值阈值），知乎强调"请求数 + KV 总长"两维。知乎可能只是"参考其思路"，不是"采用其算法"。书里引用要写"参考"，不要写"采用"。
9. **DualPath 与 DeepInfer 的关系未公开**：论文作者含 DeepSeek 人员、实验全部用 3FS 做后端（arXiv §A.4），且测了 layerwise prefill 的 block 布局（§A.5）。它与 DeepInfer 生产系统是"同一套代码的不同切片"还是"平行原型"，两处文档都没说。这会影响"能否把论文数据当 DeepInfer 数据"的判断——结论是**不能**。
10. **recipe 分段编码的等价性前提未文档化**（#60）：官方只有一句"根据 BPE 原理"。如果书里要讲"为什么可以分段并行"，需要自己补出"跨边界无 merge"这一前提，并标明是推断。
11. **`Text` 类型的输出解析路径会退化为按拼写匹配**（#58 疑点）：当上游只给文本、不给 token ID 时，`</think>` 的注入防护不成立。文档只提了 token 路径的防护，没提这个退化路径的风险。

## 5. 可用于图表的材料

### 5.1 架构分层图（可直接照 `architecture_overview.png` 重画）
```
User Request (JSON)
   │
DeepSeek Recipe            ← 协议 → Conversation → prompt → token IDs
   │  Token List + Sampling Parameters
Entrypoint ×3              ← 前缀匹配 / SWA 目录分片 / 路由
   ├── Prefill Load Balancer ──→ Prefill Service  [内含 EPLB]
   ├── Decode  Load Balancer ──→ Decode  Service  [内含 EPLB]
   └── VE      Load Balancer ──→ Vision Encoder Service
Prefill Service ──(KVCache 蓝箭头)──→ Decode Service
Prefill/Decode Service ←──双向──→ External KVCache Storage (3FS)
图例：黑箭头 = Request，蓝箭头 = KVCache
```

### 5.2 Prefill LB 的四层决策表
| 层级 | 决策对象 | 输入信号 | 决策 | 频率 |
|---|---|---|---|---|
| 桶级 | 4 个长度档（<16K / 16–64K / 64–256K / ≥256K） | 上一轮该档 compute-quota 需求占比 | 给每档分多少引擎；每档独立 per-batch quota 上限 | 每轮 |
| DP 组间 | 引擎 | 剩余 compute-quota | 引擎按剩余配额**拉取**；不做前缀亲和 | 每次拉取 |
| DP 组内 | DP rank | 当前 quota 负载 | 新请求 → 负载最低的 rank | 每请求 |
| 单卡内 | chunk 大小 | `compute_quota_limit`（ns）+ `cost(q,c)` | 对二次式反解 q，取预算内最大 q，≤32,768 | 每步 |

### 5.3 `cost(q,c)` 反解伪代码（据 #17 复原，系数未知）
```
budget_ns = all_reduce_min(compute_quota_limit, group=DP)   # 组内取最紧预算
q = max{ q_int | cost(q, c) <= budget_ns }                   # 对二次式求根后取整
q = min(q, 32768)                                            # 硬上限
chunk = q                                                    # 本步实际 prefill 的 token 数
```
（注：文档没有说明"已算过部分"的代价是否要先从 budget 里扣除，也没有说明多条序列的代价如何合成整步代价——见 §4 问题 3。上图只保留文档明确写出的三步。）

### 5.4 KV 命中三段与"能否掩盖"
| 段 | 执行者 | 动作 | 在关键路径？ |
|---|---|---|---|
| 匹配 | Entrypoint（查本机 `kvcache-store`，CHECK 角色） | 元数据哈希链前缀匹配，得命中长度 | **是**（路由前必须知道） |
| 取 KV | 独立进程 | 3FS → 宿主 shm，异步 | 否 |
| 换入 | engine 进程内线程 | shm → HBM，逐层 H2D，用 GDR | 否（与前向重叠） |

### 5.5 存储介质三行表
| 内容 | 介质 | 副本 | 管理方 |
|---|---|---|---|
| 运行中的 block KV + state KV | 推理卡显存 | — | P/D 实例 |
| SWA KV | 产出它的机器内存 | 单副本，3FS 上没有 | P/D 实例；跨机走 400G VLAN |
| 分页 block KV | 独立 3FS 集群 | 应用层无副本，靠 3FS | `kvcache-store` |

### 5.6 弹性容错参数表
| 参数 | 值 | 出处 |
|---|---|---|
| 缩容宽限期 | 1 min，未完成请求连 KV 一次性转发 | QA:199 |
| 缩容门限 | quota 利用率 > 0.8 禁止缩容 | QA:203 |
| 重启时间 | 分钟级 | QA:207 |
| 容错手段 | 配额冗余 + 请求级重路由（不靠缩短重启） | QA:207 |
| EP 故障 | 单卡故障 → 整实例不可用；按健康卡数重定 EP size | QA:211 |
| EPLB prefill | 4 冗余专家/卡；窗 100；重排 1000 | QA:99 |
| EPLB decode | 1 冗余专家/卡（EP64→7 专家/卡）；窗 1000；重排 10000 | QA:103 |
| 存储单机 | 20×30.73 TB；读 ≥40 GB/s；写 ≥5 GB/s | QA:77 |
| 存储配比 | 14.4 TB/GPU（逻辑） | QA:81 |
| 3FS TTL | agent 3 天 / chatbot 7 天 | QA:153 |

### 5.7 recipe 流式状态机的阶段列表（11 态）
`Content` → `Json` → `AfterJson`；`ToolCalls` → `ToolName` → `ToolCallArguments` → `ToolCallArgumentName` → `ToolCallArgumentStringAttribute` → `ToolCallArgumentValue`；`Reasoning`；`Finished`。
匹配方式：逐 byte、每态一组标记、KMP 增量、`buffered_len` 保留半匹配进度。

### 5.8 开源"插槽"清单表
| 插槽 | 位置 | 现状 | 用途 |
|---|---|---|---|
| `EngineType` 三态 | `engine_type.py:11-13` / `scheduler.hpp:39-43` | 已实现 | 引擎裁剪成 P/D |
| `init_pd_split()` | `llm_engine.py:801-802` | 空 | P/D 数据结构初始化 |
| `send_processed_prompts()` | `llm_engine.py:848-850` | 空 | P→D 交接 prompt |
| `pe_prepare_save_kvcache()` / `pe_do_save_kvcache()` | `llm_engine.py:818-822` | 空 | PE 侧落盘 |
| `do_save_kvcache()` | `llm_engine.py:815-816` | 空 | 通用落盘 |
| `should_copy_state()` | `llm_engine.py:811-813` | 恒 `False` | 是否复制 state |
| `copy_state` + `do_copy_state()` | `llm_engine.py:284,787-798` | 缓冲区从不分配 | decode 侧状态副本区 |
| `pre_sampler_sync()` | `llm_engine.py:826-841` | 已实现（唯一调用点） | decode 步 save / prefill 步 prepare |
| `num_cached_tokens` / `state_id` | `pybind_sequence.cpp:126,131` | 已暴露 | 前缀/状态槽接缝 |
| `engine_group_rank` | `utils.py:223-230` | 已实现 | 多 engine world 端口隔离 |
