# R01 一手资料库：DeepInfer 与 DeepSeek-V4.1-Flash

## 0. 元信息

- 使用的代码版本：
  - `deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21 12:55 +0800，"Bump DeepEP to f6aa38fe and drop the SSH->HTT..."）
  - `deepseek-ai/deepseek-recipe` @ `935817c66f5d6e64838f419692545f06787e485f`（2026-10-20 01:49 +0800）
  - 本地路径注意：`/home/ubuntu/workspace/default/work` 是指向 `deepinfer-book-spec/work` 的软链接，两者等价。
- 查阅日期：2026-10-23（以下全部来源均为该日访问）。

### 0.1 来源清单（按等级分组）

**B 级：官方一手**

| # | 来源 | URL | 发布/更新 | 备注 |
|---|---|---|---|---|
| B1 | DeepSeek 知乎《DeepSeek V4.1 Flash 推理系统概览》 | https://zhuanlan.zhihu.com/p/2095560295930508953 | 2026-10-20 07:30 编辑 | 正文 + 6 条参考文献；已抓到全文（537 行快照） |
| B2 | DeepSeek 知乎《DeepSeek-V3 / R1 推理系统概览》 | https://zhuanlan.zhihu.com/p/27181462601 | 2025-02/03 | **未能抓取正文**（知乎反爬）；用官方英文版 B2' 替代 |
| B2' | `open-infra-index` Day 6 英文原文 | https://raw.githubusercontent.com/deepseek-ai/open-infra-index/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md | 2025-02-21 起 | 88 行，已抓全文 |
| B3 | DeepSeek 官方 X 公告（DeepInfer） | https://x.com/deepseek_ai/status/2112476496283775135 | 2026-10-20 09:30 | 4 条 thread；经镜像 twiscan 抓到 1/4 原文，文字与 The New Stack 引文一致 |
| B4 | HF 模型仓 `deepseek-ai/DeepSeek-V4.1-Flash-1017`（含 `config.json`） | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 | 2026-10-17 | 模型卡散文未抓到；`config.json`（3,311 B）已完整取得，见事实 #21b |
| B5 | 技术报告 arXiv:2609.19969 | https://arxiv.org/abs/2609.19969 | 2026-09-17 投递 | 摘要已抓；作者 DeepSeek-AI，591 人 |
| B6 | DeepInfer README | 本地 `README.md` | commit 同上 | 107 行 |
| B7 | `doc/inference_deployment_qa.md` | 本地 | commit 同上 | 312 行，官方英文 Q&A |
| B8 | `open-infra-index`《The Path to Open-Sourcing the DeepSeek Inference Engine》 | https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md | 2025-04-14 | "不现实"传言的原始出处 |
| B9 | vLLM 官方博客《DeepSeek-V4.1-Flash on vLLM: 5x Agentic Throughput Since Day 0》 | https://vllm.ai/blog/2026-10-07-deepseek-v41-flash | 2026-10-07 | 含 SWA bounded replay 与 kernel 数据 |
| B10 | SGLang 官方博客《SGLang and Miles Add Day-0 Support for DeepSeek-V4.1》 | https://www.sglang.io/blog/deepseek-v4.1-day0-support | 2026-09 | Day-0 特性/硬件覆盖表 |
| B11 | DeepSeek 官网新闻《Introducing DeepSeek-V4.1-Flash》 | https://www.deepseek.com/en/news/deepseek-v4-1-flash/ | 2026-09-09 | KV 1/4 HBM、1/8 SSD 的官方口径 |
| B12 | deepseek-recipe README | 本地 `README.md` | commit 同上 | 描述 Codon/编解码边界 |

**D 级：媒体 / 第三方（仅作旁证，不作任何结论的唯一来源）**

| # | 来源 | URL | 发布 | 信息量 |
|---|---|---|---|---|
| D1 | The New Stack, Paul Sawers | https://thenewstack.io/deepseek-deepinfer-huawei-nvidia/ | 2026-10-20 | 高：点出"去年说不实际"、缺失组件、NVIDIA/昇腾功能差异 |
| D2 | 腾讯科技《DeepSeek 进一步开放：模型之外，推理框架也开源了》 | https://news.qq.com/rain/a/20261020A0BSEW00 | 2026-10-20 17:45 | 高：把 DeepInfer 放回 2025 开源周→2025-04 说明→2026 独立仓的脉络 |
| D3 | DeepTech 深科技《DeepSeek 公布推理引擎开源路径，放弃开放完整代码库…》 | https://news.qq.com/rain/a/20250416A03YV000 | 2025-04-16 | 高：三项障碍的中文转述 + vLLM 官方回应 |
| D4 | warmctl 博客《Over 600K tokens/s: DeepSeek 4.1 Flash on 8×B300, vLLM vs DeepInfer》 | https://warmctl.io/blog/deepseek-b300-vllm-vs-deepinfer/ | 2026-10-21 | 高但需警惕：单节点、厂商博客、非 DeepSeek 生产配置 |

---

## 1. 三个最重要的发现

1. **DeepSeek 从未写过"为什么开源 DeepInfer"的专段。** 最接近"目的"的原话只有 X 推文那句 "We're sharing our production stack to help the community lower compute costs, reduce energy use, and make AI more affordable."；知乎/QA 给的是**设计目标**（多厂商芯片、8 卡 P + 64–128 卡 D、KV 留存 72h+），不是开源动机。书里若写"DeepSeek 表示开源是为了 X"，只能引这一句，不能扩写。
2. **"开源推理引擎还不现实"是二次压缩，不是 DeepSeek 原话。** 原文是 `impractical for public deployment without significant modifications`——主语是"这个引擎拿去公开部署"，不是"开源这件事"。而且当年结论是"改为向 vLLM 贡献组件"，不是"永不开源"；2026 年也没有解释障碍如何消除，只是把范围缩到"离线推理 + 规模化所需核心能力"。
3. **知乎文与 `inference_deployment_qa.md` 同源，不是两篇独立文章。** QA 自己写明 "This Q&A builds on that blog post"。差异是**层次**差异：知乎独有 V4→V4.1 的 SWA 缓存史与取舍论证、SWA KV 线上 TTL 180s/命中率 98%、3FS block 112 KiB 与 10 TiB/s、DSpark 相对 DFlash 的低秩 logit bias 机制、6 条参考文献；QA 独有低精度格式、集群/存储与最小部署单元参数、Mega MoE、FlashMLA 缺 SASS 的 5–10%、完整 DSpark 表、TP/CP 不用。

---

## 2. 事实条目

### #1 官方对"开源"的唯一目的性表述
- 证据：https://x.com/deepseek_ai/status/2112476496283775135（2026-10-20 09:30，1/4）
  > 🚀 Introducing DeepInfer, our in-house inference framework, now fully open source!
  > Powering DeepSeek-V4.1-Flash at scale with high throughput and low latency. We're sharing our production stack to help the community lower compute costs, reduce energy use, and make AI more affordable.
- 等级：B（官方 X；经镜像取得，文字与 D1 的引文逐字一致）
- 意味着什么：全书唯一可引的"开源动机"原话，落点是**降成本、降能耗、让 AI 更可负担**，不含"回馈生态""对齐 vLLM"之类。
- 疑点：镜像未显示 2/4–4/4 全文；2/4–4/4 的内容在二手摘要里出现但未能逐字核实，本笔记不引用。

### #2 知乎开篇：把"开源"和"生产成绩"绑在一句里
- 证据：B1 第 67 行
  > 此前，我们已开源 用于开发高性能算子的 TileLang 以及一系列推理算子库。**今天，我们正式开源大规模云服务推理框架 DeepInfer。** 基于这一框架，我们已实现 DeepSeek V4.1 Flash 的高吞吐、低延迟大规模推理服务。目前单一集群的服务吞吐是 155 万亿 Tokens/天，整体收入是每卡每小时 38 美金（按谷价）。
- 等级：B
- 意味着什么："开源"在这里是**声明动作**，不是论证。紧跟着给的是产能与收入，等于用生产数据给框架背书。
- 疑点：`38 美金/卡/小时` 是**收入**（revenue），不是毛利、不是成本节约；媒体标题写成"一张卡每小时收入 38 美元"是对的，书里不能写成利润。

### #3 官方对"开源了什么"的自我限定
- 证据：`README.md:3`
  > DeepInfer is the high-performance inference framework that powers DeepSeek's large-scale production services. This open-source release provides offline inference with the core features needed for serving at scale.
  以及 `README.md:38-42`
  > Required for a complete inference service (not included in this release): - A KV cache system - VE/Prefill/Decode disaggregation (most required engine functionality is already implemented) - Load balancing across data-parallel replicas for both prefill and decode
- 等级：B
- 意味着什么："开源了引擎"≠"开源了服务"。README 自己列了三项缺口，这是书里必须与"155 万亿 tokens/天"并列写的限定条件。
- 疑点：无。

### #4 官方明确承认 KV 管理与负载均衡代码未完整开源
- 证据：B1 第 71 行
  > 推理系统中的 KVCache 管理与负载均衡组件因与内部基础设施耦合较深，相关代码目前尚未完整开源。本文将从上述设计目标出发，介绍 DeepSeek V4.1 Flash 推理系统的整体架构与关键实现，以及这两部分的设计。
  英文对应 B7:21
  > The KV cache management and load balancing components of the inference system are tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced.
- 等级：B
- 意味着什么：知乎里最详细的两章（KVCache Storage、四个 Load Balancer）**在仓库里没有对应实现**。读码员必须按"文档有、代码无"处理，不能反推代码结构。
- 疑点：无（这是全书最重要的一条边界声明）。

### #5 2025-04 文档的三项障碍（"不现实"的原点）
- 证据：B8
  > While we initially considered open-sourcing our full internal inference engine, we identified several challenges:
  > * Codebase Divergence: Our engine is based on an early fork of vLLM from over a year ago. Although structurally similar, we've heavily customized it for DeepSeek models, making it difficult to extend for broader use cases.
  > * Infrastructure Dependencies: The engine is tightly coupled with our internal infrastructure, including cluster management tools, making it impractical for public deployment without significant modifications.
  > * Limited Maintenance Bandwidth: As a small research team focused on developing better models, we lack bandwidth to maintain a large open-source project.
- 等级：B
- 意味着什么：三条障碍分别是**代码分叉、基础设施耦合、人力**。第二条的 `impractical` 修饰的是 "public deployment"，不是 "open-sourcing"。
- 疑点：文件发布日期在仓库中不易直接读出；D3 记为 2025-04-14，仓库创建日为 2025-02-21。书里用"2025 年 4 月"较稳。

### #6 2025 年的结论是"换路径"，不是"不做"
- 证据：B8
  > Considering these challenges, we've decided to collaborate with existing open-source projects as more sustainable alternatives. Moving forward, we will work closely with existing open-source projects to: * Extract Standalone Features ... * Share Optimizations ...
  > [!NOTE] To clarify, this article outlines our approach to open-sourcing of our DeepSeek-Inference-Engine codebase only. Regarding future model releases, we maintain an open and collaborative stance ... enabling the community to achieve state-of-the-art (SOTA) support from Day-0.
- 等级：B
- 意味着什么：2025 年的立场是"**不从零开源整仓，改为拆组件 + 贡献 vLLM**"，并且单独声明"这只针对推理引擎代码库，模型发布仍走 Day-0 同步"。所以 2026 年的 DeepInfer 与 2025 年的判断不构成"打脸"，是路径换了（D2 也是这个判断）。
- 疑点：无。

### #7 媒体如何把"部署不现实"压成"开源不现实"
- 证据：D1 首段
  > DeepSeek has open-sourced the inference engine it uses to serve its models in production, a step it said last year wasn't yet practical.
  中文侧对应 D3 标题与正文
  > DeepSeek公布推理引擎开源路径，**放弃开放完整代码库**，承诺将核心优化注入vLLM生态
  > 在评估了直接开源内部完整推理引擎的可行性后，DeepSeek 认为存在几个关键障碍。
- 等级：D
- 意味着什么：**没有找到任何一家媒体使用"开源推理引擎还不现实"这一字面**。传播链是：原文 `impractical for public deployment`（讲部署）→ 英文媒体 "wasn't yet practical"（主语被省略）→ 中文转述"放弃开放/不现实"（主语彻底丢失）。这是翻译与省略造成的语义漂移，不是 DeepSeek 的表态变化。
- 疑点：不排除有中文自媒体使用该确切措辞，但检索未见可引用的一手转述；书里若要用这句话，应写成"有媒体这样概括"，不要写成"DeepSeek 曾说"。

### #8 DeepSeek 自己也没解释旧障碍怎么消失的
- 证据：D1
  > So what's changed now, exactly? Well, DeepSeek _hasn't_ explained how it overcame those earlier difficulties, although the scope of the new release offers some clues. ... several components needed to operate a complete production service are still missing. These include a KV cache management system, infrastructure for separating different stages of inference across machines, and load balancing across replicas.
  > ... That narrower release may help explain how the company has made its technology available despite the challenges it identified last year.
- 等级：D（判断）+ B（缺失组件清单见 #3）
- 意味着什么：目前唯一自洽的解释是"**范围缩小**"：三项障碍里的"基础设施耦合"被绕开的方式，就是不发布与内部基础设施耦合的那三层。这是推断，DeepSeek 没有明说。
- 疑点：属 E 级推断的边界；书里应写成"从发布范围看，最可能的解释是……"，不要写成官方说法。

### #9 知乎/QA 给的是设计目标，不是开源理由
- 证据：B1:59-63 / B7:7-13
  > DeepSeek V4.1 Flash 推理系统从设计之初，就确立了三个目标：
  > 1. 兼容多种厂商的计算芯片 … 2. 基于常见硬件组织 Prefill 和 Decode：用常见的 8 卡机器做 Prefill，用常见的 64～128 卡超节点做 Decode … 3. 支持将 KVCache 保留 72 小时以上 …
- 等级：B
- 意味着什么：这三条决定了全书的技术叙事骨架（多芯片、非对称 P/D、长留存 KV）。它们回答"系统为什么长这样"，不回答"为什么开源"。
- 疑点：无。

### #10 V4 的 SWA 落盘方案，以及 V4.1 为什么弃用 Zero SWA Caching（知乎独有）
- 证据：B1:220
  > V4 的做法是落盘加精确重算：SWA KVCache 在两个位置写入持久化存储，未命中时重算"层数 × 窗口"个 token 做完整前向，逐层精确恢复窗口状态。这条路径代价很高，必须用足够大的 SSD 容量把未命中率压到极低，代价是 SWA KVCache 占去持久 KVCache 近一半容量，且未能带来相应的命中收益：Block KVCache 呈长尾复用，一段前缀可在数天内被反复命中；SWA KVCache 只在活跃会话内被分钟级复用，会话结束或进入下一轮即失效。以数天留存的 SSD 容量承载分钟级生命期的数据，在多轮短对话场景下尤其不划算。因此，V4 技术报告提出的 Zero SWA Caching（即完全不落盘、仅依赖精确重算）因其代价在生产部署中过高，在 V4.1 推理系统中未实际采用。
- 等级：B
- 意味着什么：这是全书解释"SWA bounded replay 为什么必须存在"的最佳段落：它把设计取舍讲成了**生命周期错配**（数天容量 vs 分钟数据），比单纯说"重算便宜"有力得多。
- 疑点：`inference_deployment_qa.md` 完全没有这段历史与论证（见 #16 对照）。

### #11 SWA KV 的线上寿命与命中率（知乎独有）
- 证据：B1:202
  > 由于 SWA KV 只在会话内的分钟级窗口里被复用，池子总容量虽小但周转快，足以覆盖同时活跃的会话，线上数据显示，其 TTL 约 180 秒，命中率在 98% 以上。被驱逐造成的未命中由上述的 Encoder SWA Bounded Replay 重算。
- 等级：B
- 意味着什么："98% 命中率"是 SWA 不落盘仍可行的**定量前提**；没有这个数，"不持久化"看起来就是纯损失。
- 疑点：`180 秒` 与 `98%` 只出现在知乎，QA 无；引用时必须注明出自知乎，且未说明统计窗口与流量类型。

### #12 3FS 的 block 粒度与聚合带宽（知乎独有）
- 证据：B1:196
  > 3FS 分布式文件系统为 KVCache 提供约 100 PiB 的 SSD 存储，agent 流量 KVCache 的 TTL 达到 3 天，chatbot 流量达到 7 天。… 根据经验，3 天左右的 TTL 会造成 chatbot 流量的 KVCache 命中率下降 0.5%-1%；带宽方面，3FS 存储集群针对 KVCache block 粒度（约 112 KiB）的随机读写进行优化，峰值聚合读带宽达到 10 TiB/s，峰值聚合写带宽约 850 GiB/s。
- 等级：B
- 意味着什么：`112 KiB` 解释了为什么"block KV cache"的读写是随机小块而非大文件顺序 IO，也是 3FS 需要专门优化的原因。`10 TiB/s : 850 GiB/s` 的读写比（约 12:1）本身就是"prefill 重读、decode 少写"的负载画像。
- 疑点：QA 只有单机 `40 GB/s 读 / 5 GB/s 写`（B7:77），与这里的集群聚合值不是一个量级，书里不可混用。

### #13 DSpark 相对 DFlash 的机制差异（知乎独有）
- 证据：B1:229
  > DSpark 是 DeepSeek 提出的投机解码算法框架，相比 DFlash 在输出的 base logits 上叠加只依赖前一 token id 的低秩 logit bias，用极轻量的串行修正补回相邻 token 的局部条件化信息，提升 draft 准确率，具体可参考 DSpark 论文[4]。
- 等级：B
- 意味着什么：这句话给出 DSpark 的**一句话定义**（半自回归 + 低秩 logit bias），是全书解释"为什么 draft 更准"的唯一官方表述。QA 只讲"验证长度怎么调度"，不讲 draft 怎么来的。
- 疑点：`DFlash` 在仓库中的实现位置未核实（留给读码员）；B1 未给 DFlash 的出处。

### #14 动态验证长度的量化收益（QA 比知乎细）
- 证据：B7:247
  > Dynamic-length verification cuts the number of draft tokens verified per step by about 30% (5.00 → 3.2–4.4) while leaving the average accepted length essentially unchanged (3.29 vs. 3.20). Computed from EP64/DP64 benchmark data, that works out to +4–7% throughput at the production operating point (~60 concurrent sequences per rank).
  知乎对应处只给结论：`相比固定验证 5 个 token，线上吞吐提升 6%`（B1）
- 等级：B
- 意味着什么：知乎的"6%"是 QA 那张 6 行表在 60 conc 处的 +5.4% 的口头化。**书里用表格，不要用 6%**，因为表里 16 conc 是 −1.5%、96 conc 是 +11.0%，单点数字会误导。
- 疑点：表内数字标注为 "estimated throughput"，是估算不是实测。

### #15 EPLB 的 `moe_lb` 两处官方数字互相冲突
- 证据：B1:164
  > 统计窗 100 步，每 1000 步重排，线上数据实测 moe_lb（最慢 rank / 均值）= 1.0320（EP2/DP2）/ 1.0618（EP4/DP4）。
  B7:99
  > Prefill settings: start from 4 redundant experts per GPU …, a 100-step statistics window, and a re-assignment every 1,000 steps. Measured `moe_lb` (slowest rank / mean) = 1.074 (EP4DP4) / 1.045 (EP2DP2).
- 等级：B（两处都是官方）
- 意味着什么：同一指标、同一配置，知乎 1.0320/1.0618，QA 1.074/1.045，**四个数字没有一个对得上**。书里只能二选一，或者写成区间并注明"官方两版文档数字不一致"。
- 疑点：可能是不同统计窗口/不同批次数据的重测；官方未说明。这是需要向 DeepSeek 或维护者确认的点。

### #16 QA 独有：低精度格式逐项
- 证据：B7:217-221
  > - Main KV: per-16, e4m3 — like NVFP4 but without the global scale.
  > - Indexer KV: per-32, ue8m0, MXFP4.
  > - MoE routed-expert weights: per-32, ue8m0, MXFP4.
- 等级：B
- 意味着什么：知乎全文没有 `MXFP4` / `e4m3` 字样（已 grep 确认 0 次）。书的"低精度"小节只能靠 QA。"per-16 e4m3 但没有 global scale"是理解 DeepSeek 版 NVFP4 的关键差异。
- 疑点：未给 `main KV` 的 scale 具体排布（是否 per-16 沿 K 维还是 head 维），需读码确认。

### #17 QA 独有：集群与存储的工程参数
- 证据：B7:73-87
  > Budget roughly 400 Gbps of interconnect bandwidth per 8 GPUs. In practice, DeepSeek's inference nodes are deployed in different but adjacent data centers.
  > We recommend 20 × 30.73 TB SSDs per KV cache storage server, with each server providing at least 40 GB/s read and 5 GB/s write throughput and enough total capacity to retain 24 hours of data.
  > In practice, about 14.4 TB per GPU measured as logical size; the physical capacity required is larger than that.
  > Data is stored on 3FS with 10+2 erasure coding; each EC group tolerates two simultaneous node failures. A read issues requests to all 11 storage nodes in the group simultaneously; any 10 responses are enough to decode and return.
- 等级：B
- 意味着什么：这组数字让"KV 缓存要多少台存储"变成可算的题（14.4 TB/GPU × GPU 数 ÷ 单机 614.6 TB）。知乎只有 `10+2 EC` 与 `100 PiB`，没有这些配置。
- 疑点：`30.73 TB` 是单盘裸容量（20 盘 ≈ 614.6 TB/台），未扣 EC 与水位。

### #18 QA 独有：Mega MoE 与 FlashMLA 的 5–10%
- 证据：B7:231-237
  > Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE.
  > They are broadly the same; the indexer and mega series differ somewhat. FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower. End-to-end performance is close — at most 5% apart.
- 等级：B
- 意味着什么：官方承认"开源版 ≈ 内部版，差距 ≤5%"——这是评估 DeepInfer 复现价值时最重要的一句，也是全书中少见的**自曝差距**。
- 疑点：`Mega MoE` 在知乎出现 0 次；开源仓库中是否有 Mega MoE 实现需读码确认（README 未列该名称）。

### #19 vLLM 官方口径：SWA bounded replay 的收益与代价
- 证据：B9
  > We implemented SWA bounded replay with CUDA graphs, achieving a ~30% TTFT reduction.
  > **Decoder replay with CUDA graphs cuts prefill computation time by 30–40%.** ... without them, launch overhead outweighs the GPU savings and replay is slower than the baseline (up to +12% at 1K on DEP2).
  > Although SWA bounded replay is not exact, DeepSeek has reported only negligible quality loss. We confirmed this in vLLM on benchmarks including GSM8K and GPQA and observed no meaningful accuracy difference (gaps within about 1.5 standard errors).
- 等级：B
- 意味着什么：vLLM 独立复现了 SWA bounded replay，并给出**副作用**：短 prompt（1K）下若没有 CUDA graph，重放比不重放还慢 12%。这条"负收益区间"是知乎/QA 都没有的。
- 疑点：测试条件为 NVIDIA GB200、TP4/DEP2、关闭 prefix caching、单请求，不能直接搬到 DeepSeek 生产配置。

### #20 SGLang 官方口径：Day-0 覆盖与 Engram 放置
- 证据：B10
  > SGLang supports flexible placement of Engram tables in GPU or host memory. On NVIDIA GB-series systems, SGLang takes advantage of fast CPU–GPU connectivity to move these tables into host memory, freeing GPU HBM for KV cache.
  > All Day-0 recipes use preview images; support has not shipped in a stock SGLang release yet.
- 等级：B
- 意味着什么：`Engram` 是 V4.1 的新组件，SGLang 的解法是"放 host memory 换 HBM"；这与 DeepInfer 的 `ENGRAM_FORCE_GPU_STORAGE` 是同一权衡的两个方向，可做对比表。
- 疑点：SGLang 提到 "We are still integrating the new DeepSelect"，说明 Day-0 覆盖并不完整。

### #21 模型规模与 KV 占用的官方数字
- 证据：B5 摘要
  > ... a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill. ... These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash. Further, through a dedicated deployment optimization known as SWA Bounded Replay, DeepSeek-V4.1-Flash reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 ...
- 等级：B
- 意味着什么：`890 B/token`、1/4、1/8 三个数字的**权威出处是技术报告摘要**，不是知乎。B11 的官方新闻口径一致（1/4 HBM、1/8 SSD）。
- 疑点：`552B backbone` 与 SGLang 的 `552B-parameter MoE` 一致。技术报告 PDF 正文（1.81 MB）未能取到——HF 的 `raw` 链接只返回 LFS 指针文件（`size 1809802`），所以报告里除摘要外的细节（如质量损失幅度、消融）本轮未核实。

### #21b `config.json` 已取得：结构参数可逐项核对
- 证据：`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json`（3,311 B，全文已缓存）
  > "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41"
  > "text_config": { "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64, "num_key_value_heads": 1, "head_dim": 512, "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8, "max_position_embeddings": 1048576 }
  > "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "routed_scaling_factor": 1.5, "sliding_window": 128
  > "kv_source_layer_ids": [2, 8, 14, 20], "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36], "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8
  > "hc_mult": 4, "hc_sinkhorn_iters": 20, "engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, "engram_n_heads": 8, "engram_head_dim": 256
  > "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256, "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3
  > "quantization_config": { "quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4" }
  > "vision_config": { "num_hidden_layers": 32, "hidden_size": 1024, "patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936 }
- 等级：B（HF 模型仓配置文件）
- 意味着什么：**`dspark_markov_rank: 256` 直接印证知乎"低秩 logit bias"的说法**（事实 #13）——DSpark 确实带一个 rank=256 的低秩 Markov 修正项；`dspark_block_size: 5` 对应"一次生成 5 个 draft"；`sliding_window: 128` 对应 SWA 窗口；`kv_source_layer_ids` 只有 4 层（2/8/14/20）说明**全局 KV 来自 4 个源层、由其余层复用**，这是"1/4 KV"的结构来源；`engram_layer_ids [1,14]` 说明 Engram 只挂 2 层，但有 3.84 亿行 ×2 的大表。
- 疑点：`num_nextn_predict_layers: 3` 与 `dspark_block_size: 5` 数值不一致，二者语义关系（MTP 层数 vs 每步 draft 数）需读码确认；`compress_ratios` 数组长度 43 ≠ `num_hidden_layers` 40，含义未确认（推测含 VE/额外层，标 E）。

### #22 第三方单节点实测：DeepInfer 在 8 卡上并不快
- 证据：D4
  > vLLM 658K input tok/s vs DeepInfer 72K on 8K prompts (9.2×); decode 1K→1K 55.0K vs 27.9K (2.0×)
  > On a single node, with the settings DeepSeek recommends, it is not the faster engine.
- 等级：D（有方法、有版本号，但单节点、厂商博客、非官方配置）
- 意味着什么：这是目前唯一**公开的 DeepInfer vs vLLM 同机对比**。它说明 DeepInfer 的优势在 rack-scale EP（64–128 卡），不在单节点；书里若引用必须写清"8×B300 单节点、EP8×DP8、离线 API"。
- 疑点：作者自述首次运行用默认参数只得 37.2K，改配置后 55.0K——说明结果对配置极敏感；且 DeepInfer 侧走 `LLM.generate` 省掉了 HTTP/tokenizer 开销，比较并不对等（对 DeepInfer 有利）。另外该文用的是 `8b16149`（仓库第 2 个 commit），与本书锁定的 `9bd9cf2` 相差 2 个 commit。

### #23 腾讯科技：DeepInfer 与 Harness 的分工
- 证据：D2
  > DeepInfer也与DeepSeek此前开源的Harness有所区别：Harness负责组织模型、工具和任务执行过程，DeepInfer负责底层模型计算。两者可以配合，分别影响任务如何完成，以及计算如何高效进行。
- 等级：D
- 意味着什么：给读者一个不混淆的边界：Harness = agent 编排层，DeepInfer = 推理执行层，deepseek-recipe = 协议编解码层。
- 疑点：无（属描述性，与 B12 的 README 自述一致）。

---

## 3. 必答问题逐条回答

### Q1. DeepSeek 对"为什么开源 DeepInfer"的原话是什么？

**答：不存在"为什么开源"的专段论述。** 三处最接近的原话，按可引用强度排序：

1. **X 推文（唯一带目的从句的句子）**——事实 #1：
   > We're sharing our production stack to help the community lower compute costs, reduce energy use, and make AI more affordable.
   （中文：我们把生产栈分享出来，帮助社区降低算力成本、减少能耗，让 AI 更可负担。）
2. **知乎/QA 的收尾句（讲的是系统目标，不是开源动机）**——事实 #2、#9：
   > 围绕这些目标，我们持续优化计算、通信、调度与存储，在满足延迟要求的前提下提高系统的有效吞吐，用更少的算力与能源支撑更多请求，让智能服务以更低的价格触达更多用户。
3. **对未来开源范围的承诺**——B1:69 / B7:19：
   > 未来，DeepSeek 新模型的推理代码及后续性能优化也将在 DeepInfer 中持续开源。

**不能写进书里的三种说法**（都无一手支持）：
- "为了回馈开源社区 / 因为受益于 vLLM 所以开源"——2025-04 文档有这个逻辑（"our inference engine is built upon vLLM … we want to give back"），但 **2026 年的公告没有重复这句话**；README 的 Acknowledgments 只写 "originally built primarily on vLLM"。
- "为了建立生态 / 对抗闭源"——纯媒体解读（D 级）。
- "因为障碍已经解决"——DeepSeek 未解释（事实 #8）。

### Q2. 媒体传"开源推理引擎还不现实"的原始出处与真实背景是什么？

**原始出处**：`deepseek-ai/open-infra-index` 仓库 `OpenSourcing_DeepSeek_Inference_Engine/README.md`，标题《The Path to Open-Sourcing the DeepSeek Inference Engine》，2025 年 4 月（D3 记为 4 月 14 日）。原文关键句（事实 #5）：

> Infrastructure Dependencies: The engine is tightly coupled with our internal infrastructure, including cluster management tools, **making it impractical for public deployment without significant modifications**.

**真实背景，四层**：

1. **`impractical` 的主语是"部署"，不是"开源"。** 整句在讲：这个引擎跟内部集群管理工具绑死，不大改就没法给外部公开部署。它没说"开源这件事不现实"。
2. **当年给的是三条具体障碍，不是一句态度。** 代码分叉（一年多前的 vLLM 分支 + 深度定制）、基础设施耦合、人力有限（小研究团队维护不了大型开源项目）——事实 #5。
3. **当年的结论是"换路径"，不是"拒绝"。** 明确写 "we've decided to collaborate with existing open-source projects as more sustainable alternatives"，做法是拆独立库 + 把优化贡献回 vLLM——事实 #6。同期 vLLM 官方账号回应称这是"以正确的方式开源引擎"（D3 转述，未核到原推）。
4. **"不现实"是传播中的语义漂移，且没有确切字面来源。** 传导链：原文 `impractical for public deployment`（讲部署）→ 英文媒体 "a step it said last year wasn't yet practical"（主语省略，见事实 #7）→ 中文"放弃开放完整代码库 / 开源不现实"（主语彻底丢失）。**检索未发现任何一家媒体使用"开源推理引擎还不现实"这一确切句子**；最接近的英文表述是 The New Stack 的 "wasn't yet practical"，最接近的中文表述是 DeepTech 的"放弃开放完整代码库"。

**2026 年为什么又能开了**：DeepSeek 自己没说（事实 #8）。从发布范围看，最合理的解释是**把与内部基础设施耦合的三层（KV cache 系统、P/D 分离、DP 副本负载均衡）留在闭源侧**，只开源"离线推理 + 规模化所需核心能力"（事实 #3、#4）。因此更准确的说法是：**"当年不开源整仓"这件事没有反转，"开源整仓"这件事从未发生过**——2026 年开的是一个范围明确收窄的引擎。

### Q3. 知乎文章和 `inference_deployment_qa.md` 有哪些内容不重合？知乎多讲了什么？

**先定性**：两者同源。QA 开头自述（B7:21）：

> **This Q&A builds on that blog post** and will be updated continuously to answer more detailed questions …

QA 的第 1–21 行基本是知乎开篇（第 55–71 行）的英文逐段翻译：同一句"模型划定能力边界 / 推理框架解决成本"、同样三条设计目标、同样 155 万亿 tokens/天与 38 美金/卡/小时、同样"KV 管理与负载均衡未完整开源"的声明。**所以差异不在"讲了什么主题"，而在"讲到什么颗粒度"。**

#### A. 知乎独有（QA 完全没有，已逐项 grep 确认）

| # | 内容 | 知乎位置 | QA 情况 |
|---|---|---|---|
| A1 | **V4 → V4.1 的 SWA 缓存演进史与取舍论证**：V4 落盘 + 精确重算；SWA 占掉近一半持久容量却无相应命中收益；`Zero SWA Caching` 因代价过高在 V4.1 未采用 | B1:220 | 无。QA 只讲"SWA KV 不进持久缓存"的现状，不讲为什么曾经进过、为什么退出 |
| A2 | **SWA KV 线上 TTL ≈ 180 秒、命中率 > 98%** | B1:202 | 无（QA 无 180 / 98% 任何数字） |
| A3 | **3FS block 粒度 ≈ 112 KiB、峰值聚合读 10 TiB/s、峰值写 ≈ 850 GiB/s** | B1:196 | 无（QA 只有单机 40 GB/s / 5 GB/s） |
| A4 | **DSpark 相对 DFlash 的机制**：base logits 上叠加只依赖前一 token id 的低秩 logit bias，用极轻量串行修正补局部条件化 | B1:229 | 无。QA 只讲 DSpark 的"变长验证 + Confidence Head 调度"，不讲 draft 模型本身怎么来 |
| A5 | **6 条参考文献**：DualPath (SIGCOMM'26)、V4.1 技术报告 (arXiv:2609.19969)、3FS、DSpark (arXiv:2607.05147)、SGLang (NeurIPS'24)、NanoFlow (OSDI'25) | B1:254-259 | 无参考文献表；只在正文提到 "DualPath's intra-group scheduling algorithm" 而无出处 |
| A6 | **线上统计图 / 离线基准图（图形式）与图注** | B1 全文 | QA 只有 `wide_ep_decode_pareto.png` 与 SWA 表、DSpark 表 |
| A7 | **EPLB `moe_lb` 的数值为 1.0320 (EP2/DP2) / 1.0618 (EP4/DP4)** | B1:164 | QA 给 1.074 (EP4DP4) / 1.045 (EP2DP2)——**四个数字全不一致**，见事实 #15 |

#### B. QA 独有（知乎没有，已逐项 grep 确认）

| # | 内容 | QA 位置 | 知乎情况 |
|---|---|---|---|
| B1 | **低精度格式逐项**：main KV `per-16, e4m3`（像 NVFP4 但无 global scale）、indexer KV `per-32 ue8m0 MXFP4`、MoE 路由专家权重 `MXFP4` | B7:217-221 | 全文 0 次 `MXFP4` / `e4m3` |
| B2 | **集群与存储配置**：跨数据中心 400 Gbps / 8 卡；存储服务器 20 × 30.73 TB SSD、单机 ≥40 GB/s 读 / 5 GB/s 写、容量留 24h；14.4 TB/GPU（逻辑）；10+2 EC 的 11 节点读 | B7:73-89 | 0 次 `400Gbps` / `14.4` / `30.73`；只有 `10+2` 与 `100 PiB` |
| B3 | **部署形态**：1P+1D 可部署、P:D ≈ 1:1、最小高效单元 64+64、VE 单卡够用、**TP 与 CP 目前不用** | B7:47-67 | 0 次"张量并行" |
| B4 | **`state_size_to_disk` 的算术**：20 层 × 134 token × 528 B = 1,415,040 B/序列 | B7:169 | 无该算式 |
| B5 | **Mega MoE**：scale-up 域内通解，任何 EP size / batch 都显著快于 unfused MoE | B7:231 | 0 次 `Mega MoE` |
| B6 | **FlashMLA 缺一个手调 SASS 优化 → decode kernel 慢 5–10%，端到端 ≤5%** | B7:235 | 0 次 `SASS` |
| B7 | **DSpark 完整 6 行表**（16/32/48/60/72/96 conc，−1.5% → +11.0%）与"内部 draft 模型略新但差距小" | B7:249-260 | 只有一句"线上吞吐提升 6%" |
| B8 | **decode 不做 KV offload、不用 Hi-Sparse；prefill 也用 paged KV** | B7:143-149 | 0 次 `offload` / `Hi-Sparse` |
| B9 | **自动扩缩容与容错**：CE 收到关机通知后 1 分钟宽限并整体转移请求+KV；配额利用率 >0.8 禁止缩容；重启分钟级；大 EP 故障后按健康卡数重定 EP size | B7:193-211 | 有"宽限"与"分钟量级"，但无 0.8 阈值、无 EP size 重定 |
| B10 | **EPLB decode 参数**：1 个冗余专家起步（EP64 向上取整为 7）、1000 步统计窗、10000 步重排、`moe_lb` = 1.090 | B7:101-103 | 有 `1.090` 与该段策略（此项知乎也有，仅参数颗粒度略低） |

#### C. 一句话总结

**知乎多讲"为什么这样设计"（历史、取舍、生命周期论证、可追溯的引用）；QA 多讲"具体怎么配"（位宽、盘数、带宽、步数、阈值、表格）。** 写书时：讲设计动机用知乎，讲工程参数用 QA，两者冲突处（A7）标注官方两版不一致。

---

## 4. 新发现的问题

1. **`moe_lb` 到底是多少？**（事实 #15）知乎 1.0320/1.0618 vs QA 1.074/1.045，同一指标四个数全冲突。这直接影响"EPLB 效果有多好"的叙述，建议向维护者或 issue 求证，或在书里并列并注明"官方两版不一致"。
2. **DSpark 的论文在哪？** 知乎引 `arXiv:2607.05147`（《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》），但仓库 README 只把它当 feature，未链论文。需确认该 arXiv 编号是否可访问、是否即官方 DSpark 论文。
3. **HF 模型卡的散文说明与技术报告 PDF 正文没拿到，但 `config.json` 拿到了。** `config.json`（3,311 B）本轮已完整取得并核对（事实 #21b）；技术报告 PDF 经 HF `raw` 链接只返回 LFS 指针（`size 1809802`），模型卡页面本身只显示文件列表，因此报告的正文细节（质量损失幅度、消融、与 V4 的逐项对比表）本轮未核实。若书里要写"40 层 / 全局 KV 来自第 2、8、14、20 层"，可直接引 `config.json`（B 级），不必绕道 vLLM 博客。
4. **`Mega MoE` 在开源代码里存在吗？** QA 说它是 scale-up 域内通解（事实 #18），但 README 的 feature 列表没有这个词，知乎 0 次。若代码里找不到，就属于"文档有、代码无"。
5. **"155 万亿 tokens/天"的边界是什么？** 官方只说"单一集群"（B1:67 / B7:17）。是哪个集群、含哪些流量（web/APP/API）、统计窗口多长，均未说明；与 2025 年 V3/R1 的"608B 输入 + 168B 输出/天"不可直接比较。
6. **`38 美金/卡/小时` 是收入还是毛利？** 官方写 "整体收入" / "overall revenue"，即收入。但按什么口径折算（含缓存命中折扣？含谷价？）未说明。媒体标题易被读成利润，书里必须写明"收入"。
7. **昇腾侧的能力缺口有多大？** README 说支持 Ascend 950 NPU，但 `DEEPINFER_BATCH_INVARIANT` 与确定性采样标注 "NVIDIA only"（README:27-28），D1 也点出这一点。昇腾上到底缺哪些 feature，官方没给对照表。

---

## 5. 可用于图表的材料

### 5.1 时间线（"开源路径"叙事骨架）

| 时间 | 事件 | 来源 |
|---|---|---|
| 2025-02-21 起 | 开源周：FlashMLA / DeepEP / DeepGEMM / DualPipe / EPLB / 3FS | D2、B2' |
| 2025-02-24 前后 | 《DeepSeek-V3/R1 推理系统概览》：278 节点峰值、545% 理论成本利润率 | B2' |
| 2025-04-14 | 《The Path to Open-Sourcing the DeepSeek Inference Engine》：三条障碍，改为贡献 vLLM | B8、D3 |
| 2026-09-09 | V4.1-Flash 发布（552B MoE、CED、890 B/token） | B11、B5 |
| 2026-09-10 / 09-17 | API 上线；技术报告 arXiv:2609.19969 | B11、B5 |
| 2026-10-17 | V4.1-Flash-1017 权重开源 | B3（X 2026-10-17） |
| 2026-10-20 07:30 | 知乎《V4.1 Flash 推理系统概览》 | B1 |
| 2026-10-20 09:30 | X 公告 DeepInfer 开源（1/4） | B3 |
| 2026-10-20 | DeepInfer 仓库 + `deepseek-recipe` 公开；README / QA 同日 | B6、B7 |
| 2026-10-20 12:09 | DeepInfer `initial commit`（`d585e65`）——仓库公开历史仅 4 个 commit，无更早历史 | A（`git log`） |
| 2026-10-21 12:55 | HEAD `9bd9cf2`（Bump DeepEP / 去掉 SSH→HTTP），即本书锁定版本 | A |

### 5.2 三代推理系统对比表（草稿）

| 维度 | V3/R1（2025-02） | V4.1-Flash（2026-09/10） | 差异要点 |
|---|---|---|---|
| 硬件 | H800 × 8/节点，峰值 278 节点 | NVIDIA SM100（B200/B300/GB300）、昇腾 950 | 从单一 NVIDIA 到双 vendor |
| 并行 | Prefill EP32/DP32（4 节点）；Decode EP144/DP144（18 节点） | Prefill EP2–8/DP2–8；Decode EP64–128（Wide-EP） | Decode EP 从 144 降到 64–128，但单卡专家更少 |
| 阶段划分 | P/D 分离 | VE/P/D 三分离 | 多模态带来 VE 阶段 |
| 通信重叠 | Prefill 双 micro-batch；Decode 5 级流水 | 未在知乎/QA 展开（待读码） | 需补 |
| KV 留存 | 磁盘 KV 命中 56.3%（342B/608B） | 3FS ~100 PiB；agent 3 天 / chatbot 7 天；SWA KV 180 秒 | 从"落盘"到"分层：block 落盘 + SWA 只留内存" |
| 负载均衡 | Prefill / Decode / EP 三个 | Prefill / Decode / VE / EP 四个 | 多一个 VE LB |
| 生产数字 | 73.7k 输入 tok/s/节点；14.8k 输出 tok/s/节点；545% 理论利润率 | 155 万亿 tokens/天/集群；38 美元/卡/小时（收入） | **口径不同，不可直接比** |

### 5.3 开源范围边界图（材料齐备，可直接画）

```
┌─────────────── 已开源（Apache 2.0） ───────────────┐
│ DeepInfer（离线推理）                              │
│  · kernels: FlashMLA / DeepGEMM / DeepEP /         │
│    TileKernels / DeepSelect / TileLang ops         │
│  · Attention DP + balanced CP；MoE EP + EPLB       │
│  · DSpark + confidence-scheduled verification      │
│  · SWA bounded replay（encoder/decoder）           │
│  · 异步调度（零开销 batch scheduler）              │
│  · batch-invariant / 确定性采样（仅 NVIDIA）        │
│  · 结构化输出（regex / JSON schema / EBNF）        │
│ deepseek-recipe（Rust + Python 绑定）              │
│  · Messages / Chat Completions / Responses 互转    │
│  · 分段并行 tokenize、流式增量解析                 │
└────────────────────────────────────────────────────┘
┌─────────────── 未开源（README 明列） ───────────────┐
│ · KV cache 系统（3FS 集群、kvcache-store）          │
│ · VE / Prefill / Decode 分离部署                    │
│ · DP 副本间的 Prefill / Decode 负载均衡             │
└────────────────────────────────────────────────────┘
        ▲ 知乎 + QA 详细描述了后三层的设计，
          但仓库里没有对应代码（事实 #4）
```

### 5.4 关键数字表（全部带出处与条件）

| 数字 | 含义 | 条件 | 出处 | 等级 |
|---|---|---|---|---|
| 155 万亿 | tokens/天 | 单一集群 | B1:67 / B7:17 | B |
| 38 美元 | 收入/卡/小时 | 谷价 | B1:67 / B7:17 | B |
| 552B / 8B / 16B | 总参 / prefill 激活 / decode 激活 | — | B5, B11 | B |
| 890 B | 全局 KV/token | 恒在 HBM | B5 | B |
| 40 层 / 5120 hidden | 文本层数与宽度 | `text_config` | `config.json` | B |
| 384 + 1 专家, top-6 | 路由专家数 / 共享专家 / 每 token 专家数 | `n_routed_experts` 等 | `config.json` | B |
| 4 个 KV 源层 | 全局 KV 只由 2/8/14/20 层产出 | `kv_source_layer_ids` | `config.json` | B |
| 128 | SWA 窗口 | `sliding_window` | `config.json` | B |
| 256 | DSpark 低秩 Markov rank | `dspark_markov_rank` | `config.json` | B |
| 1,048,576 | 最大位置（1M 上下文） | yarn factor 16，原 65536 | `config.json` | B |
| 1/4、1/8 | KV 占用 vs V4-Flash | HBM / SSD+host | B5, B11 | B |
| 100 PiB | 3FS SSD 容量 | KVCache 专用 | B1:196 | B |
| 112 KiB | 3FS KV block 粒度 | 随机读写优化 | B1:196 | B |
| 10 TiB/s / 850 GiB/s | 3FS 峰值聚合读/写 | 集群级 | B1:196 | B |
| 180 s / 98% | SWA KV TTL / 命中率 | 线上数据，未给窗口 | B1:202 | B |
| 3 天 / 7 天 | block KV TTL | agent / chatbot | B1:196, B7:153 | B |
| 0.5%–1% | chatbot 命中率损失 | TTL=3 天时 | B1:196, B7:153 | B |
| 1,415,040 B | SWA state/序列 | 20 层 × 134 tok × 528 B | B7:169 | B |
| 5.00 → 3.2–4.4 | 每步验证 draft 数 | EP64/DP64 | B7:247 | B |
| +4%–7% | 吞吐增益 | ~60 conc/rank | B7:247 | B |
| +5.4%（60 conc）/ −1.5%（16 conc） | 同上的分点值 | 表 | B7:251-256 | B |
| 5%–10% / ≤5% | FlashMLA decode kernel / 端到端差距 | 内部版 vs 开源版 | B7:235-237 | B |
| 30%–40% | prefill 计算时间降幅 | vLLM，GB200，单请求，关 prefix cache | B9 | B |
| ~30% | TTFT 降幅 | 同上，加 CUDA graph | B9 | B |
| 5.3× / 1.9× | 吞吐 / 低延迟提升 | vLLM，SemiAnalysis AgentX，GB300 | B9 | B |
| 1,130 tok/s/user | 单用户速度 | 4×GB300，bs=1，EP1，4K in/1K out | B10 | B |
| 658K vs 72K | prefill tok/s，vLLM vs DeepInfer | 8×B300 单节点，8K prompt，2048 并发 | D4 | D |
| 55.0K vs 27.9K | decode tok/s，vLLM vs DeepInfer | 8×B300，1K→1K | D4 | D |

### 5.5 可直接复用的原文金句（≤5 行，附出处）

1. `We're sharing our production stack to help the community lower compute costs, reduce energy use, and make AI more affordable.` — B3
2. `making it impractical for public deployment without significant modifications` — B8
3. `The KV cache management and load balancing components of the inference system are tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced.` — B7:21
4. `以数天留存的 SSD 容量承载分钟级生命期的数据，在多轮短对话场景下尤其不划算。` — B1:220
5. `FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.` — B7:235
