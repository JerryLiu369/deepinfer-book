---
# R16 性能数据基准：DeepInfer / vLLM / SGLang 全量数字、统一大表与"不可比性"论证

## 0. 元信息

- 代码版本：`deepseek-ai/DeepInfer` @ `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`（2026-10-21 12:55 +0800）。本地 `research/sources/DeepInfer`。
- 查阅日期：2026-10-23（所有 URL 均为该日访问）。
- 主要来源：
  - 仓库内：`README.md:77-83`、`doc/inference_deployment_qa.md`（312 行）、`doc/images/{decode,prefill,wide_ep_decode}_pareto.png`、`examples/sweep/{run_sweep.sh,plot_pareto.py,ve_sweep.py}`、`examples/benchmark.py`（588 行）、`examples/README.md:30-121`
  - 已完成的笔记：R01（§5.4 数据表、#14/#15/#18/#19）、R09（§5.2 DSpark 表）、R12（#23/#24/#25、生产数字）、R13（#32-#35、§3 第 10 问）
  - vLLM：官方 X `@vllm_project` 2026-10-21（status 2112703311933747347）；博客《DeepSeek-V4.1-Flash on vLLM: 5x Agentic Throughput Since Day 0》2026-10-07（vllm.ai/blog/2026-10-07-deepseek-v41-flash）
  - SGLang/LMSYS：《SGLang and Miles Add Day-0 Support for DeepSeek-V4.1》2026-09-10；《DeepSeek-V4.1 Flash on SGLang: from 35 to 873 tokens/s》2026-09；《SGLang Low-Latency Mode》2026-10-18
  - 第三方：SemiAnalysis InferenceX（GB300 NVL72 / B300 / B200 三个 pairing 页，run 2026-10-02→10-22）；Lambda 模型卡（4×B200）；warmctl/dev.to《DeepSeek V4.1 Flash on 8 B300: vLLM vs DeepInfer》2026-10-21
- **读数声明（重要）**：仓库不发布 `sweeps/*/results.jsonl`，三张 Pareto 图的数据点是我按刻度读的（无原始 jsonl）。读数用"每 step token 数 = 4 GPUs × prefill_batch_size"做了自洽校验：五个 prefill 点的比值为 1.06/1.00/1.00/0.99/1.00，故读数误差 ≤5%（decoder 曲线无此校验，误差按 ±3% 计）。本笔记的所有派生的 sps、ms/step、比值都基于这些读数，**只用于趋势与量级，不用于小数点后比较**。

## 1. 三个最重要的发现

1. **README 的两张图不是实测吞吐，是"实测步频 × 假设接受率"的换算值。** `plot_pareto.py:59` 把 τ 定义为 `1 + nextn × a`，`:66-67` 把横纵两轴同时乘 τ；而 `benchmark.py:405` 的注释明说 `decode_avg_sps` 与接受率无关（VARLEN=0 时每 step 恒提交 1+nextn 个 token），`:408` 落盘的吞吐是"满接受口径"。也就是说图上每个点 = 实测 sps × 3.25，acceptance 75% 是**设定值**，不是测出来的。这一条决定了全书所有 decode 数字的性质：它们是容量估算，不是端到端实测。
2. **Wide-EP 买的是低并发区的交互性，不是峰值吞吐。** 同一 context/nextn 下，EP64 相对 EP8 的吞吐倍数随 batch 单调收敛：8 并发 1.53×、64 并发 1.27×、256 并发 1.05×；每用户速度 1.34× → 1.03×。而每卡专家数从 52 降到 7（MoE 权重访存理论削减 7.43×）。访存省了 7.4 倍、吞吐只多 5%–53%，说明 64 卡 EP 的瓶颈已经不在 HBM 带宽，而在 all2all/启动开销——这条推翻了"wide-EP 靠省访存换吞吐"的通俗解释。
3. **所有跨引擎数字至少在一个维度上不可比，且差异可达 2–10 倍。** 唯一四个维度都能对齐的是 vLLM 的 "6%"（GB200=SM100、131K、nextn=3、batch 8–256 与 DeepInfer sweep 完全同参），但它恰好缺 GPU 数与 TP/EP 拓扑；反过来，SGLang 的 873.6 tok/s（4×GB300、TP4、**模拟**接受长度 5.5）与 DeepInfer 的 2,600 tok/s/GPU（64×SM100、EP64、**假设**接受率 75%）放在同一张表里，任何相除都无意义。书中若要并列，必须先并置"口径八问"（§5.3）。

## 2. 事实条目

### #1 官方性能入口只有两张图，正文不给任何数字
- 证据：`README.md:77-83`
  > ## Performance benchmarks
  > | Prefill | Decode |
  > Measured on 4 SM100 GPUs for prefill and 64 for decode — each within one scale-up domain. To reproduce the benchmarks, see run_sweep.sh.
- 等级：A
- 意味着什么：全书所有"DeepInfer 多快"的数字都必须从图里读或从 QA 表里抄，没有一份官方文本给出"XX tok/s"的句子。写作时不能写"DeepSeek 表示达到 XX"。
- 疑点：README 未给 GPU 型号（只给架构代号 SM100）、未给 HBM 容量、未给软件版本、未给测量日期。

### #2 decode 曲线：EP64DP64 / 64 卡 / 131K / nextn=3 / 假设接受率 75%
- 证据：`doc/images/decode_pareto.png` 图内文字（tesseract 可直读）
  > `EP64DP64 | 384+64 experts | context 131,072 | nextn=3 fixed | per-token acceptance 75% → τ=3.25`
  > 纵轴 `Output Throughput (tokens/s/GPU)`（0–30,000），横轴 `Per-User Speed (tokens/s/user)`（0–350），点标注 `decode_num_requests (requests per DP rank)`
- 等级：A（图在仓库内；坐标值为读数）
- 意味着什么：这是全书 decode 章节的主图，六个点就是 `run_sweep.sh:36` 的 `(8 16 32 64 128 256)`。
- 疑点：横轴"每用户速度"含 τ，所以它不是 TTFT/ITL 实测，而是"假定接受率下的期望 token 率"。

### #3 decode 六个数据点（读数）与派生的步频
- 证据：同一张图；派生量按 τ=3.25 反算
- 等级：A（图）/ E（派生）
- 意味着什么：见下表。sps 是唯一"实测"的量，其余都是乘 τ 的结果。

| requests/rank | tok/s/user（图） | tok/s/GPU（图） | 反算 sps | 反算 ms/step | 反算 raw tok/s/GPU（τ=1） |
|---:|---:|---:|---:|---:|---:|
| 8 | 325 | 2,600 | 100.0 | 10.0 | 800 |
| 16 | 318 | 5,050 | 97.8 | 10.2 | 1,554 |
| 32 | 298 | 9,600 | 91.7 | 10.9 | 2,954 |
| 64 | 248 | 15,800 | 76.3 | 13.1 | 4,862 |
| 128 | 177 | 22,800 | 54.5 | 18.4 | 7,015 |
| 256 | 103 | 26,100 | 31.7 | 31.6 | 8,031 |

- 疑点：`raw` 一列是我把 τ 除掉的结果，**不是官方数字**，只用于说明"真实每步产出"与"图上产出"差 3.25 倍。

### #4 prefill 曲线：EP4DP4 / 4 卡 / 131K / 带 Decoder Bounded Replay(140)
- 证据：`doc/images/prefill_pareto.png` 图内文字
  > `EP4DP4 | 384+32 experts | context 131,072 | Decoder Bounded Replay (140 tokens)`
  > 横轴 `Latency (ms / prefill step)`（0–800），纵轴 `Non-cached Input Throughput (tokens/s/GPU)`（0–100,000），点标注 `prefill_batch_size (tokens per DP rank)`
- 等级：A / 读数
- 意味着什么：官方 prefill 数字**已经含了 YoCo 截断的收益**（layers 21–39 只算末尾 140 token），拿它和没开 replay 的引擎比，等于让对手背一半的计算。
- 疑点：`run_sweep.sh:47` 传 `--yoco_clip`（默认 140），图注的 "140 tokens" 即此；README 没提这一点。

### #5 prefill 五个数据点（读数），且与"每步 4×batch token"自洽
- 证据：同图；`run_sweep.sh:45` 的扫描值 `(4096 8192 16384 32768 65536)`；`benchmark.py:394` 的 `prefill_tokens_per_s_per_gpu = window_tokens/window_seconds/cp_size`
- 等级：A（图）/ A（自洽校验）
- 意味着什么：

| batch/DP | 延迟 ms/step | tok/s/GPU | 反算 tokens/step | 4×batch | 比值 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 72 | 60,300 | 17,366 | 16,384 | 1.06 |
| 8,192 | 115 | 71,500 | 32,890 | 32,768 | 1.00 |
| 16,384 | 201 | 81,800 | 65,767 | 65,536 | 1.00 |
| 32,768 | 372 | 86,800 | 129,158 | 131,072 | 0.99 |
| 65,536 | 731 | 89,300 | 261,113 | 262,144 | 1.00 |

- 疑点：4/5 个点比值 1.00，说明我读的坐标基本正确；第 1 点偏高 6%，说明 72 ms 应为 ~76 ms，误差 5%。

### #6 prefill 的拐点：8K–16K 之后饱和
- 证据：上表派生
  > 4,096→8,192：吞吐 +18.6%，延迟 +59.7%；8,192→16,384：+14.4%，+74.8%；16,384→32,768：+6.1%，+85.1%；32,768→65,536：+2.9%，+96.5%
- 等级：E（基于读数）
- 意味着什么：官方曲线在 87–89K tok/s/GPU 附近饱和，batch 再大一倍只多 2.9% 吞吐却多付一倍延迟。书里讲"prefill 是算子效率问题不是 batch 问题"可以直接用这四个边际数。
- 疑点：QA:115 明确"每个 prefill step 取 `q` 使 `q` 不超过 **32,768-token 预算**"，而扫描点给到 65,536（`max_num_batched_tokens`），是预算的两倍。因此该点要么被引擎切成 2 个 chunk，要么 benchmark 的 `--prefill_batch_size` 与生产 scheduler 的 `q` 不是同一个量；"per prefill step" 到底指一个 chunk 还是一个调度步，图注没定义。

### #7 Wide-EP 图：EP8DP8 vs EP64DP64，同 context/nextn/τ
- 证据：`doc/images/wide_ep_decode_pareto.png`；QA:55-57
  > 图例 `EP64DP64 | 384+64 experts` / `EP8DP8 | 384+32 experts`；副标题与 decode 主图相同
  > QA:57 `Wide-EP substantially reduces per-step memory traffic in the MoE expert part and, with good EPLB support, effectively raises TPS. The chart below shows EP8 vs. EP64 benchmark data (offline benchmark script, configuration as shown).`
- 等级：A / 读数
- 意味着什么：官方只用一句定性话解释 wide-EP（"显著降低每步访存"），**全文没有任何访存流量的数字**；能拿到的只有两条曲线的读数差。
- 疑点：两张图的 `384+64` 与 `384+32` 说明两组的冗余专家数不同（EP8 组是默认 32，EP64 组被向上取整成 64），所以这不是"只改 EP"的对照实验，专家总数从 416 变 448。

### #8 每卡专家数 7 vs 52：访存差的唯一可算来源
- 证据：QA:103
  > Settings: start from 1 redundant expert per GPU (rounded up so that the total expert count is a multiple of ep_size) — at EP64, for example, this works out to 7 experts per GPU after rounding, excluding the shared expert
  `examples/README.md:63`
  > The number of redundant experts is passed through DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS, defaulting to 32. This env var is only a lower bound: the engine rounds num_routed_experts + num_extra_experts up so that it is divisible by ep_size
- 等级：A
- 意味着什么：EP64 → (384+64)/64 = **7**；EP8 → (384+32)/8 = **52**；比值 **7.43×**。这是"访存流量"唯一能从官方材料严格算出的量（按每专家参数量相同、精度相同计）。
- 疑点：这是权重访存，不含 KV 访存与 all2all 流量；QA 说的 "memory traffic" 到底含不含 KV、含不含通信 buffer，未说明。

### #9 EP8 vs EP64 六点读数与收敛曲线
- 证据：`wide_ep_decode_pareto.png` 读数
- 等级：A / 读数（±3%）
- 意味着什么：

| batch | EP8 tok/s/user | EP8 tok/s/GPU | EP64 tok/s/user | EP64 tok/s/GPU | 吞吐倍数 | 速度倍数 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 243 | 1,700 | 327 | 2,600 | 1.53× | 1.34× |
| 16 | 233 | 3,350 | 319 | 5,050 | 1.51× | 1.37× |
| 32 | 222 | 6,900 | 298 | 9,600 | 1.39× | 1.34× |
| 64 | 200 | 12,450 | 248 | 15,800 | 1.27× | 1.24× |
| 128 | 150 | 19,100 | 177 | 22,800 | 1.19× | 1.18× |
| 256 | 100 | 24,850 | 103 | 26,100 | 1.05× | 1.03× |

- 疑点：EP64 的 256 点略高于纵轴最高刻度 25,000（我的读数 26,100），若读数偏低/偏高 5%，1.05× 的结论仍在 1.00–1.10 区间内。

### #10 decode 的边际收益拐点：64 并发之后陡降
- 证据：#3 派生
  > 8→16：吞吐 +94.2%，每用户速度 −2.2%；16→32：+90.1%，−6.3%；32→64：+64.6%，−16.8%；64→128：+44.3%，−28.6%；128→256：+14.5%，−41.8%
- 等级：E
- 意味着什么：曲线的"膝点"在 16–32，"肩点"在 64；128→256 已经进入"用一半交互性换 14% 吞吐"的区间。书里讲"生产为什么跑在 ~60 并发/rank"（QA:247）时，这张边际表就是量化依据。
- 疑点：边际收益同时受 EPLB/通信影响，非纯排队效应。

### #11 DSpark 变长验证表：完整六行，−1.5% → +11.0%
- 证据：`doc/inference_deployment_qa.md:249-256`
  > | conc (seqs/rank) | fixed-5 avg verified length | dynamic-length avg verified length | fixed-5 accepted length | dynamic-length accepted length | fixed-5 estimated throughput (tok/s/rank) | dynamic-length estimated throughput (tok/s/rank) | dynamic vs. fixed-5 gain |
  > | 16 | 5 | 4.44 | 3.32 | 3.24 | 4891 | 4816 | −1.5% |
  > | 60 | 5 | 3.58 | 3.3 | 3.21 | 13730 | 14466 | +5.4% |
  > | 96 | 5 | 3.24 | 3.3 | 3.15 | 16817 | 18660 | +11.0% |
- 等级：B（官方文档）
- 意味着什么：这是全书唯一一张"官方给出逐点数字"的性能表，且单位是 `tok/s/rank`（不是 /GPU，但本合同构）。
- 疑点：列名写 "**estimated** throughput"，是估算不是实测；表中只有 6 个点、没有硬件型号、没有说明 τ 怎么算。

### #12 "约 30%" 的口头结论比表更乐观
- 证据：QA:247 与 #11 的六行平均
  > Dynamic-length verification cuts the number of draft tokens verified per step by about 30% (5.00 → 3.2–4.4) while leaving the average accepted length essentially unchanged (3.29 vs. 3.20). … +4–7% throughput at the production operating point (~60 concurrent sequences per rank).
  六行平均：dynamic verified = **3.68**（削减 **26.4%**）；fixed accepted **3.293** vs dynamic **3.192**
- 等级：E（用官方表算）
- 意味着什么：正文"saves 30%"略高于自己表里的 26.4%；"unchanged (3.29 vs 3.20)"实际是 3.29 vs 3.19，且逐行看 dynamic 的接受长度**每一行都更低**。书里应写"约 26%（表内均值）/ 官方口径 ~30%"，并注明 +4–7% 只在 48–72 并发成立（16 并发是 −1.5%，96 并发是 +11.0%）。
- 疑点：知乎把这段压缩成"线上吞吐提升 6%"，落在 60 并发的 +5.4% 上（R01 #14）。

### #13 负载均衡的四个官方数字互相打架
- 证据：QA:99、QA:103、QA:127
  > Prefill … Measured moe_lb (slowest rank / mean) = **1.074 (EP4DP4) / 1.045 (EP2DP2)**
  > Decode … Measured moe_lb = **1.090**
  > we reach roughly **80%** of a perfectly balanced offline benchmark
  知乎（R01 #15 / R12 #23）：**1.0320 (EP2/DP2) / 1.0618 (EP4/DP4)**
- 等级：B（两处都是官方）
- 意味着什么：同一指标 EP4DP4 有 1.074 与 1.0618 两个值；EP2DP2 有 1.045 与 1.0320。书里只能写区间或注明"官方两版不一致"。`80%` 是唯一的总收益数字，但没说是吞吐还是有效吞吐。
- 疑点：decode 的 1.090 没标 EP 配置，只与 "at EP64" 的例子相邻。

### #14 生产规模数字：155 万亿 tokens/天、$38/卡·小时
- 证据：QA:17
  > A single cluster currently serves 155 trillion tokens per day, with overall revenue of US$38 per GPU-hour (at valley pricing).
- 等级：B
- 意味着什么：全书唯一的营收/规模锚点，引用必须带"单集群""谷价"两个限定；不能除以 GPU 数反推单卡吞吐（集群里含 prefill、VE、空闲余量）。
- 疑点：与 vLLM X 的 "$39/gpu/hr" 只差 1 美元，容易被误读为"同一口径的对比"，实际一个是 DeepSeek 自家谷价收入、一个是 vLLM 在 AgentX 上按同一价目折算的推算值。

### #15 部署形态决定了所有数字的"系统边界"
- 证据：QA:46-52
  > Prefill: EP2–8 / DP2–8; use the smallest EP/DP that HBM allows.
  > Decode: EP64–128 / DP64–128 (Wide-EP); use the largest EP/DP that the available GPUs allow.
  > **TP and CP are not used for now.**
  `README.md:38-42`：KV cache 系统、VE/Prefill/Decode 分离、DP 副本间负载均衡"not included in this release"
- 等级：A/B
- 意味着什么：官方曲线的"64 卡 decode"是**没有 KV cache 系统、没有 P/D 分离、没有跨副本 LB、TP=1/CP=1** 的裸引擎数字。任何"端到端服务吞吐"都不能与它直接比。
- 疑点：README 说 P/D 分离"most required engine functionality is already implemented"，但 benchmark 脚本没有走 P/D 路径。

### #16 开源版与内部版的差距：kernel 5–10%，端到端 ≤5%
- 证据：QA:235-237
  > FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower.
  > End-to-end performance is close — at most 5% apart.
- 等级：B
- 意味着什么：这是官方自曝的"复现折扣"：任何人拿开源 DeepInfer 复现官方曲线，先天的差距上限是 5%。书里评估第三方复现数据时应先扣这 5%。
- 疑点：只提 FlashMLA，没提 Mega MoE / MegaAttention 是否有同类差距。

### #17 vLLM "6%" 的原文与四个条件
- 证据：`https://x.com/vllm_project/status/2112703311933747347`（2026-10-21T00:31:36Z）
  > Under DeepSeek's decode configuration (DeepSeek-V4.1-Flash, 131K context, nextn=3), vLLM on GB200 achieves throughput within 6% of DeepInfer's published results across batch sizes 8–256.
  > With Wide-EP and EPLB, vLLM can match DeepInfer's decode performance within 6% and generate $39/gpu/hr in revenue at DeepSeek's off-peak pricing, at 200 P90 TPS on the AgentX workload.
- 等级：B（官方账号一手发言）
- 意味着什么：batch 8–256 与 `run_sweep.sh:36` 的六个点逐一对应，说明 vLLM 是照着 DeepInfer 的复现脚本参数对标的，不是自选负载。
- 疑点：**未给 GPU 数、未给 TP/EP/DP 拓扑、未说 6% 是高还是低、未说 acceptance 口径**。这四项缺一项，per-GPU 对比就不成立。

### #18 vLLM 自己的 TTFT 表：GB200、单请求、关 prefix cache
- 证据：vLLM 博客 2026-10-07（Figure 4 文字）
  > Median of 3 runs on NVIDIA GB200; prefix caching off. Percentages relative to replay off.
  > TP4：1K 160.9 ms（+3.1% / −35.8%），4K 165.4 ms（+4.3% / −35.3%），8K 170.0 ms（+4.6% / −33.8%），32K 540.1 ms（−30.4% / −39.3%），128K 2,267 ms（−32.9% / −38.6%）
  > DEP2：1K 103.8 ms（+11.9% / −30.2%），4K 135.3 ms（−13.2% / −31.8%），8K 246.9 ms（−33.5% / −35.4%），32K 983.1 ms（−37.2% / −38.0%），128K 3,950 ms（−36.3% / −37.7%）
- 等级：B
- 意味着什么：这是全书唯一"带绝对毫秒值"的 prefill 延迟表，可用来说明 SWA bounded replay 的收益与代价；注意 TP4 与 DEP2 是两条不同拓扑的曲线，同一模型同一硬件差 1.7×（128K：2,267 vs 3,950 ms）。
- 疑点：单请求、关 prefix cache、median of 3——与"生产 prefill 吞吐"完全不同的测量场景。

### #19 vLLM 的"负收益区间"：没有 CUDA graph 时 replay 更慢
- 证据：同博客
  > without them, launch overhead outweighs the GPU savings and replay is slower than the baseline (up to +12% at 1K on DEP2)
  > Decoder replay with CUDA graphs cuts prefill computation time by 30–40%.
- 等级：B
- 意味着什么：这是全书唯一一条"优化在短 prompt 上反向变慢"的证据，适合放在"优化的适用边界"一节。
- 疑点：+12% 只出现在 DEP2/1K；TP4/1K 是 +3.1%。

### #20 SGLang 单流 873.6 tok/s 是**模拟**接受长度 5.5
- 证据：`https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization`
  > BS=1 **873.6** tokens/s，measured accept length **5.505**
  > PLAIN DECODE / DSPARK · RANDOM 4K/1K · **SIMULATED ACCEPT LENGTH 5.5**
  > Acceptance is simulated, so the generated text is not used to judge answer quality; simulation mode also disables the in-graph acceptance path.
  > 复现环境：4× GB300，attention TP4，`SGLANG_SIMULATE_ACC_LEN=5.5`
- 等级：B
- 意味着什么：4×GB300 上单流 873.6 tok/s = 218.4 tok/s/GPU；但它不是"实测接受率下的真实吞吐"，而是"把接受长度钉在 5.5"的工程上限。书里引用必须写"模拟接受 5.5"。
- 疑点：模拟接受长度 5.5 与 DeepInfer 图的 75%/τ=3.25 不是同一口径（见 §3.6）。

### #21 SGLang 的 DSpark 分解：步频降 30%，产出涨 3.8×
- 证据：同博客表
  > DSpark off · EP4：BS=1 **223.50** tok/s；DSpark on · EP4：**853.49**（accept 5.520）；DSpark on · TP4+padding：**873.63**（accept 5.505）
  > EP4 gives 854.64 tokens/s and TP4 gives 873.63 tokens/s, a 2.22% gain
- 等级：B
- 意味着什么：223.5 → 853.5 是 3.82×，但反算步频：223.5 sps（1 token/step）→ 154.6 sps（5.52 token/step），**每步慢了 1.45×**。所以 "tok/s" 的涨幅里，一部分被 verify 的开销吃掉。这是解释"投机解码不是免费的 5 倍"的最好例子。
- 疑点：TP4 vs EP4 只差 2.22%，但 rank 间到达 finalize 的间隔从 11.14 µs 降到 3.40 µs——说明瓶颈是"最慢 rank"，不是平均带宽。

## 3. 必答问题逐条回答

### 1. README 中 prefill 与 decode Pareto 图的详细数据点、坐标轴、硬件、并发档位、关键拐点；`run_sweep.sh` 的测试参数与方法

**坐标系与硬件**（#1-#5）
- decode：横轴 Per-User Speed (tokens/s/user) 0–350，纵轴 Output Throughput (tokens/s/GPU) 0–30,000。配置 `EP64DP64 / 384+64 experts / context 131,072 / nextn=3 / acceptance 75% → τ=3.25`，**64×SM100**。
- prefill：横轴 Latency (ms/prefill step) 0–800，纵轴 Non-cached Input Throughput (tokens/s/GPU) 0–100,000。配置 `EP4DP4 / 384+32 experts / context 131,072 / Decoder Bounded Replay (140 tokens)`，**4×SM100**。
- wide-EP 对照图：同 decode 坐标系，两条线 `EP64DP64 | 384+64` 与 `EP8DP8 | 384+32`。
- "SM100"= Blackwell 第一代（GB200/B200 级），**不是 GB300（SM103）**；README 只给架构代号，未给具体 SKU。

**并发档位**（#2、#3）
- decode 六档：8/16/32/64/128/256，单位是"每个 DP rank 的请求数"。因为 DP=64，全集群并发 = 档位 × 64（256 档 = 16,384 请求）。
- prefill 五档：4,096/8,192/16,384/32,768/65,536，单位是"每个 DP rank 的 batch token 数"，全集群单步 = ×4。

**拐点**（#6、#10）
- decode：16–32 是膝点（+90% 吞吐只掉 6% 速度），64 是肩点，128 之后进入平台（+14.5% 吞吐换 −41.8% 速度）。
- prefill：8K–16K 是膝点（+14.4% 吞吐 / +74.8% 延迟），32K 之后饱和（+2.9% / +96.5%），渐近线 ~89–90K tok/s/GPU。

**`run_sweep.sh` 的参数与方法**（`run_sweep.sh:29-49,68-89`；`examples/README.md:51-56`）
- 一维扫描，一个点起一次 `benchmark.py`，全部 `PROFILE=0`（profiler 会扰动稳态），每点换一个 `MASTER_PORT`（避免 TIME_WAIT 卡 rendezvous）。
- decode：`DEEPINFER_USE_NEXTN_DECODE=<nextn>`，固定 `--context_length <ctx> --non_cached_tokens 4096`，扫 `--decode_num_requests`。
- prefill：固定 `--context_length <ctx> --yoco_clip`（默认 140），扫 `--prefill_batch_size`。
- 每点记录：`prefill_window_steps/tokens/seconds`、`decode_avg_sps`、`decode_window_steps` 等；出图用 `plot_pareto.py`。
- 测量窗：prefill 跳 5 个 warmup chunk，取 10 个稳态区间；decode 跳 50 步 warmup + 60 步 logged，取 60 个满 batch 区间；与 MoE LB flush 相撞的区间被丢弃。
- 吞吐定义：prefill = `window_tokens/window_seconds/cp_size`；decode = `avg_sps × num_requests × (1+nextn)/cp`（**满接受口径**，与真实接受率无关，`benchmark.py:408`）。
- 出图时再乘 τ=1+nextn×a（`plot_pareto.py:59,66-67`），所以**图 ≠ jsonl 里的 tokens/s**：同一个点，jsonl 记的是 ×4，图上画的是 ×3.25。

**方法学缺口**
- 仓库不发布 `results.jsonl`，无法复核图上的点；`ve_sweep.py` 只测 VE，且明确 `--dp 1` 即够（编码是单卡计算）。
- benchmark 强制 `DEEPINFER_FORCE_LOAD_BALANCE=1`、`DEEPINFER_MAX_BACKTRACKING_TOKENS=1`、`sampler_force_topk=1024`，这些都不是"默认生产配置"，而是为了让扫描可复现。
- 没有 prefix cache 命中、没有 P/D 分离、没有请求到达过程（offline，全部请求一次性 admit）。

### 2. 部署 Q&A 中 EP8 vs EP64 的访存流量与吞吐对比、DSpark 变长验证表

**访存流量：官方只给定性，没有数字。**（#7、#8）
- 原文只有一句："Wide-EP substantially reduces per-step memory traffic in the MoE expert part"（QA:57），全篇 `grep -i "traffic\|bandwidth"` 无第二个定量句。
- 可严格算出的只有"每卡专家数"：EP64 → 7，EP8 → 52（默认冗余 32，向上取整到 64），**7.43×**。若按 FP4 权重 0.5 B/param 计，MoE 权重访存从 ~34.5 GB/step 降到 ~4.3 GB/step（552B/8 vs 552B/64，E 级估算，假设专家参数均匀）。
- **不能算的**：KV 访存（131K × 890 B × batch，与 EP 无关）、all2all 流量（随 EP 增大而增大）、EPLB 复制的额外权重。

**吞吐对比**（#9）：六点见 #9 表。结论三句：
1. 低并发（8）EP64 是 EP8 的 1.53×（吞吐）/ 1.34×（每用户速度）；
2. 高并发（256）收敛到 1.05× / 1.03×——EP8 已达到 EP64 的 95% 峰值吞吐；
3. 因此 wide-EP 的收益是"同样交互性下少用卡 / 同样卡数下更高交互性"，不是"峰值吞吐高 50%"。
- 访存省 7.43× 而吞吐只多 1.05–1.53×，说明 EP64 的步时间由 all2all/启动开销主导：batch=8 时 EP64 的步时间 10.0 ms 是"权重访存下界"（4.3 GB @ 8 TB/s ≈ 0.54 ms）的 **18.6 倍**，EP8 的 13.4 ms 只是其下界（4.31 ms）的 **3.1 倍**（E 级估算，8 TB/s 按 B200 HBM 规格假设）。

**DSpark 变长验证表**（#11、#12）：完整六行见 #11；增益曲线 −1.5% → +1.6% → +3.6% → +5.4% → +6.7% → +11.0%，单调上升（16 并发为负）。平均验证长度 5.00 → 3.68（−26.4%），接受长度 3.293 → 3.192（−3.1%）。官方口径"+4–7% at ~60 conc"与表内 60 并发 +5.4% 一致；知乎的"6%"就是这一格的取整（R01 #14）。

### 3. vLLM "解码吞吐在 DeepInfer 6% 以内"的数据细节

- **出处**：vLLM 官方 X `@vllm_project`，2026-10-21T00:31:36Z，status 2112703311933747347；**不是博客**（vLLM 博客索引里 `DeepInfer`/`6%` 命中 0，R13 #32）。
- **硬件**：`vLLM on GB200`。GB200 = SM100，与 DeepInfer 曲线的架构代号一致——这是全书唯一"硬件代际对得上"的跨引擎对比。**GPU 数未公布**；对照侧是 64×SM100。
- **批次**：batch sizes 8–256，与 `run_sweep.sh:36` 的六个点逐一对应（每 DP rank 的 `decode_num_requests`）。
- **基线**：DeepInfer **公布曲线**（`decode_pareto.png`），即 64 卡 / EP64DP64 / 131K / nextn=3 / τ=3.25 的那 6 个点；不是同机 A/B。若 vLLM 低 6%，对应 2,444/4,747/9,024/14,852/21,432/24,534 tok/s/GPU；若高 6%，对应 2,756/5,353/10,176/16,748/24,168/27,666（E 级算术，仅示范量级）。
- **前置条件**：Wide-EP + EPLB；已集成 MegaAttn/MegaMoE/Mega-mHC 等 DeepSeek 内核；nextn=3（注意 vLLM 配方默认 5）。
- **缺失参数（六个）**：① GPU 数；② TP/EP/DP 拓扑与 EP 规模；③ acceptance 从哪来（实测/假设/模拟）、τ 怎么算；④ "throughput" 是 output token 还是 total token；⑤ 6% 的方向；⑥ vLLM 版本与测量日期（推文只给 InferenceX 面板链接）。
- **同一条推文里的另外两个数字**：`$39/gpu/hr`（DeepSeek 谷价、200 P90 TPS、AgentX 负载）、以及 R13 记录的 `826–829 TPS per user`。后者物理上不可能是单用户解码速度（552B 模型、1.2 ms/token），我未能核实其定义，**建议全书不用**（见 §4）。
- **不能说的**：不能说"同机差 6%"、不能说"vLLM 快 6%"、不能外推到单机（warmctl 单节点方向相反，2.0–4.3×，见 §3.4 的 T4/T5）。

### 4. SGLang / 独立第三方的测试数据

**SGLang 官方（7 处，全部 B 级）**
| 场景 | 数字 | 条件 |
|---|---|---|
| 单流 decode 工程上限 | **873.63 tok/s**（218.4/GPU），accept 5.505 | 4×GB300，attn TP4 + MoE TP4，random 4k/1k，**模拟**接受 |
| DSpark 关闭/开启 | 223.50 → 853.49 tok/s（3.82×） | 同上，MoE EP4 |
| MoE EP4 vs TP4 | 854.64 → 873.63（+2.22%），rank 到达间隔 11.14 → 3.40 µs | 同上 |
| 低延迟插件（AgentX trace） | P90 每用户 **671.2 → 928.4 tok/s（1.38×）** @1 并发；MI355X 上 **331.6 → 516.0（1.56×）** @4 并发 | 4×GB300 / 4×MI355X，DSpark |
| prefill bounded replay | **1.37×（4×GB300）/ 1.56×（8×H200）**，8×8K prompt 成对测试；AIME 2026 pass@1 相同（453/480） | SGLang 自家实现 |
| Engram 放 host | KV 容量 **+36%**，decode 吞吐与 TTFT 持平 | 4×GB300，TP4/EP4 |
| Day-0 单流 | **1,130 tok/s/user / 221 tok/s/GPU** | 4×GB300，bs=1，EP1，4K/1K |

**SemiAnalysis InferenceX（B 级数字，E 级口径）**
- GB300 NVL72：50 tok/s/user → **359,012 tok/s/GPU**；75 → 285,907（SGLang）；100 → 267,304（SGLang）；150 → 165,938（SGLang）；200 → **471,578**（Dynamo vLLM）；峰值 **1,011,514 tok/s/GPU**。
- B300：50 → 328,417；75 → 376,639；100 → 315,201；150 → 138,983；200 → 112,649；峰值 406,374。
- B200：75 → 222,682；100 → 178,100（30/50 tok/s 为空）。
- **这组数字有两个硬问题**：① 非单调（GB300 的 150 → 165,938，200 → 471,578，2.8×；B200 的 75 → 222,682 高于 100 → 178,100）。物理上"交互性更高 ⇒ 吞吐更低"，同一条前沿不可能出现这种跳变，说明它是**每个交互档取不同配置的最优值**，不是一次运行的一条曲线，跨档相减/相除无意义。② 359K tok/s/GPU 若按 output token 计，等于 11.5 PFLOPS/卡（16B 激活 × 2 FLOP × 359K），超过任何 FP4 卡的实用 MFU；只有按"total token（含 prefill、可能含 prefix 命中）"计才可能成立。页面自始至终没定义 token 计数口径。

**独立第三方单机实测（C–D 级）**
- Lambda（4×B200，vLLM TP4，DSpark 5 + adaptive verification，8K in/1K out，512 prompts，AIPerf）：C1 **374.8 tok/s**（410.8 tok/s/user）；C16 2,314.7（156.1）；C64 3,713.9（66.8）；C256 **4,858.7**（29.6）。折每卡：93.7 / 578.7 / 928.5 / **1,214.7 tok/s/GPU**。
- warmctl/dev.to（8×B300，vLLM 0.31.1rc1 DEP8 vs DeepInfer 8b16149 EP8×DP8，均 DSpark 5，随机 prompt）：
  | 场景 | DeepInfer | vLLM | 倍数 |
  |---|---:|---:|---:|
  | 1K→1K, 2048 并发 | 27,906 | 54,985 | 2.0× |
  | 1K→1K, 4096 并发 | 26,294 | 55,506 | 2.1× |
  | 8K→1K, 2048 并发 | 6,933 | 29,500 | 4.3× |
  | 32K→1 prefill, 512 并发 | 64,847 | 589,924 | 9.1× |
- **warmctl 最关键的一句**：随机 prompt 下 draft 接受率只有 **19%（1K）/ 12%（8K）**，所以"每个 decode 数字都是地板"。这与 DeepInfer 图假设的 75% 相差 4–6 倍，是全书解释"为什么公开曲线好看、单机实测难看"的核心证据。
- 无 SGLang 侧的第三方 DeepInfer 对比数据；GPUStack 的《DeepInfer vs. vLLM on 8×B200》只读摘要（R13 #35），未取得数字。

### 5. 全书唯一的标准性能数据大表

见 **§5.1**。字段：数据指标 / 数值 / 硬件环境 / 模型配置 / 并发负载 / 数据来源 / 证据等级。所有其他章节的性能表都应是它的子集或视图，不得另立口径。

### 6. 深度论证：为什么这些数字不能直接横向比较（下分 6.1–6.7 七个论证点）

**6.1 硬件代际与 SKU：SM100 ≠ SM103 ≠ MI355X**
- DeepInfer 曲线只写 "SM100"（#1），即 GB200/B200 一代；vLLM 6% 用 GB200（可对齐）；但 SGLang 的两组王牌数字是 **4×GB300**（SM103），InferenceX 的主力是 **GB300 NVL72**，AMD 侧是 **MI355X**。
- 同一模型在 TP4 下，GB200 的 128K TTFT 是 2,267 ms，而 GB300 上的吞吐又是另一套（#18、§3.4）。跨代并列等于把"制程+带宽+容量"三重差异塞进一个数字。
- 书里写法：任何表格必须有一列"架构代号"，SM100/SM103/MI355X 三者不能合并成"NVIDIA GPU"。

**6.2 TP vs EP 拓扑：per-GPU 的分母根本不是同一份工作量**
- DeepInfer 生产与官方曲线：**TP=1、CP=1**，EP64–128（QA:52）。vLLM 低延迟与 Lambda：**TP4**；SGLang：attn TP4 + MoE TP4。warmctl：DEP8/EP8。
- 每卡权重（552B/卡数）：EP64 → **8.6B**；TP4 或 EP4 → **138B**；EP8/TP8 → **69B**。按 FP4 0.5 B/param：4.3 GB vs 69 GB，**16×**。
- 所以"每卡 tok/s"里的每卡，在 EP64 下只需读 4.3 GB 权重，在 TP4 下要读 69 GB。**同一 tok/s/GPU 在两种拓扑下代表完全不同的效率**：EP64 的 26,100 tok/s/GPU 若换算成"每 GB 权重产生的 token"，是 TP4 数字的十几倍。
- 更关键的是瓶颈位置不同：EP64 在 batch=8 时的步时间是权重访存下界的 18.6 倍（通信/启动主导），EP8 只有 3.1 倍（接近带宽主导）。这解释了两件事：为什么 wide-EP 省 7.43× 访存只换来 1.05–1.53× 吞吐（#9），以及为什么 TP4 的单流数字（SGLang 873 tok/s、Lambda 374.8 tok/s）不能拿来否定 EP64 的高并发数字。
- 书里写法：比较前先问"TP 几、EP 几、DP 几"，三者任一不同，per-GPU 数字不可并列。

**6.3 负载分布：context、batch 形状、输出长度、是否 offline**
- DeepInfer decode：131K context（`--non_cached_tokens 4096`，即 126,976 已缓存 + 4,096 非缓存后缀），offline、全部请求同时 admit、固定输出长度、无到达过程。
- Lambda：8K in / 1K out、512 prompts。SGLang：4K/1K、BS=1。warmctl：1K→1K / 8K→1K。InferenceX：AgentX 多轮 agent trace。
- context 从 4K 到 131K，KV 读放量差 32×（890 B/token × 长度）；输出从 1 到 1,024，摊销完全不同；offline 无排队，等效于"永远满 batch"。
- 书里写法：把"131K/offline/固定输出"作为 DeepInfer 曲线的固有前缀，不能与"4K/1K 在线服务"同表。

**6.4 前缀命中率：token 计数器可以是三样东西**
- 三种口径混用：(a) **output token**（decode 产出，DeepInfer 图、SGLang、Lambda 都是这个）；(b) **total token = input + output**（warmctl 的 `Total tok/s` 列、InferenceX 很可能）；(c) **含 prefix cache 命中的 token**（成本≈0，但可以被计入"吞吐"）。
- 证据：InferenceX 的 359K tok/s/GPU 只有在 (b)/(c) 下才物理成立（§3.4）；warmctl 的同一组实测里三种口径同时给出：1K→1K 2048 并发 output **54,985** / total **109,969**（2.0×）；8K→1K output 29,500 / total **265,496**（9.0×）；2K→128 output 19,753 / total **335,801**（**17.0×**）。input 越长、输出越短，两个口径差得越离谱。
- DeepInfer 的 benchmark 明确"无 prefix cache"（offline、每请求独占前缀）；生产有 3 天/7 天 TTL 的高命中（R01 #12）。所以官方曲线是"无缓存红利"的下界，而一旦把命中 token 计入吞吐就能报出任意高的上界——同一引擎两个方向都能"变快"。
- 书里写法：任何 tok/s 必须标注 `output` / `total` / `with-cache-hit` 三者之一。

**6.5 投机 MTP：接受率是最大的单一杠杆，而四个玩家用了四种口径**
- 公式：`输出 tok/s = 步频(nextn, batch) × τ(acceptance)`。两个因子都会动。
- 四种口径：
  1. DeepInfer 图：nextn=3，acceptance **假设 75%**，τ=1+3×0.75=**3.25**（#2、#3）；
  2. DeepInfer 生产：nextn=5，**实测**接受长度 3.29（= 逐 token 0.75 的几何期望 3.288，见下）；
  3. SGLang：nextn=5，接受长度 **模拟钉死 5.5**（#20）；
  4. Lambda/vLLM：nextn=5 + adaptive verification，实测接受率未公布；warmctl 随机 prompt 实测 **19%/12%**（§3.4）。
- **DeepInfer 的 τ 是线性近似，不是期望值**：若各 draft 位置独立接受，期望接受长度 = Σa^i。a=0.75、n=3 → **2.73**，而图用 **3.25（高 19%）**；a=0.75、n=5 → **3.288**，恰好等于 QA 实测的 3.29–3.32。也就是说：生产实测证明"逐 token 0.75 的独立几何模型"是对的，那么 nextn=3 的正确 τ 应是 2.73，图的 3.25 偏乐观约 19%（除非接受过程完全相关，才会趋近线性上界）。
- **杠杆有多大**：以 256 并发点（实测 sps=31.7）为例，只改接受率假设：
  | 假设 | 线性 τ | 每用户 tok/s | 每卡 tok/s | 几何 τ | 每用户 | 每卡 |
  |---|---:|---:|---:|---:|---:|---:|
  | a=0.75（官方图） | 3.25 | 103 | 26,368 | 2.73 | 87 | 22,185 |
  | a=0.50 | 2.50 | 79 | 20,283 | 1.88 | 59 | 15,212 |
  | a=0.19（warmctl 1K） | 1.57 | 50 | 12,738 | 1.23 | 39 | 10,003 |
  | a=0.12（warmctl 8K） | 1.36 | 43 | 11,034 | 1.14 | 36 | 9,218 |
  **同一份实测步频，换个接受率假设，报出来的吞吐差 2.6 倍。** 这就是为什么"DeepInfer 26,100 tok/s/GPU"和"Lambda 1,215 tok/s/GPU"不能相减（§6.2 的 16× 权重差 + 本节 2.6× 的 τ 差，已经解释了绝大部分 21× 的表观差距）。
- **SGLang 的分解**（#21）：DSpark 关闭 223.5 tok/s（1 token/step，sps=223.5）；开启后 853.5（5.52 token/step，sps=154.6）。产出涨 3.82×，但**步频降了 1.45×**。所以"投机解码带来 3.8×" 里，verify 的开销已经扣掉；而 DeepInfer 的图把 nextn 固定为 3，图上的点**不含**"nextn 从 3 加到 5 会让步频下降"这一项。
- 书里写法：任何 decode 数字必须写 `(nextn, acceptance, τ, 来源: 实测/模拟/假设)` 四元组；缺一项就是不可比。

**6.6 度量对象与时间窗：tok/s/GPU、tok/s/user、P90、TTFT、TTFT-vs-throughput 前沿**
- DeepInfer：稳态 `avg_sps`（60 个区间，剔除 LB flush），再乘 τ → 是"容量/期望值"，不含冷启动、不含排队、不含 P99。
- SGLang 低延迟：**P90 每用户速度**（671.2 → 928.4），是分位数不是均值；且 AgentX trace 有到达过程。
- vLLM 博客：TTFT 绝对毫秒（median of 3，单请求，关 prefix cache）与 AgentX 前沿（1.9×/5.3×）两套完全不同的量。
- InferenceX：**每个交互档取不同配置的最优值**拼成阶梯（非单调已证），是"前沿包络"，不是一条曲线。
- Lambda：AIPerf 均值，512 prompts/点，含 TTFT/ITL 分位数。
- 把 P90 与均值、把前沿包络与单点、把 TTFT 与 decode 吞吐放进一张表，等于用不同单位的尺子量同一根棍子。

**6.7 系统边界：64 卡裸引擎 vs 单节点全栈 vs NVL72 整机**
- DeepInfer 曲线：64 卡在一个 scale-up domain 内，**没有** KV cache 系统、没有 P/D 分离、没有跨副本 LB（#15）。
- vLLM 6%：Wide-EP + EPLB，但同样是对"公布曲线"，不是生产链路。
- warmctl：8 卡单节点、两个引擎都"什么都自己干"，DeepInfer 因此慢 2.0–4.3×（§3.4 的 T4/T5）——**这不是引擎优劣，是拓扑不匹配**：DeepInfer 没有 TP 路径，8 卡上只能 CP1/DP8/EP8，而 vLLM 的配方就是为单节点设计的。
- InferenceX：GB300 NVL72 整机（72 卡）+ P/D 分离 + Dynamo/router。
- 书里写法：每个数字都要标"系统边界"（裸引擎 / 引擎+P/D / 整机+路由）。

**一句话结论**：这些数字之间不存在"换算系数"，只存在"条件集合"。可比的前提是七个条件同时对齐：架构代号、卡数、TP/EP/DP、context+输出长度+是否 offline、token 计数口径、τ 四元组、系统边界。全书满足这七条的对子只有一对——DeepInfer 曲线 vs vLLM 的 6%——而它恰好缺卡数与拓扑。

## 4. 新发现的问题

1. **图与 jsonl 是两套口径，仓库只发布图。** `benchmark.py:408` 落盘 `×（1+nextn）`，`plot_pareto.py:67` 画 `×τ`。同一组数据能画出两个差 19%（3.25/4 vs 1）的曲线。任何人想复现官方图，必须知道要传 `--draft_acceptance 0.75`，而 `run_sweep.sh` 只在 decode 分支透传这个参数。建议书里给复现命令时写全 `bash examples/sweep/run_sweep.sh decode ce-128k 131072 3 0.75`。
2. **τ 的线性公式与官方自己的实测不符（19%）。** 见 §6.5。这是全书最值得向维护者确认的一处：是笔误、是"完全相关"上界、还是刻意取保守/乐观？在确认前，DeepInfer 曲线的每用户速度应标注"按官方 τ 约定"。
3. **`826–829 TPS per user`（vLLM 推文串）物理上不成立**，R13 #32 的引文里也没有这句，我未能定位到原文。建议全书不用；若必须提，写"vLLM 另称在 AgentX 上达到 826–829（未说明单位与口径）"。
4. **InferenceX 的阶梯表内部非单调**（GB300：150 → 165,938，200 → 471,578），且 "tokens/s per GPU" 未定义计数口径。不能作为"某引擎在某交互档的实测吞吐"引用，只能作为"该平台前沿包络"引用，且必须带"每档选不同配置"的说明。
5. **warmctl 的 DeepInfer prefill 数字（64,847 tok/s/node = 8.1K/GPU）与官方图（60,300–89,300 tok/s/GPU）差 7–11 倍**，而作者写"DeepInfer's own benchmark agrees (66K prefill tok/s per node), so it's not our harness"——66K/node 折 8.25K/GPU，与官方图的 60–89K/GPU 根本不在一个量级，这句"agrees"没有依据。最可能的解释是**两种 prefill 制度不同**：官方图叫 `Non-cached Input Throughput`，`run_sweep.sh:47` 的 prefill 分支只传 `--context_length/--yoco_clip`，`--non_cached_tokens` 取 `benchmark.py:32` 的默认 **4096**，测的是"131K 上下文里只算 4,096 token 后缀"的**缓存已热**情形，结构性地不计 127K 前缀的 KV 写入；warmctl 是 32K **冷** prompt 全量 prefill，要写 32K×890 B 的 KV。这一条如果成立，官方 prefill 曲线描述的是 agentic 多轮场景的增量 prefill，而不是"冷启动 prefill"，书里必须写清，否则会被读者当成 32K 全量 prefill 的吞吐。
6. **`384+64` 与 `384+32` 不是同一实验条件下的两个 EP 值**（冗余专家数被 ep_size 向上取整），所以 wide-EP 图严格说是"EP8+32冗余 vs EP64+64冗余"，不是单变量对照。
7. **prefill 的 65,536 点与 QA 的 32,768 token 预算冲突。** QA:115 说每个 prefill step 的 `q` 不超过 32,768 token 预算，`run_sweep.sh:45` 的扫描值却到 65,536（`max_num_batched_tokens`）；"per prefill step" 是否等于"per chunk"未定义，图的横轴语义存疑，也说明 benchmark 的 batch 上限与生产 scheduler 的容量不是同一个量。
8. **`DEEPINFER_FORCE_LOAD_BALANCE=1` 在 benchmark 里被强制打开**，其语义（是否插入均衡用的 dummy token / 是否改变 MoE 路由）未在文档说明；若它会改变有效计算量，官方曲线就不是"默认配置"的数字。留给读码员（R05/R08）确认。

## 5. 可用于图表的材料

### 5.1 全书标准性能数据大表（唯一口径源）

说明：`D`=DeepInfer 官方，`V`=vLLM 官方，`S`=SGLang 官方，`T`=第三方。等级按 04 规范；带 `读数` 者为从 PNG 读出的坐标（±3–5%）。

| # | 数据指标 | 数值 | 硬件环境 | 模型配置 | 并发负载 | 数据来源 | 等级 |
|---|---|---|---|---|---|---|---|
| D1 | decode 每卡输出吞吐 @8 req/rank | 2,600 tok/s/GPU（325 tok/s/user） | 64×SM100，EP64DP64，单 scale-up 域 | V4.1-Flash，384+64 专家，131K ctx，nextn=3，a=75%，τ=3.25 | 8 req/DP rank（全集群 512） | `decode_pareto.png` | A（读数） |
| D2 | 同上 @16 | 5,050 tok/s/GPU（318 tok/s/user） | 同上 | 同上 | 16 req/rank（1,024） | 同上 | A（读数） |
| D3 | 同上 @32 | 9,600 tok/s/GPU（298 tok/s/user） | 同上 | 同上 | 32 req/rank（2,048） | 同上 | A（读数） |
| D4 | 同上 @64 | 15,800 tok/s/GPU（248 tok/s/user） | 同上 | 同上 | 64 req/rank（4,096） | 同上 | A（读数） |
| D5 | 同上 @128 | 22,800 tok/s/GPU（177 tok/s/user） | 同上 | 同上 | 128 req/rank（8,192） | 同上 | A（读数） |
| D6 | 同上 @256 | 26,100 tok/s/GPU（103 tok/s/user） | 同上 | 同上 | 256 req/rank（16,384） | 同上 | A（读数） |
| D7 | decode 实测步频（派生） | 100.0 / 97.8 / 91.7 / 76.3 / 54.5 / 31.7 sps | 同上 | 同上（τ 除掉） | 同 D1–D6 | #3 派生 | E |
| D8 | prefill 非缓存输入吞吐 @4,096 | 60,300 tok/s/GPU，72 ms/step | 4×SM100，EP4DP4 | V4.1-Flash，384+32 专家，131K ctx，decoder replay 140 tok | 4,096 tok/DP rank（16,384/step） | `prefill_pareto.png` | A（读数） |
| D9 | 同上 @8,192 | 71,500 tok/s/GPU，115 ms | 同上 | 同上 | 8,192/rank（32,768） | 同上 | A（读数） |
| D10 | 同上 @16,384 | 81,800 tok/s/GPU，201 ms | 同上 | 同上 | 16,384/rank（65,536） | 同上 | A（读数） |
| D11 | 同上 @32,768 | 86,800 tok/s/GPU，372 ms | 同上 | 同上 | 32,768/rank（131,072） | 同上 | A（读数） |
| D12 | 同上 @65,536 | 89,300 tok/s/GPU，731 ms | 同上 | 同上 | 65,536/rank（262,144） | 同上 | A（读数） |
| D13 | wide-EP 吞吐倍数 @8 req/rank | EP64 是 EP8 的 **1.53×**（2,600 vs 1,700） | 64×SM100 EP64 vs 8×SM100 EP8 | 131K，nextn=3，τ=3.25，专家 448 vs 416 | 8 req/DP rank | `wide_ep_decode_pareto.png` | A（读数） |
| D14 | wide-EP 吞吐倍数 @256 | **1.05×**（26,100 vs 24,850）；速度 1.03× | 同上 | 同上 | 256 req/DP rank | 同上 | A（读数） |
| D15 | 每卡专家数（访存代理指标） | EP64=**7**，EP8=**52**，比值 **7.43×** | — | 384 routed + 冗余（EP8:32，EP64:64） | — | QA:103 + `examples/README.md:63` | A |
| D16 | DSpark 变长验证增益 @16/32/48/60/72/96 | **−1.5% / +1.6% / +3.6% / +5.4% / +6.7% / +11.0%** | 未说明（称 EP64/DP64 benchmark） | V4.1-Flash，nextn=5，confidence 调度 | 16–96 seqs/rank | QA:249-256 | B |
| D17 | 变长验证吞吐 @60 | 13,730 → **14,466 tok/s/rank** | 同上 | 同上 | 60 seqs/rank | QA:254 | B |
| D18 | 变长验证平均长度 | 验证 5.00 → **3.68**（−26.4%）；接受 3.293 → **3.192** | — | 同上 | 六点均值 | #12 派生 | E |
| D19 | moe_lb（prefill） | **1.074（EP4DP4）/ 1.045（EP2DP2）**；知乎为 1.0618 / 1.0320 | 未说明 | 4 冗余专家/卡，100 步窗，1000 步重排 | 生产 | QA:99 / 知乎 | B（两版冲突） |
| D20 | moe_lb（decode） | **1.090** | EP64（推断） | 1 冗余专家/卡，1000 步窗，10000 步重排 | 生产 | QA:103 | B |
| D21 | 负载均衡总收益 | 完美均衡离线基准的 **~80%** | — | — | 生产 | QA:127 | B |
| D22 | 生产规模 | **155 万亿 token/天**；**$38/GPU·hr**（谷价） | 单集群 | V4.1-Flash | 生产 | QA:17 | B |
| D23 | 开源版 vs 内部版 | kernel **5–10%** 慢；端到端 **≤5%** | — | FlashMLA 缺一条 SASS | — | QA:235-237 | B |
| D24 | SWA state/序列（派生） | **1,415,040 B** = 20 层 × 134 tok × 528 B | — | V4.1-Flash | 每活跃序列 | R01 #17 / QA:169 | B（算术） |
| V1 | prefill TTFT 128K（TP4） | 2,267 ms → **1,392 ms（−38.6%）** | 4×GB200，TP4 | V4.1-Flash，单请求，关 prefix cache，replay on + CUDA graph | 1 req，128K 输入 | vLLM 博客 2026-10-07 | B |
| V2 | prefill TTFT 1K（TP4） | 160.9 ms → **103.3 ms（−35.8%）**；无 graph 时 +3.1% | 同上 | 同上 | 1 req，1K | 同上 | B |
| V3 | prefill TTFT 128K（DEP2） | 3,950 ms → **2,461 ms（−37.7%）** | 4×GB200，DEP2 | 同上 | 1 req | 同上 | B |
| V4 | decode 吞吐对齐度 | **within 6%** of DeepInfer 公布曲线（方向未说明） | GB200（**卡数未公布**） | V4.1-Flash，131K，nextn=3，Wide-EP + EPLB | batch 8–256 | vLLM X 2026-10-21 | B（缺参） |
| V5 | AgentX 相对提升 | **1.9×**（低并发）/ **5.3×**（150 TPS 约束下） | GB300（InferenceX 面板） | V4.1-Flash，FP4，TP4 低延迟 / DEP2 高吞吐 | AgentX trace | vLLM 博客 | B |
| V6 | AgentX 收入推算 | **$39/GPU·hr** @200 P90 TPS | GB200/GB300（未明） | 同上 | AgentX，200 P90 TPS | vLLM X | B |
| V7 | SWA replay 的 prefill 降幅 | **30–40%** prefill 计算时间；**~30%** TTFT | GB200 | 同上 | 单请求 | vLLM 博客 | B |
| V8 | sparse MQA logits | 每层 **1.2×@8K，14–23×@512K**；4×GB300 上 decode **+3–6%**，prefill **1.43×@512K / 2×@1M** | GB300 | 同上 | 长上下文 | vLLM 博客 | B |
| V9 | 单算子收益 | Mega-mHC **1.14–1.51×**；Mega-Gate **1.18–1.31×**；WO-A **~2.1×**（ITL −6–7%）；mHC 多流 **~4%** 延迟 | GB200/GB300 | 同上 | 小 batch | vLLM 博客 | B |
| S1 | 单流 decode 工程上限 | **873.63 tok/s**（218.4 tok/s/GPU），accept 5.505 | 4×GB300，attn TP4 + MoE TP4 | V4.1-Flash，DSpark 5，random 4k/1k | BS=1 | SGLang 博客 2026-09 | B |
| S2 | DSpark 开关 | **223.50 → 853.49 tok/s（3.82×）**；步频 223.5 → 154.6（−31%） | 4×GB300，attn TP4 + MoE EP4 | 同上 | BS=1 | 同上 | B |
| S3 | MoE EP4 vs TP4 | 854.64 → **873.63（+2.22%）**；rank 到达间隔 11.14 → 3.40 µs | 4×GB300 | 同上 | BS=1 | 同上 | B |
| S4 | 低延迟模式 P90 每用户速度 | **671.2 → 928.4 tok/s（1.38×）** | 4×GB300 | V4.1-Flash + DSpark，AgentX trace | 1 并发 | SGLang LMSYS 2026-10-18 | B |
| S5 | 同上（AMD） | **331.6 → 516.0 tok/s（1.56×）** | 4×MI355X | 同上 | 4 并发 | 同上 | B |
| S6 | SWA bounded replay（SGLang 实现） | prefill **1.37×（4×GB300）/ 1.56×（8×H200）**；AIME 2026 pass@1 相同 | 4×GB300 / 8×H200 | 8×8K prompt 成对测试 | batch 8×8K | SGLang Day-0 博客 | B |
| S7 | Engram 放 host memory | KV 容量 **+36%**，decode 吞吐与 TTFT 持平 | 4×GB300，TP4/EP4 | V4.1-Flash | 成对测试 | 同上 | B |
| S8 | Day-0 单流 | **1,130 tok/s/user，221 tok/s/GPU** | 4×GB300 | 4K/1K，bs=1，EP1 | 1 并发 | 同上（R01 §5.4） | B |
| T1 | 4×B200 C1 | **374.8 tok/s（93.7/GPU）**，410.8 tok/s/user | 4×B200，TP4，vLLM | V4.1-Flash，DSpark 5 + adaptive verification，8K/1K | 1 并发，512 prompts | Lambda 模型卡 | C |
| T2 | 4×B200 C16 / C64 | 2,314.7（578.7/GPU）/ 3,713.9（928.5/GPU） | 同上 | 同上 | 16 / 64 并发 | 同上 | C |
| T3 | 4×B200 C256 | **4,858.7 tok/s（1,214.7/GPU）**，29.6 tok/s/user | 同上 | 同上 | 256 并发 | 同上 | C |
| T4 | 8×B300 单节点 decode 1K→1K | DeepInfer **27,906** vs vLLM **54,985**（2.0×） | 8×B300，DeepInfer EP8×DP8 / vLLM DEP8 | 均 DSpark 5，随机 prompt（接受率 19%） | 2,048 并发 | warmctl 2026-10-21 | C（条件披露） |
| T5 | 同上 8K→1K | 6,933 vs 29,500（4.3×，接受率 12%） | 同上 | 同上 | 2,048 并发 | 同上 | C |
| T6 | 同上 prefill 32K→1 | 64,847 vs 589,924（9.1×） | 同上 | 同上 | 512 并发 | 同上 | C |
| T7 | InferenceX GB300 NVL72 前沿 | 50 tok/s/user → **359,012**；75 → 285,907；100 → 267,304；150 → 165,938；200 → 471,578；峰值 **1,011,514**（tok/s/GPU，口径未定义） | GB300 NVL72，vLLM/SGLang/Dynamo，FP4 | V4.1-Flash | AgentX，每档不同配置 | InferenceX 2026-10-22 | B（数字）/ E（口径） |
| T8 | InferenceX B300 / B200 前沿 | B300：50 → 328,417；75 → 376,639；100 → 315,201。B200：75 → 222,682；100 → 178,100 | B300 / B200，FP4 | 同上 | AgentX | InferenceX | B/E |

**表注（必须随表印出）**
- D1–D7：纵轴含 τ=3.25（假设 75% 接受率），不是实测 token 率；实测步频见 D7。
- D8–D12：是"缓存已热"的增量 prefill。`--context_length 131072` 但 `--non_cached_tokens` 取 `benchmark.py:32` 默认 4,096，只计后缀 token，不含 127K 前缀的 KV 写入；不能当作 32K 冷 prompt 的 prefill 吞吐（见 §4 第 5 条）。
- D13–D15：EP8 组与 EP64 组的冗余专家数不同（32 vs 64），不是单变量对照。
- V1–V3：单请求、关 prefix cache、median of 3；TP4 与 DEP2 是两条不同拓扑。
- V4：只有"within 6%"一句，无 GPU 数、无拓扑、无方向。
- S1–S3：接受长度 5.505 是**模拟**值（`SGLANG_SIMULATE_ACC_LEN`）；S4–S5：P90 分位数，不是均值。
- T1–T3：4 卡 TP4，与 DeepInfer 的 64 卡 EP64 在"每卡权重"上差 16×。
- T7–T8：每档是不同配置的最优值，跨档不可相除；"per GPU" 未定义 token 口径。

### 5.2 EP8 vs EP64 收敛表（可直接做成双折线图）

| 每 DP rank 并发 | EP64DP64 tok/s/GPU | EP8DP8 tok/s/GPU | EP64/EP8 吞吐 | EP64/EP8 每用户速度 |
|---:|---:|---:|---:|---:|
| 8 | 2,600 | 1,700 | 1.53× | 1.34× |
| 16 | 5,050 | 3,350 | 1.51× | 1.37× |
| 32 | 9,600 | 6,900 | 1.39× | 1.34× |
| 64 | 15,800 | 12,450 | 1.27× | 1.24× |
| 128 | 22,800 | 19,100 | 1.19× | 1.18× |
| 256 | 26,100 | 24,850 | 1.05× | 1.03× |

### 5.3 "口径八问"清单（书里每个性能数字都应能回答）

1. 架构代号：SM100 / SM103 / MI355X / 其他？
2. 卡数与系统边界：裸引擎 / 引擎+P-D / 整机+路由？是否在同一 NVLink 域？
3. 并行拓扑：TP / EP / DP / CP 各几？每卡权重多少 GB？
4. 负载形状：输入长度、输出长度、是否 offline、有无请求到达过程？
5. token 口径：output / total / 含 prefix 命中？
6. 投机设置：nextn 几、acceptance 多少、τ 从哪来（实测/模拟/假设）、用线性还是几何公式？
7. 时间窗与统计量：warmup 多少、测量多少步、均值还是 P90、是否剔除非稳态？
8. 基线：和谁比、是同机 A/B 还是"对公布的数"？

### 5.4 τ 敏感性表（把"同一实测步频换假设"的效应画成表）

以 D6 的 256 并发点（实测 31.7 sps）为例：

| acceptance 假设 | 线性 τ (1+n·a) | 每用户 tok/s | 每卡 tok/s | 几何 τ (Σa^i) | 每用户 tok/s | 每卡 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75（官方图） | 3.25 | 103 | 26,368 | 2.73 | 87 | 22,185 |
| 0.50 | 2.50 | 79 | 20,283 | 1.88 | 59 | 15,212 |
| 0.19（warmctl 1K 实测） | 1.57 | 50 | 12,738 | 1.23 | 39 | 10,003 |
| 0.12（warmctl 8K 实测） | 1.36 | 43 | 11,034 | 1.14 | 36 | 9,218 |

### 5.5 建议配图（三张）

1. **双面板 Pareto**：左 decode（EP64/EP8 两条线，标注膝点 16–32、肩点 64）、右 prefill（标注饱和 32K），副标题写全配置与"τ=3.25 为设定值"。
2. **边际收益柱状图**：decode 五段的 `Δ吞吐%` 与 `Δ每用户速度%` 并列，直观看"64 之后开始亏"。
3. **口径矩阵**：横轴按 §5.3 的八问（架构、卡数/边界、TP/EP、负载、token 口径、τ、时间窗、基线），纵轴五个数据源（DeepInfer 曲线、vLLM 6%、SGLang 873、Lambda、InferenceX），打勾/打叉——一眼看出只有 vLLM 6% 接近全对齐。
