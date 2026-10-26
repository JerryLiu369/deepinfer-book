# 05 事实基线

本文件记录编写任务包时（2026-10-23）已经核对过的事实。用途：

1. 让主代理和子代理有一个可靠的起点，少走弯路。
2. 提醒容易混淆、容易写错的地方。

**这些事实仍需在 P0 阶段用最新代码复核一遍**（仓库可能已有新提交）。复核时如发现变化，以新代码为准，并记入 `work/decisions.md`。

核对时使用的版本：`deepseek-ai/DeepInfer` @ `9bd9cf2`（2026-10-21，"Bump DeepEP to f6aa38fe…"，当时仓库共 4 个提交）。

## 1. 基本信息

| # | 事实 | 来源 | 等级 |
| --- | --- | --- | --- |
| 1.1 | 2026-10-20 由 DeepSeek 在 X 上宣布开源，仓库 `github.com/deepseek-ai/DeepInfer`，Apache-2.0 许可证 | README、官方 X | B |
| 1.2 | 定位原文："the high-performance inference framework that powers DeepSeek's large-scale production services. This open-source release provides offline inference with the core features needed for serving at scale." | README | B |
| 1.3 | 只支持一个模型：DeepSeek-V4.1-Flash；默认 checkpoint 为 `deepseek-ai/DeepSeek-V4.1-Flash-1017` | README | B |
| 1.4 | 硬件：NVIDIA SM100 GPU、华为昇腾 950 NPU | README | B |
| 1.5 | 环境：Linux、GCC ≥ 13、Python 3.12、CUDA 13（NVIDIA）或 CANN 9.2（昇腾）；`export DEEPINFER_ACCELERATOR=nvidia|ascend` 后运行 `./build.sh` | README | B |
| 1.6 | 完整服务所需但**未包含**：KV cache 系统；VE/Prefill/Decode 分离（README 说"大部分所需引擎功能已实现"）；prefill 与 decode 数据并行副本之间的负载均衡 | README | B |
| 1.7 | 致谢原文："originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM" | README | B |
| 1.8 | 多个文件头写有 `# Derived from vLLM; modified by DeepSeek.`，包括 `entrypoints/llm.py`、`worker/base.py`、`worker/cache_engine.py`、`worker/worker.py`、`worker/model_runner.py`、`distributed/shm_broadcast.py`、`sequence.py`、`model_executor/layers/sampler.py`、`rotary_embedding.py`、`layernorm.py`、`vocab_parallel_embedding.py`、`model_loader.py`、`model_executor/utils.py`、`parallel_utils/communication_op.py`（不完整，R03 需给出完整列表） | 源码 | A |
| 1.9 | 同时开源 `deepseek-recipe`：API 协议与 token 序列之间的编解码组件，与官方 API 行为对齐；核心用 Rust 实现，提供 Python 绑定；支持 Chat Completions、Responses、Messages 三种 API | `doc/inference_deployment_qa.md` | B |
| 1.10 | 部分代码注释是中文（例如 `llm_engine.py`、`eplb/__init__.py`） | 源码 | A |

## 2. README 列出的特性与开关

| 特性 | 开关 / 用法 | 备注 |
| --- | --- | --- |
| 优化内核 | FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect，以及仓库内 TileLang 算子（`deepinfer/model_executor/layers/tilelang_ops`） | |
| 并行 | Attention DP / balanced CP，MoE EP + EPLB | 示例：`--ep 8`；`--ep 2 --dp 8`；`--ep 2 --cp 8` |
| DSpark 与按置信度安排验证 | `DEEPINFER_ENABLE_MTP_VARLEN=1`；`DEEPINFER_USE_NEXTN_DECODE` 控制最大草稿长度（默认 5） | |
| decoder SWA 有界重放 | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`（V4.1 默认）；保留的后缀必须覆盖 SWA 窗口 + 最大回退长度 | |
| 异步调度（零开销批调度器） | 默认开启 | |
| 批不变推理 | `DEEPINFER_BATCH_INVARIANT=1`（仅 NVIDIA） | |
| 确定性采样 | `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0`，并给 `SamplingParams` 传固定 `seq_seed`（仅 NVIDIA） | |
| 结构化输出 | `SamplingParams(so_args={"schema_type": ..., "schema": ...})`，支持 regex、JSON schema、EBNF | 示例见 `examples/_structured_output.py` |

`deepinfer/environ.py` 中约有 62 行包含 `DEEPINFER_`，附录 B 需完整整理。

## 3. 代码结构（粗略）

- 顶层目录：`csrc/`（C++，调度与输入准备等）、`deep_cluster/`（硬件与拓扑绑定）、`deepinfer/`（主体）、`doc/`、`eval/`、`examples/`、`scripts/`、`tests/`、`vendor/`（git submodule）。
- `vendor/` submodule：DeepEP、DeepEP-Ascend、DeepGEMM、DeepGEMM-Ascend、DeepSelect、FlashMLA。
- `deepinfer/` 子目录：`_C`、`distributed`、`engine`、`entrypoints`、`hugging_face`、`model_executor`、`npu_ops`、`structure_output`、`ve`（视觉编码器）、`worker`。
- 代码量（只统计 .py/.cpp/.h/.cu 等，粗略）：`model_executor/` 约 3.5 万行，占绝大部分；`worker/` 约 2,800 行；`engine/` 约 2,000 行；`csrc/` 约 2,400 行。R03 需精确统计。
- `tilelang_ops/` 下有根目录、`cuda/`、`asc/` 三套同名文件，推测根目录为公共层或分发层（E 级，R10 核实）。
- 关键文件：`engine/llm_engine.py`（约 1,000 行）、`worker/cache_engine.py`（约 1,000 行）、`worker/model_runner.py`、`models/v4_modules/attn.py`（约 1,000 行）、`csrc/engine/zos.cpp`。
- 代码中出现、需要解释的内部名字：ZOS、YOCO、Engram、mHC/HC、DFlash、DSpark、nextn/MTP、indexer（`index_attn.py`）、compress、cross_layer、StateBlock、userbuffer、shm_broadcast、EPLB shuffle、Mega MoE。
- 在 `deepinfer/` 中搜索 `tensor_parallel`、`tp_size` 无结果，开源版可能没有张量并行（E 级，R08 核实）。
- `llm_engine.py` 中有 `EngineType.PROMPT`、`EngineType.COMPLETION`、`EngineType.NORMAL` 三种引擎类型，并有 `send_processed_prompts`（"把已经处理完的 prompt sequence 发送给 completion engine"）、`pe_do_save_kvcache` 等虚函数（约 815–850 行、947–948 行）。这是 README 所说"P/D 分离的大部分引擎功能已实现"的痕迹：引擎区分 prefill 引擎（PE）和 decode 引擎（CE），但跨机传输与存储的实现是空的虚函数（A 级，R12 进一步核实）。
- `llm_engine.py:888` 注释："开启 yoco 截断时，允许抢占导致的结果变化"，说明 YOCO 截断与抢占之间存在结果差异的已知取舍（A 级，R07 解释）。
- `llm_engine.py:997-1002`：每步把实测的 forward 与 sample 耗时回传给调度器（`scheduler.update_mtp_runtime`），可能用于投机解码长度决策（E 级，R09 核实）。
- EPLB 中使用 `torch.distributed._symmetric_memory`（`eplb/__init__.py:8`），并有注释说明 NPU 暂不支持 symm_mem（同文件约 199 行）。

## 4. 部署问答（`doc/inference_deployment_qa.md`）中的关键数字

以下全部为 B 级（DeepSeek 官方说法），**描述的是 DeepSeek 的生产系统，不全是开源代码中的功能**。

- 规模：单个集群每天服务 **155 万亿 token**；整体收入约 **每 GPU 小时 38 美元（按谷时定价）**。
- 三个设计目标：支持多厂商加速器；prefill 用普通 8 卡机、decode 用普通 64–128 卡超节点；KV cache 保留超过 72 小时。
- 架构：E/P/D 分离 + 专用存储节点；注意力 DP，DP 数 = EP 数。
- 并行配置：prefill EP2–8/DP2–8（在 HBM 允许的情况下用尽量小的）；decode EP64–128/DP64–128（Wide-EP，在 GPU 数允许的情况下用尽量大的）；VE 单卡；**TP 和 CP 暂不使用**（注意 README 列出了 balanced CP，代码支持 `--cp`，这是"支持"与"生产使用"的区别）。
- 最小有效部署单元：64 卡 prefill + 64 卡 decode；P:D 卡数比约 1:1。
- 跨数据中心：每 8 卡约需 400 Gbps 互联带宽；DeepSeek 实际部署在不同但相邻的数据中心。
- KV 存储：3FS，10+2 纠删码；约 100 PiB SSD；TTL agent 流量 3 天、chatbot 流量 7 天；每 GPU 约 14.4 TB 逻辑容量；推荐每台存储服务器 20 × 30.73 TB SSD，读 ≥ 40 GB/s、写 ≥ 5 GB/s。
- EPLB：prefill 每卡 4 个冗余专家、统计窗口 100 步、每 1,000 步重分配，`moe_lb` 1.074（EP4DP4）/ 1.045（EP2DP2）；decode 每卡 1 个冗余专家（EP64 时取整后每卡 7 个专家，不含共享专家）、窗口 1,000 步、每 10,000 步重分配，`moe_lb` 1.090。
- Prefill 负载均衡：按上下文长度分 4 档（<16K / 16–64K / 64–256K / ≥256K）；单卡按时间预算和二次代价模型 `cost(q, c)` 求分块大小，上限 32,768 token。
- 负载均衡总体效果：约达到完全均衡离线基准的 80%。
- SWA KV：存储粒度 128 token；V4.1 存前 20 层；每序列存 20 层 × 134 token × 528 B = 1,415,040 B；528 B/token 为 MXFP8 SWA KV 格式。
- 有界重放：只重算最近 128 个 token；从位置 s 重放时，位置 i 的 query 只看 max(s, i−127) 到 i 的 key；精确重建 decoder SWA KV 需要把最后 2,560 个 token 跑过 20 个 decoder 层；decoder 有界重放"几乎让 prefill 总计算量减半"。
- 命中策略：输入长度 L 时最大命中长度 ≤ L − 134（128 窗口 + 6 个投机回退 token）。
- 低精度：main KV 为 per-16 e4m3（类似 NVFP4 但没有全局 scale）；indexer KV 为 per-32 ue8m0 MXFP4；MoE 路由专家权重为 per-32 ue8m0 MXFP4。
- 内核：开源 FlashMLA 缺一个手工 SASS 优化，解码内核慢 5%–10%；端到端差距最多 5%。
- DSpark：生产中草稿模型一次生成 5 个 token；变长验证使每步验证 token 数减少约 30%（5.00 → 3.2–4.4），平均接受长度基本不变（3.29 对 3.20）；生产工作点（每卡约 60 并发）吞吐 +4–7%。数据表：并发 16 时 −1.5%，32 时 +1.6%，48 时 +3.6%，60 时 +5.4%，72 时 +6.7%，96 时 +11.0%。
- 异步调度：参考 SGLang 和 NanoFlow；GPU 执行 `forward(N) → sample(N)` 时，CPU 并行执行 `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`。
- 弹性：decode 引擎收到下线通知后有 1 分钟宽限期，未完成请求连同 KV 一次性转给其他引擎；配额利用率高于 0.8 时禁止缩容；重启为分钟级。
- README 中的基准：prefill 用 4 张 SM100，decode 用 64 张，均在一个 scale-up 域内；复现脚本 `examples/sweep/run_sweep.sh`。

## 5. 容易混淆的地方（重点提醒）

1. **"38 美元"与"38.96 美元"不是一回事。** 38 美元/GPU 小时是 DeepSeek 自己说的生产集群整体收入（谷时定价）。38.96 美元是 vLLM 团队在 GB300 NVL72 上用 SemiAnalysis AgentX 负载、按 DeepSeek 谷时价格换算出的**估算毛收入**，不含任何成本。两者条件不同，不能比较。
2. **"支持"不等于"生产在用"。** 例如代码支持 CP，但文档说生产暂不使用 CP。
3. **"文档描述"不等于"代码里有"。** 负载均衡器、kvcache-store、3FS 集成、Entrypoint、弹性伸缩都只在文档中描述。
4. **"vLLM 在 DeepInfer 的 6% 以内"是 vLLM 方面的说法**（目前见于 AlphaSignal 报道，D 级），条件为 GB200、decode、batch 8–256。必须找到 vLLM 官方原始博客（R13/R16）再引用，并写清条件。
5. **模型参数：** 有报道称 V4.1-Flash 为 552B 参数 MoE、上下文 131K、KV 每 token 约 890 字节、使用 "Compressed Sparse Attention 2"、FP4 KV 和跨层 KV 共享。这些来自 AlphaSignal（D 级），必须以 HF 模型卡和 `config.json` 为准（R02）。
6. **"异步调度/零开销调度"不是 DeepInfer 首创。** 文档明确说参考了 SGLang 和 NanoFlow。
7. **作者没有运行过 DeepInfer。** 书中所有关于运行行为的描述都来自代码阅读或官方文档。

## 6. 来源可信度提醒

| 来源 | 处理方式 |
| --- | --- |
| DeepInfer 仓库源码、README、`doc/` | 首选（A/B） |
| DeepSeek 官方知乎、官方 X、HF 模型卡 | 首选（B） |
| vLLM、SGLang 官方仓库、文档、官方博客 | 可用（B/C），注意时效 |
| Bloomberg、The Next Web、The Decoder、TechTarget、The New Stack、MarkTechPost | 只用于背景（D） |
| AlphaSignal | 只作线索，需追溯原始出处（D） |
| `deepinfer.net` | **不是** DeepSeek 官方站点（官方是 `github.com/deepseek-ai`），按第三方内容对待，只作线索（D） |
| dev.to、个人博客、各类"XX 详解""XX vs YY"文章（如 essamamdani.com、aireiter.com、particula.tech、deepinfra.com 博客等） | 只作线索；不能作为任何结论的唯一来源（D） |
| Reddit、Hacker News、知乎回答 | 只用于第 15、17 章的社区观点（D） |

## 7. 一手资料链接

- 仓库：https://github.com/deepseek-ai/DeepInfer
- 部署问答：https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md
- 官方知乎文章（V4.1 Flash 推理系统）：https://zhuanlan.zhihu.com/p/2095560295930508953
- DeepSeek-V3/R1 推理系统（背景）：https://zhuanlan.zhihu.com/p/27181462601
- 官方 X 公告：https://x.com/deepseek_ai/status/2112476496283775135
- HF 模型：https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- deepseek-recipe：https://github.com/deepseek-ai/deepseek-recipe
- 内核库：https://github.com/deepseek-ai/FlashMLA 、https://github.com/deepseek-ai/DeepGEMM 、https://github.com/deepseek-ai/DeepEP 、https://github.com/deepseek-ai/TileKernels 、https://github.com/deepseek-ai/DeepSelect
- vLLM：https://github.com/vllm-project/vllm ；V4.1-Flash recipe：https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
- SGLang：https://github.com/sgl-project/sglang
