# 03 调研任务清单

## 使用方法

- 每张任务卡就是一次子代理调用的核心内容。派发时按 `04_SUBAGENT_PROMPTS.md` 的方法拼接提示词。
- 主代理可以拆分或合并任务卡，但必须保证每个"必答问题"都有人负责。
- 子代理笔记存到 `work/notes/Rxx_简称.md`。
- 路径都相对于 DeepInfer 仓库根目录，除非另有说明。

## 派发波次

| 波次 | 任务 | 依赖 |
| --- | --- | --- |
| 第一波（侦察） | R01、R02、R03 | 无 |
| 第二波（深入） | R04–R12、R13、R14 | R01–R03 的笔记（R13、R14 只需 R01） |
| 第三波（综合） | R15、R16、R17、R18 | R03、R13、R14 等 |
| 审校 | 每章一个技术审校任务（模板 T4） | 章节初稿 |

---

## R01 一手资料收集

- **服务章节**：全书，尤其第 2、4、13 章
- **模板**：T1 研究员
- **目标**：建立一手资料库，给每份资料定证据等级，摘录关键原文。
- **必须收集**：
  1. 仓库 `README.md`、`doc/inference_deployment_qa.md`、`examples/README.md`、`eval/README.md`、`CONTRIBUTING.md`。
  2. DeepSeek 官方知乎文章：`https://zhuanlan.zhihu.com/p/2095560295930508953`（DeepSeek V4.1 Flash 推理系统）。抓不到时，查找官方微信公众号或其他官方渠道的同文；仍抓不到就记录失败原因。
  3. DeepSeek 早期推理系统文章（DeepSeek-V3/R1 推理系统，`https://zhuanlan.zhihu.com/p/27181462601`），作为背景对照。
  4. DeepSeek 官方 X 公告（2026-10-20）。
  5. Hugging Face 模型卡：`deepseek-ai/DeepSeek-V4.1-Flash-1017`（含 `config.json`）。
  6. V4.1-Flash 的技术报告或论文（如有）。
  7. `https://github.com/deepseek-ai/deepseek-recipe` 的 README。
  8. vLLM 官方关于 V4.1-Flash / DeepInfer 的博客与 recipes（`recipes.vllm.ai` 等）。
  9. SGLang 官方关于 V4.1-Flash 的文档或博客。
  10. 媒体报道：只收 2–3 篇信息量最高的，标 D 级，用于第 2 章背景。
- **必答问题**：
  1. DeepSeek 对"为什么开源 DeepInfer"的原话是什么？
  2. 有媒体说 DeepSeek 曾表示开源推理引擎"还不现实"，原始出处是什么？（可能与 2025 年 DeepSeek 关于贡献推理引擎给 vLLM 的说明有关，需核实）
  3. 知乎文章和 `inference_deployment_qa.md` 有哪些内容不重合？知乎文章多讲了什么？
- **产出**：资料清单（URL、标题、日期、访问日期、等级、一句话摘要）+ 每份一手资料的关键原文摘录（中文资料保留原文）。

## R02 DeepSeek-V4.1-Flash 模型结构

- **服务章节**：第 3 章（主）、第 7、8、10 章
- **模板**：T1 + T2（既查资料也读代码）
- **目标**：讲清模型中所有**影响推理**的结构，以及代码中对应的实现。
- **起点**：HF `config.json` 与模型卡；`deepinfer/model_executor/models/deepseek.py`、`decoder_layer.py`、`v4_modules/` 下全部文件、`models/mtp/dflash.py`、`models/vision_encoder/`；`deepinfer/hugging_face/config.py`。
- **必答问题**：
  1. 总参数、激活参数、层数、专家数（路由专家 / 共享专家）、每 token 激活专家数、上下文长度、词表大小。
  2. 注意力：MLA 如何使用？`index_attn.py`（indexer、稀疏选择）在做什么？`compress.py` 压缩什么？有没有被称为 CSA 的机制？SWA 窗口多大、哪些层用？
  3. 层结构：代码中的 encoder 层、decoder 层是什么意思（CED）？YOCO 指什么？`cross_layer.py` 的跨层 KV 共享怎么做？为什么"decoder 层的块状 KV 由最后一个 encoder 层的 hidden state 投影得到"？
  4. `hc.py`、`mhc_*.py`：mHC（超连接）是什么，对推理有什么影响？
  5. `engram*.py`：Engram 是什么？它为什么需要 `gather_engram_tokens_zos` 这类专门算子？
  6. MTP / nextn / DFlash：草稿 token 从哪里来？置信度头在哪？
  7. KV 的种类与每 token 体积：块状 KV（main KV + indexer K）、SWA KV、状态缓存各存什么、什么格式（FP4 e4m3 per-16、MXFP4、MXFP8）。有第三方说每 token 约 890 字节，核实或推算。
  8. 视觉编码器的结构与位置。
- **产出**：模型结构卡 + "模型特点 → 推理影响 → DeepInfer 中的对应代码"三列表。
- **注意**：工作量大，建议拆成 R02a（注意力与 KV）、R02b（层结构、YOCO、mHC、Engram）、R02c（MTP/DFlash、视觉编码器、参数总览）。

## R03 仓库地图

- **服务章节**：第 5 章（主）、第 16 章、附录 C
- **模板**：T2 读码员
- **目标**：每个目录、每个主要文件负责什么。
- **必答问题**：
  1. 每个顶层目录（`deepinfer/`、`csrc/`、`deep_cluster/`、`vendor/`、`examples/`、`eval/`、`tests/`、`scripts/`）的职责。
  2. `deepinfer/` 下每个子目录、每个超过 150 行的文件的职责（一句话）。
  3. 代码规模统计（按目录，Python / C++ / 其他分开）。
  4. 依赖：`requirements*.txt`、`pyproject.toml`、`setup.py`、`build.sh`、`Makefile`；`vendor/` 中每个 submodule 的用途和上游仓库。
  5. 哪些文件的文件头写着 "Derived from vLLM"？完整列表。
  6. 代码注释用什么语言？（已知部分为中文）
  7. 公开 API：`deepinfer/__init__.py` 导出了什么？`LLM` 类的接口长什么样？
- **产出**：目录职责表、文件职责表、依赖表、vLLM 派生文件列表。

## R04 请求生命周期与进程模型

- **服务章节**：第 5 章
- **模板**：T2
- **起点**：`examples/offline_inference.py` → `deepinfer/entrypoints/llm.py` → `deepinfer/engine/llm_engine.py` → `deepinfer/worker/worker.py`、`model_runner.py` → `deepinfer/model_executor/models/deepseek.py` → `layers/sampler.py`；`csrc/engine/pybind_*.cpp`；`deepinfer/distributed/shm_broadcast.py`；`deepinfer/shared_memory.py`；`deep_cluster/topology/bind.py`。
- **必答问题**：
  1. 从 `LLM(...)` 构造到 `generate()` 返回，按顺序列出每一步：文件、函数、做了什么。
  2. 有几个进程？每个 GPU 一个 worker 吗？driver 和 worker 怎么通信？
  3. 一个解码步（step）里，调度、准备输入、forward、采样、后处理分别在哪里发生？
  4. `Sequence`、`SequenceMetadata`、`SamplingParams`、输入元数据（`input_metadata/`）这些数据结构分别是什么？
  5. CUDA Graph 在哪里、用于哪些阶段（`cuda_graph_runner.py`、`dflash_cuda_graph_runner.py`）？
  6. C++ 部分（`csrc/engine/`）承担了哪些原来在 Python 里的工作？为什么要用 C++？
  7. `deep_cluster/` 如何做 CPU/NIC/GPU 拓扑绑定？
- **产出**：生命周期步骤表 + 可直接改写成 mermaid 时序图的步骤列表 + 进程模型说明。

## R05 调度

- **服务章节**：第 6 章
- **模板**：T2
- **起点**：`deepinfer/engine/llm_engine.py`（搜索 ZOS、`_execute_model_async_step`）、`deepinfer/worker/zos_status.py`、`csrc/engine/zos.cpp`、`csrc/engine/zos.hpp`、`csrc/engine/scheduler.hpp`、`csrc/engine/pybind_scheduler.cpp`、`csrc/engine/flops_estimator.hpp`、`deepinfer/model_executor/input_metadata/prepare_tensor_input_zos.py`、`tilelang_ops/schedule_mtp_varlen.py`、`worker/decode_perf_cache.py`。
- **必答问题**：
  1. 调度器的核心逻辑：如何挑选请求组成 batch？prefill 和 decode 混合吗？有没有抢占（preemption）？
  2. ZOS 开启和关闭时，一个 step 的执行顺序分别是什么？用时间线写出来。
  3. ZOS 下，CPU 安排第 N+1 步时还不知道第 N 步的采样结果，具体怎么处理：输入 token 怎么来、已结束的请求怎么处理、投机解码的接受长度怎么处理？
  4. 文档说 CPU 只确定总预算，每个请求的验证长度在 GPU 上由 TileLang 内核重算，写回 pinned buffer。找到对应代码。
  5. prefill 分块大小：文档描述的"二次代价模型 + 时间预算"在开源代码里有没有？有的话在哪？没有的话，开源版怎么决定分块？（线索：`csrc/engine/flops_estimator.hpp` 和 `tests/engine/test_flops_estimator.py`，可能与代价估计有关，需核实）
  6. `decode_perf_cache.py` 是做什么的？（可能是吞吐曲线，用于 DSpark 决策，需核实）
- **产出**：调度器说明卡 + 两条时间线 + 代码位置表。

## R06 KV cache 与状态缓存（显存内）

- **服务章节**：第 7 章
- **模板**：T2
- **起点**：`deepinfer/worker/cache_engine.py`、`tilelang_ops/store_kv_cache.py`、`load_paged_kv_cache.py`、`state_cache.py`、`cache_common.py`、`kv_quant`（昇腾目录下有）、`get_kv_index.py`、`swa_index.py`；`tests/worker/test_state_cache_ops.py`。
- **必答问题**：
  1. 块（block）大小、块表结构、分配与释放方式。
  2. 一共有几类缓存（块状 KV、indexer K、SWA 状态、压缩边界累加器 `ca_state`/`ica_state` 等）？各自形状、格式、存放位置。
  3. `StateBlock` 槽位布局：存储区、丢弃区、环形缓冲多出的 6 个位置。在代码里找到对应。
  4. 128 token 对齐存储、decode 期间的副本区（replica region）在代码里怎么实现？
  5. 开源版有没有前缀缓存（prefix caching）？如果有，在显存内怎么做；如果没有，入口在哪里预留？
  6. 显存容量规划：开源版如何决定分配多少块？
- **产出**：缓存种类表 + 槽位布局图草稿 + 代码位置表。

## R07 SWA 有界重放与 YOCO

- **服务章节**：第 8 章（主）、第 3、7 章
- **模板**：T2 + T1
- **起点**：`deepinfer/model_executor/models/v4_modules/yoco.py`、`input_metadata/yoco.py`、`examples/yoco/yoco_equiv.py`、`environ.py` 中的 `DEEPINFER_YOCO_*`；`doc/inference_deployment_qa.md` 的 "KV cache hits and recomputation" 部分。
- **必答问题**：
  1. 用一个具体数字例子（例如 L = 10,000 的输入）讲清：精确重建 SWA KV 要算什么，有界重放算什么，节省多少。
  2. "位置 i 的 query 只与 max(s, i−127) 到 i 的 key 计算"在代码中如何实现？
  3. encoder SWA 有界重放与 decoder SWA 有界重放的区别，各自在代码哪里。
  4. `KEEP_TOKENS=140` 与文档中的 134（128 + 6）、"SWA 窗口 + 最大回退长度"之间是什么关系？核对清楚。
  5. `yoco_equiv.py` 在验证什么等价性？验证方法和结论是什么？
  6. 近似对精度的影响：有没有公开数据或测试（PPL、KL、评测分数）？
  7. 文档说 decoder 有界重放"几乎让 prefill 总计算量减半"，这个结论依赖什么前提？
- **产出**：机制说明卡 + 数字例子 + 代码位置 + 精度证据。

## R08 并行与 EPLB

- **服务章节**：第 9 章
- **模板**：T2
- **起点**：`deepinfer/model_executor/parallel_utils/`（`parallel_state.py`、`communication_op.py`、`userbuffer.py`、`nccl_memory.py`）、`deepinfer/model_executor/eplb/`（`__init__.py`、`policy.py`、`shuffle.py`）、`v4_modules/moe.py`、`mlp.py`、`tilelang_ops/prepare_cp_mirror.py`；`tests/distributed/test_expert_shuffle.py`、`tests/models/test_eplb_policy.py`；`examples/offline_inference.py` 的 `--ep/--dp/--cp` 参数。
- **必答问题**：
  1. 进程组如何划分（DP、EP、CP）？
  2. MoE 层一次 forward 中 token 的分发（dispatch）和汇合（combine）流程，DeepEP 在哪一步。
  3. Attention DP 下各 DP rank 的 batch 不一样大，怎么同步？
  4. balanced CP 是什么？"balanced"平衡的是什么？`prepare_cp_mirror` 做什么？
  5. EPLB 策略算法（`policy.py`）、权重搬运（`shuffle.py`，是否用 `symm_mem`）、统计窗口与重分配周期参数在哪里配置。
  6. userbuffer 是什么，用于什么通信？
  7. 有没有张量并行（TP）代码？
- **产出**：并行结构说明卡 + MoE 数据流步骤 + EPLB 算法说明 + 代码位置。

## R09 投机解码：DSpark、DFlash、MTP

- **服务章节**：第 10 章
- **模板**：T2 + T1
- **起点**：`deepinfer/model_executor/models/mtp/dflash.py`、`input_metadata/dflash.py`、`layers/mtp_info_pool.py`、`tilelang_ops/dflash.py`、`dflash_copy.py`、`schedule_mtp_varlen.py`、`store_mtp_info.py`、`sum_smaller_probs.py`、`layers/sampler.py`；环境变量 `DEEPINFER_ENABLE_MTP_VARLEN`、`DEEPINFER_USE_NEXTN_DECODE`。
- **必答问题**：
  1. DFlash 是什么？和 MTP、nextn 是什么关系？
  2. 草稿生成、验证、接受、回退的完整流程，代码位置。
  3. 置信度头在哪里？输出什么？
  4. "按吞吐曲线选择验证前缀"的算法在代码中如何实现？吞吐曲线从哪里来？
  5. 验证采样用的是什么接受规则（严格匹配、概率接受）？`sum_smaller_probs` 用途？
  6. 把文档中的数据表转述清楚：为什么低并发时变长验证反而略亏、高并发时收益更大？
  7. DSpark 这个名字有没有论文或其他出处？
- **产出**：机制说明卡 + 流程步骤 + 数据解读。

## R10 内核与硬件抽象

- **服务章节**：第 11 章
- **模板**：T2 + T1
- **起点**：`vendor/` 各 submodule 的 README；TileKernels 仓库；`deepinfer/platform.py`、`deepinfer/npu_ops/`、`tilelang_ops/cuda/` 与 `tilelang_ops/asc/` 的对应关系、`attention_backend.py`、`flash_mla_utils.py`、`csrc/cuda_utils.cpp`、`csrc/asc_utils.cpp`。
- **必答问题**：
  1. FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect 各自是什么、在模型的哪一部分被调用（给出调用位置）。
  2. Mega MoE 是什么？在代码哪里？文档为什么说它在一个 scale-up 域内总是更快？
  3. `tilelang_ops/` 根目录、`cuda/`、`asc/` 三者是什么关系（根目录是分发层？）。
  4. NVIDIA 与昇腾版本的功能差异（例如批不变、确定性采样只支持 NVIDIA）。
  5. 低精度格式在代码中的位置：FP4 KV 量化、MXFP4 专家权重、MXFP8 SWA KV。
  6. 文档提到开源 FlashMLA 比内部版少一个 SASS 优化，记录原文。
- **产出**：内核库表 + 调用位置 + 双硬件差异表。

## R11 批不变、确定性采样、结构化输出、测试与评测

- **服务章节**：第 12 章
- **模板**：T2
- **起点**：`environ.py` 中的 `DEEPINFER_BATCH_INVARIANT`；`layers/sampler.py`、`sampler_utils.py`、`tilelang_ops/fused_sample.py`、`random_sampling.py`、`rstd_1ulp.py`（昇腾）；`deepinfer/structure_output/`；`examples/_structured_output.py`；`tests/models/`（`test_batch_invariant.py`、`test_kl.py`、`test_sampler.py`、`offline_decode_ppl.json`）；`eval/`。
- **必答问题**：
  1. 批不变是怎么实现的？哪些算子需要改？性能代价有没有数据？
  2. 确定性采样：`seq_seed` 如何作用？为什么要关闭变长 MTP？
  3. 结构化输出：用了什么语法引擎（自研还是 xgrammar 等）？编译、匹配、掩码各在哪里？
  4. PPL 基线测试、KL 测试分别保证什么？
  5. `eval/` 的流程和指标；有没有公布的分数？
- **产出**：三个机制的说明卡 + 测试体系说明。

## R12 未开源部分的系统设计与 deepseek-recipe

- **服务章节**：第 4、13 章
- **模板**：T1
- **起点**：`doc/inference_deployment_qa.md`（全文）、官方知乎文章、`deepseek-recipe` 仓库。
- **必答问题**：
  1. 把 Q&A 文档的每一个问答整理成事实条目（保留数字与条件）。
  2. 知乎文章中额外描述的架构细节（入口服务、kvcache-store、各负载均衡器、3FS 的使用）。
  3. Prefill 负载均衡的二次代价模型 `cost(q, c)`：输入、输出、求解方式。
  4. Decode 负载均衡引用的 "DualPath" 是什么？找出处。
  5. GDR 换入是什么意思？为什么 CUDA copy engine 不适合细粒度块拷贝？
  6. deepseek-recipe：支持哪些协议？防特殊 token 注入怎么做？分段并行编码为什么成立？流式解析的状态机怎么设计？用什么语言实现？
  7. 哪些"引擎功能已大部分实现"（README 说 VE/P/D 分离的大部分引擎功能已实现）在代码中能找到痕迹？
- **产出**：按主题分组的事实条目表 + recipe 说明卡。

## R13 vLLM 当前状态

- **服务章节**：第 14 章（主）、第 1 章及各技术章的"对比"小节
- **模板**：T3 对比研究员
- **起点**：vLLM 最新 release 与 main 分支代码、官方文档、官方博客。**以 2026 年的状态为准**，不要引用过时的 V0 架构描述当作现状。
- **必答问题**（每项给出 vLLM 怎么做 + 代码/文档位置）：
  1. 引擎架构（V1 引擎：进程模型、EngineCore、scheduler、worker）。
  2. 调度：continuous batching、chunked prefill、异步调度（async scheduling）现状。
  3. KV 管理：PagedAttention、前缀缓存（哈希块）、混合 KV（滑动窗口、多种 KV 类型）支持、KV offload/connector。
  4. 并行：TP、PP、DP-Attention、EP、EPLB、Wide-EP。
  5. 投机解码：支持的方法（MTP、EAGLE 系等），是否支持变长验证。
  6. 结构化输出后端。
  7. 批不变 / 确定性推理支持情况。
  8. P/D 分离（NIXL、各类 connector）。
  9. 硬件支持范围（含昇腾插件）。
  10. 对 DeepSeek-V4.1-Flash 的支持：recipe、使用的 DeepSeek 内核、团队公布的性能数据（"在 DeepInfer 的 6% 以内"的原始出处、硬件、负载、方法）。
  11. DeepInfer 开源后 vLLM 方面有什么回应或合入。
- **产出**：按维度的事实表，每条带出处。

## R14 SGLang 当前状态

- **服务章节**：同 R13
- **模板**：T3
- **必答问题**：与 R13 相同的维度，另加：
  1. RadixAttention 前缀缓存与 HiCache。
  2. 零开销批调度器（overlap scheduler）的设计与首次发布时间（DeepSeek 文档说参考了它）。
  3. DP-Attention 的起源（SGLang 较早为 DeepSeek 模型做了这一点，需核实）。
  4. SGLang 的 V4.1-Flash 方案（有资料称是带 DSpark、PD 分离、HiCache、router 的预览版镜像，需核实）。
  5. 路由器（sgl-router / model gateway）。
- **产出**：同 R13。

## R15 血统分析：DeepInfer 与 vLLM 的代码关系

- **服务章节**：第 5、14 章
- **模板**：T2
- **依赖**：R03（派生文件列表）、R13
- **必答问题**：
  1. DeepInfer 的 `llm_engine.py`、`worker.py`、`model_runner.py`、`cache_engine.py`、`sequence.py`、`sampler.py` 等文件，最像 vLLM 哪个版本的哪些文件？（这些文件名很像 vLLM V0 时代的结构，需要用代码比对证实或否定）
  2. 选 3–4 个文件做对照：保留了什么、改了什么、删了什么。
  3. DeepInfer 从 SGLang、TensorRT-LLM、NanoFlow 借鉴的具体是什么（以文档和代码为证）？
  4. DeepInfer 没有的 vLLM 通用能力（模型注册表、OpenAI 服务器、多 LoRA、多种量化后端等），是删掉了还是从未有过？
- **产出**：血统对照表 + 结论（注明证据等级）。

## R16 性能数据整理

- **服务章节**：第 2、9、10、14 章
- **模板**：T1
- **必答问题**：
  1. DeepSeek README 中 prefill 与 decode Pareto 图的内容（读图：坐标轴、配置、关键点）；`examples/sweep/run_sweep.sh` 的测试配置。
  2. Q&A 中 EP8 vs EP64 图、DSpark 表的数据。
  3. vLLM 公布的对比数据（硬件、负载、方法、原始链接）。
  4. SGLang 公布的数据（如有）。
  5. 独立第三方测试（如 SemiAnalysis 等，如有）。
  6. 列出这些数字之间**不能**直接比较的原因（硬件、负载、指标定义、是否含 MTP 等）。
- **产出**：性能数据表（每行：数字、指标定义、硬件、配置、负载、来源、等级）+ 可比性说明。

## R17 社区反响与行业评论

- **服务章节**：第 15、17 章
- **模板**：T1
- **范围**：Reddit r/LocalLLaMA、Hacker News、知乎、X 上的技术讨论；vLLM/SGLang 维护者的公开评论；GitHub issues 中的高频问题。
- **必答问题**：
  1. 技术社区最关心的 5–10 个问题是什么？（可直接补充第 17 章的问题清单）
  2. 有信息量的批评和质疑有哪些？
  3. GitHub issues 中暴露的使用障碍（构建失败、硬件限制等）。
- **注意**：只记录有信息量的观点，不记情绪化评论。全部标 D 级。
- **产出**：问题清单 + 观点摘录（带链接）。

## R18 上手条件与配置速查

- **服务章节**：第 16 章、附录 B
- **模板**：T2
- **起点**：`build.sh`、`Makefile`、`requirements*.txt`、`scripts/`、`deepinfer/environ.py`、`deepinfer/config.py`、`deepinfer/engine/arg_utils.py`、`examples/_config.py`、`examples/benchmark.py`、`examples/README.md`。
- **必答问题**：
  1. 构建流程每一步做什么；需要从网络拉取什么；常见失败点（结合 R17 的 issues）。
  2. 全部 `DEEPINFER_*` 环境变量：名称、默认值、含义、所属功能（做成表）。
  3. 引擎参数（`arg_utils.py`、`config.py`）。
  4. 最小可运行配置：单机 8 卡能否运行？需要什么显存？
  5. `benchmark.py` 与 profile 的用法，如何从 trace 中看出用了哪些内核。
  6. 没有 SM100/昇腾 950 时，有哪些部分可以在普通环境中阅读或运行（例如纯 CPU 测试、TileLang 算子编译）？
- **产出**：环境变量表、参数表、构建步骤说明、学习路线建议。
