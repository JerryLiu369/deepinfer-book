# 待核实与待澄清问题清单（Open Questions）

记录调研过程中发现的矛盾、缺失或已澄清的问题与证据链。

| 编号 | 问题描述 | 发现任务 | 涉及章节 | 当前状态与处理建议 |
| --- | --- | --- | --- | --- |
| **Q01** | **EPLB 负载不均衡度数字冲突**：知乎文章给出 prefill `moe_lb=1.0320`（EP4DP4）/ `1.0618`（EP2DP2），而 `doc/inference_deployment_qa.md` 给出 `1.074` / `1.045`。 | R01, R08 | 第 9 章 | **已查明**：代码中实际存在两套口径——`stats_logger.py:238-241` 的 `Σmax / Σmean` 与 `eplb/__init__.py:310` 的 `rank_balance`，且窗口统计取决于日志刷新周期。书中注明两套口径及测试批次差异，不作为单一定论。 |
| **Q02** | **CED 命名与 2,560 token 回放窗口**：代码中无 "CED" 缩写（仅 Q&A 提及），且文档所谓精确重建需跑最后 2,560 token 与 SWA 128/140 窗口换算关系。 | R02b, R07 | 第 3、8 章 | **已查明**：CED 即 YOCO 的 Encoder-Decoder 结构；2,560 tokens 是文档对“精确依赖历史”（20层 × 128 窗口 = 2,560）的理论上限推导，而有界重放直接利用截断将其压缩为常数 128/140。书中详细对比理论上限与有界截断。 |
| **Q03** | **KV 块体积口径差异**：R02b 初算 MXFP8 下为 1,320 B/token，而 R02a 结合 `use_fp4_sparse_kv=True` 精确复现官方 890 B/token（288B main + 68B index-K）。 | R02a, R06 | 第 7 章 | **已查明**：以 `cache_engine.py:327-332` 为准。4 个源层（2/8/14 为 C2 ratio 2，20 为 C1 ratio 1），总体积精确为 `3×(288+68)/2 + (288+68) = 890 B/token`。第 7 章给出完整算式。 |
| **Q04** | **3FS 持久化与 HBM 常驻表述**：技术报告摘要称块状 KV "always in HBM"，而部署 Q&A 详述其分级存入 3FS 独立集群。 | R02a, R06 | 第 4、7、13 章 | **已查明**：decode 运行时全量驻留 GPU HBM（不发生逐步 Host/Disk 换入换出），而跨请求数日长周期持久化（3–7天）与 Prefill 阶段由独立 3FS 存储集群管理。 |
| **Q05** | **开源代码中是否存在二次代价模型与 Quota 分块**：文档 §115 详述的 `compute_quota_limit` 二次代价模型在代码中是否存在。 | R05 | 第 6 章 | **已查明**：开源代码中**无** `compute_quota_limit` 与二次代价求解器。`csrc/engine/flops_estimator.hpp` 的二次项仅用于统计 MFU 的 FLOPs 记账。开源版分块直接按剩余 token 预算与 CP 对齐切分。书中明确指出版权差异。 |
| **Q06** | **开源版是否存在自动前缀缓存（Prefix Caching）**：代码中 `prefix_caching_hit` 与显存块分配的关系。 | R04, R06 | 第 5、7 章 | **已查明**：开源代码内部无自动 Hash/Radix 前缀匹配，`prefix_caching_hit` 仅是 fork/branch 引用计数，仅对外暴露显式接口（`cached_hit_tokens`），自动跨请求命中是未开源的外部 Entrypoint/kvcache-store 职责。 |
