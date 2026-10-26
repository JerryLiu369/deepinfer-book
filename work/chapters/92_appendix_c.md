# 附录 C 源码文件索引

为了方便读者在阅读源码时精准对照，下表整理了 DeepInfer 仓库中核心代码文件的路径、核心职责以及全书中对该文件的详细讨论章节：

| 文件绝对路径（相对仓库根目录） | 核心职责与功能定义 | 本书详细讨论章节 |
| --- | --- | --- |
| `deepinfer/entrypoints/llm.py` | 用户入口类 `LLM` 定义，离线批处理 `generate` 接口实现。 | 第 2、5 章 |
| `deepinfer/engine/llm_engine.py` | 引擎控制中枢，负责多 Worker 进程管理、迭代驱动与输出分发。 | 第 5、6 章 |
| `deepinfer/worker/worker.py` | 单卡 Worker 进程执行逻辑，承接共享内存广播，驱动模型运算。 | 第 5、6 章 |
| `deepinfer/worker/model_runner.py` | 模型执行封装层，管理 CUDA Graph 录制重放与底层张量输入组装。 | 第 5、11 章 |
| `deepinfer/worker/cache_engine.py` | 显存缓存引擎，管理 890B 块状 KV、StateBlock 与显存高水位线。 | 第 7、8 章 |
| `deepinfer/worker/zos_status.py` | ZOS 异步调度状态管理，定义 `MetadataZOS` 与输入重映射表。 | 第 6 章 |
| `csrc/engine/scheduler.hpp` | C++ 调度器核心，负责批次构建、队列轮转与前缀划分。 | 第 5、6 章 |
| `csrc/engine/block_manager.hpp` | C++ 显存块管理器，负责空闲链表维护与引用计数管理。 | 第 5、7 章 |
| `csrc/engine/zos.cpp` | C++ 异步流水线底层状态机，生成 `remap` 映射并控制同步屏障。 | 第 6 章 |
| `csrc/engine/mtp_varlen.hpp` | DSpark 变长验证期望吞吐求解器，根据 GPU 延迟曲线裁剪长度。 | 第 10 章 |
| `deepinfer/model_executor/models/deepseek.py` | V4.1-Flash 主干网络定义，调度 40 层 Transformer 与首尾组件。 | 第 3、5 章 |
| `deepinfer/model_executor/models/decoder_layer.py` | 单个 Transformer 解码层骨架，串联 MLA 注意力与 MoE 专家网络。 | 第 3 章 |
| `deepinfer/model_executor/models/v4_modules/attn.py` | MLA 多头潜在注意力与 SWA 局部注意力核心实现，包含第 20 层 KV 投影。 | 第 3、8 章 |
| `deepinfer/model_executor/models/v4_modules/mlp.py` | MoE 专家网络前向实现，集成 DeepEP 通信与 Mega MoE 融合单内核。 | 第 3、9、11 章 |
| `deepinfer/model_executor/models/v4_modules/yoco.py` | YOCO 跨层缓存复用核心，定义有界截断逻辑与紧凑视图。 | 第 3、8 章 |
| `deepinfer/model_executor/models/v4_modules/index_attn.py` | 稀疏索引器（Sparse Indexer）实现，负责粗选 2048 块与精选 512 Token。 | 第 3、7 章 |
| `deepinfer/model_executor/models/v4_modules/hc.py` | mHC 超连接实现，负责 4 残差流混洗与 Sinkhorn 迭代归一化。 | 第 3 章 |
| `deepinfer/model_executor/models/v4_modules/engram.py` | Engram 静态 N-gram 嵌入表，管理 1.57 TB 权重并提供快速语义增强。 | 第 3 章 |
| `deepinfer/model_executor/models/mtp/dflash.py` | DFlash 3 层并行草稿模型定义，包含置信度头 `ConfidenceHead`。 | 第 3、10 章 |
| `deepinfer/model_executor/eplb/policy.py` | EPLB 专家负载均衡策略，实现冗余专家分配的两级贪心装箱算法。 | 第 9 章 |
| `deepinfer/model_executor/eplb/shuffle.py` | EPLB 权重搬运机制，在 NVLink 域内通过对称内存（`symm_mem`）直读复制。 | 第 9 章 |
| `deepinfer/model_executor/layers/sampler.py` | 采样器实现，包含批不变定点采样与绑定序列绝对位置的 RNG 定序。 | 第 12 章 |
| `deepinfer/model_executor/layers/tilelang_ops/ops.py` | TileLang 算子分发中枢，根据硬件平台动态重定向至 `cuda/` 或 `asc/`。 | 第 5、11 章 |
| `deepinfer/hugging_face/config.py` | 模型配置解析器，从 HF 模型卡推导层数、SWA 窗口与压缩比边界。 | 第 3 章 |
| `deepinfer/structure_output/` | 结构化输出支持，集成 xgrammar 自动机生成 Pinned 位掩码。 | 第 12 章 |
| `examples/yoco/yoco_equiv.py` | SWA 有界重放 L1/L2 等价性测试脚本，验证紧凑视图与掩码的一致性。 | 第 8、16 章 |
| `deep_cluster/topology/bind.py` | 硬件亲和性强绑定工具，将进程锁定在物理直连的 CPU 与 NUMA 节点。 | 第 5 章 |
