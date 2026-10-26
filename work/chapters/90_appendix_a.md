# 附录 A 术语表

本书涉及大量源自前沿学术界、工程界以及 DeepSeek 内部特化的专业技术术语。为便于读者翻阅，下表按拼音字母顺序整理了全书核心术语的标准中英文名称、缩写、一句话物理定义及首次详细讨论章节：

| 术语（中文/英文/缩写） | 物理本质与一句话定义 | 首次出现章节 |
| --- | --- | --- |
| **批不变推理（Batch-Invariant Inference）** | 消除浮点规约顺序对批大小依赖的确定性推理模式，确保单请求输出完全独立于同批其他请求。 | 第 12 章 |
| **超连接（Hyper-Connections，mHC）** | 将传统单残差流拓宽为多子流（如 4 流）并在子层前后执行高频混洗的深度网络连接架构。 | 第 3 章 |
| **草稿模型（Draft Model）** | 投机解码中负责先行低成本预测未来若干候选 Token 的轻量级前向模块（如 DFlash）。 | 第 1 章、第 10 章 |
| **多头潜在注意力（Multi-Head Latent Attention，MLA）** | 将 Key 与 Value 投影至低维共享潜在空间（Latent Vector）的高压缩比注意力机制。 | 第 1 章、第 3 章 |
| **滑动窗口注意力（Sliding Window Attention，SWA）** | 每一个 Token 仅关注自身前文固定局部窗口（如 128 Token）的受限自注意力机制。 | 第 3 章、第 8 章 |
| **镜像上下文并行（Mirror Context Parallelism，Mirror CP）** | 将长序列首尾两段对偶绑定至同一个计算节点，以平抑因果自注意力下三角计算倾斜的并行方案。 | 第 9 章 |
| **宽专家并行（Wide Expert Parallelism，Wide-EP）** | 在 Decode 阶段跨数十张甚至上百张卡（如 EP64–128）极度切分 MoE 专家以成倍摊薄权重访存的架构。 | 第 4 章、第 9 章 |
| **连续批处理（Continuous Batching）** | 打破请求强绑定、在单步迭代级别动态纳管请求进出以彻底消除尾部算力气泡的调度机制。 | 第 1 章 |
| **首字延迟（Time to First Token，TTFT）** | 客户端从发送 Prompt 到收到大语言模型输出的第一个 Token 所经历的时间（由 Prefill 决定）。 | 第 1 章 |
| **微秒级直通显存（GPUDirect RDMA，GDR）** | 网络接口卡越过 CPU 内存中转，直接通过 PCIe 总线与 GPU 显存执行零拷贝 DMA 传输的硬件技术。 | 第 13 章 |
| **有界重放（Bounded Replay）** | 在状态缺失或跨层时，仅重算末尾固定窗口（128 Token）并强制截断历史以换取常数开销的近似重构机制。 | 第 8 章 |
| **预填充阶段（Prefill Phase）** | 推理引擎一次性处理用户输入 Prompt 的大矩阵计算阶段，本质属于计算密集型（Compute-Bound）。 | 第 1 章 |
| **解码阶段（Decode Phase）** | 推理引擎逐字自回归生成回答的向量矩阵计算阶段，本质属于显存带宽受限型（Memory-Bound）。 | 第 1 章 |
| **专家负载均衡器（Expert Parallelism Load Balancer，EPLB）** | 监控 MoE 各专家在线命中率，通过冗余副本与两级贪心装箱动态消解热点倾斜的调度算法。 | 第 9 章 |
| **字间延迟（Time Between Tokens，TBT）** | 模型在自回归流式生成过程中，相邻两个输出 Token 之间的物理时间间隔（由 Decode 决定）。 | 第 1 章 |
| **置信度头（Confidence Head）** | DFlash 模块末端用于预测每个草稿 Token 被主模型接纳概率的轻量级单层标量投影网络。 | 第 3 章、第 10 章 |
| **装箱策略（Bin-Packing Policy）** | EPLB 中以最小化全局单卡最大负载为目标，将热门专家冗余实例安插进空闲卡的核心组合优化算法。 | 第 9 章 |
| **阻断式注入防护（Special Token Sanitization）** | 在分词阶段强制剥离新增词表中的控制字符，防止外部恶意输入构造提示词越狱的安全防护技术。 | 第 2 章、第 13 章 |
| **3FS 分布式文件系统（Fire-Flyer File System）** | DeepSeek 自研的高性能分布式闪存存储集群，为上万计算节点提供数十 TiB/s 级别的 KV 共享池。 | 第 4 章、第 13 章 |
| **CED 架构（Compress-Encoder-Decoder）** | 将全模型划分为前 20 层 Encoder 与后 20 层 Decoder、由第 20 层一次性投影共享块状 KV 的骨干架构。 | 第 3 章、第 8 章 |
| **DFlash 模块** | 挂载在主干顶部的 3 个非因果轻量草稿层，一次并行前向产出 5 个候选 Token 的投机生成器。 | 第 3 章、第 10 章 |
| **DSpark 算法** | 结合置信度头存活率与 GPU 实测延迟曲线，动态为每个请求求解最优验证长度的投机调度算法。 | 第 10 章 |
| **Engram 嵌入增强** | 依托外挂的 1.57 TB 静态多哈希 N-gram 查找表，直接提取固定词组语义的低计算复杂度旁路模块。 | 第 3 章 |
| **TileLang 语言** | 一种将高级 Python 语法自动编译优化为英伟达 CUDA 与华为昇腾达芬奇汇编的领域特定开发语言。 | 第 5 章、第 11 章 |
| **YOCO 架构（You Only Cache Once）** | Decoder 层不持有独立块状 KV，而是完全别名复用 Encoder 隐状态投影输出的跨层缓存复用设计。 | 第 3 章、第 8 章 |
| **ZOS 调度器（Zero-Overhead Scheduling）** | CPU 提前排定下一步前向计算并由 GPU 就地补全未知状态的零开销异步批处理流水线机制。 | 第 5 章、第 6 章 |
