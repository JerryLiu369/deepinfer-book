# 附录 B 环境变量与配置速查

DeepInfer 内部包含大量的底层性能调优参数，绝大多数参数既可以通过命令行 `EngineArgs` 传入，也可以通过特定的 `DEEPINFER_*` 环境变量直接硬覆盖。下表系统整理了仓库中定义的 64 项核心环境变量及其物理语义：

### 表 B-1 核心环境变量清单（从 `deepinfer/environ.py` 严格整理）

| 环境变量名称 | 数据类型与默认值 | 物理功能与生效模块 | 源码定义与调用位置 |
| --- | --- | --- | --- |
| `DEEPINFER_BATCH_INVARIANT` | `bool` (默认 `0`) | 开启批不变模式，禁用 split-KV/split-K，保证输出完全独立于同批其他请求。 | `environ.py:34`, `sparse_decode.cpp` |
| `DEEPINFER_ENABLE_MTP_VARLEN` | `bool` (默认 `1`) | 开启 DSpark 变长验证投机调度；关闭后回退至固定长度验证。 | `environ.py:46`, `mtp_varlen.hpp` |
| `DEEPINFER_USE_NEXTN_DECODE` | `int` (默认 `5`) | 投机解码最大草稿 Token 长度（V4.1-Flash 硬编码建议为 5）。 | `environ.py:58`, `dflash.py` |
| `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | `int` (默认 `140`) | Decoder 层 Prefill 有界重放保留的末尾 Token 数量（$\ge 134$）。 | `environ.py:82`, `prepare_tensor_input_v4.cpp` |
| `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE` | `str` (默认 `"compact"`) | 有界重放的计算模式（`"compact"` 紧凑切除无用行，`"mask"` 掩码保留）。 | `environ.py:86`, `attn.py` |
| `DEEPINFER_ENABLE_ZOS` | `bool` (默认 `1`) | 开启 ZOS 零开销异步调度流水线；关闭后回退至经典同步串行执行。 | `environ.py:28`, `worker.py` |
| `DEEPINFER_FORCE_LOAD_BALANCE` | `bool` (默认 `0`) | 强制开启 EPLB 专家动态重平衡，定期扫描命中率并搬运权重。 | `environ.py:114`, `eplb/policy.py` |
| `DEEPINFER_EPLB_INTERVAL` | `int` (默认 `100`) | EPLB 触发动态重平衡的计算步（Step）间隔周期。 | `environ.py:118`, `eplb/policy.py` |
| `DEEPINFER_ACCELERATOR` | `str` (默认自动探测) | 底层算子加速后端选择（`"cuda"` 或 `"ascend"`），决定编译产物目录。 | `environ.py:12`, `platform.py` |
| `DEEPINFER_LOG_LEVEL` | `str` (默认 `"INFO"`) | 全局日志输出级别（`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`）。 | `environ.py:18`, `logger.py` |
| `DEEPINFER_USE_TILELANG_KERNEL` | `bool` (默认 `1`) | 是否优先调用 TileLang 编译出的融合高性能算子；关闭后回退 PyTorch。 | `environ.py:40`, `ops.py` |
| `DEEPINFER_MAX_BATCH_SIZE` | `int` (默认 `256`) | 单卡 CUDA Graph 支持的最大请求序列并发数上限。 | `environ.py:64`, `cuda_graph_runner.py` |
| `DEEPINFER_GPU_MEMORY_UTILIZATION` | `float` (默认 `0.90`) | 显存规划高水位线比例，预留 10% 显存防动态突发 OOM。 | `environ.py:70`, `cache_engine.py` |
| `DEEPINFER_SWA_WINDOW_SIZE` | `int` (默认 `128`) | 滑动窗口注意力局部窗口大小，V4.1-Flash 严格要求为 128。 | `environ.py:90`, `v4_modules/attn.py` |
| `DEEPINFER_DISABLE_CUDA_GRAPH` | `bool` (默认 `0`) | 是否强制禁用 CUDA Graph 录制，完全走动态 Kernel Launch（仅用于调试）。 | `environ.py:76`, `model_runner.py` |

### 表 B-2 核心启动参数（`EngineArgs` 与命令行速查）

| 命令行参数 | 类型与默认值 | 对应 EngineArgs 字段 | 含义说明 |
| --- | --- | --- | --- |
| `--model` | `str` (必须) | `model` | Hugging Face 格式的模型本地权重目录路径。 |
| `--ep` | `int` (默认 `8`) | `ep_size` | 专家并行度（EP），单机测试通常设为 8，超节点通常设为 64 或 128。 |
| `--dp` | `int` (默认等于 ep) | `dp_size` | 注意力数据并行度（DP），在 DeepInfer 中必须严格等于 `ep_size`。 |
| `--cp` | `int` (默认 `1`) | `context_parallel_size` | 上下文并行度（CP），生产环境建议保持 1。 |
| `--max-model-len` | `int` (默认 `1048576`) | `max_model_len` | 模型单序列支持的最大上下文长度（最高支持 1M Token）。 |
| `--block-size` | `int` (固定 `128`) | `block_size` | 显存分页块大小，代码硬编码为 128，不可随意更改。 |
| `--tensor-parallel-size` | `int` (固定 `1`) | — | 引擎不支持张量并行，此参数恒定为 1。 |
