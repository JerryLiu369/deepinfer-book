# 附录 E 版本信息

全书所有文字、数字计算、架构图与代码行号均严格对齐以下软件与硬件基线：

### 核心项目代码版本台账（锁定于 2026-10-23）

| 软件仓库名称 | 官方托管地址 | 锁定 Commit Hash | 提交日期与时区 | 说明 |
| --- | --- | --- | --- | --- |
| **DeepInfer** | `deepseek-ai/DeepInfer` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` | 2026-10-21 12:55:43 +0800 | 主仓库，包含 4 次提交 |
| **deepseek-recipe** | `deepseek-ai/deepseek-recipe` | `935817c66f5d6e64838f419692545f06787e485f` | 2026-10-20 01:51:02 +0800 | Release v0.2.0 |
| **vLLM** | `vllm-project/vllm` | `2d40c0609fd672e845ddfef0324ab6d8dd616af4` | 2026-10-23 10:41:07 +0800 | main 分支，包含 Triton MTP PR |
| **SGLang** | `sgl-project/sglang` | `79eb10e182c62af6564e5d8aa2b62e19810b6cc2` | 2026-10-22 20:01:42 -0700 | main 分支，包含 v0.5.9+ |

### 底层算子子模块版本（Git Submodules）

| 子模块目录路径 | 上游项目名称 | 锁定 Commit Hash | 关联特性 |
| --- | --- | --- | --- |
| `vendor/DeepEP` | `deepseek-ai/DeepEP` | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` | NVIDIA 平台高吞吐 All-to-All 专家通信 |
| `vendor/DeepEP-Ascend` | `deepseek-ai/DeepEP-Ascend` | `3b25377d04b24fc6154698ded78a2bcb2c59afff` | 华为昇腾 950 平台 HCCL 专家通信 |
| `vendor/DeepGEMM` | `deepseek-ai/DeepGEMM` | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` | NVIDIA 平台 FP8 矩阵乘与 Mega MoE 融合 |
| `vendor/DeepGEMM-Ascend`| `deepseek-ai/DeepGEMM-Ascend` | `3a35326271b069b8b5bf1a754435acd6cc91575b` | 华为昇腾 950 平台 GEMM 内核 |
| `vendor/DeepSelect` | `deepseek-ai/DeepSelect` | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | 硬件级稀疏 Indexer 与 Top-K 索引选择 |
| `vendor/FlashMLA` | `deepseek-ai/FlashMLA` | `2e5429fc5653bab6e081f09477126f731882a6a9` | MLA 低维共享潜在空间解码加速内核 |

### 构建与排版环境工具链

- **Pandoc**：3.1.3
- **WeasyPrint**：62.3（Python 3.12 原生 PDF 引擎）
- **Node.js**：v20.18.0
- **Mermaid CLI（`mmdc`）**：12.0.0
- **中文字体**：Noto Serif CJK SC（思源宋体，Bold / Regular 完整字形集）
- **无头浏览器**：Google Chrome 133.0.6943.126
