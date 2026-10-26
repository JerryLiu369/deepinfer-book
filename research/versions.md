# 基础代码版本基线（Versions Baseline）

全书所有源码引用、文件行号与架构分析均基于以下确定的 Commit Hash 与基线版本（记录于 2026-10-23）。

## 1. 核心仓库

| 仓库名称 | 官方地址 | 锁定 Commit / Tag | 提交日期 (ISO) | 备注 |
| --- | --- | --- | --- | --- |
| **DeepInfer** | `https://github.com/deepseek-ai/DeepInfer` | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` | 2026-10-21 12:55:43 +0800 | 主仓库（共4次提交，包含Q&A文档与DeepEP更新） |
| **deepseek-recipe** | `https://github.com/deepseek-ai/deepseek-recipe` | `935817c66f5d6e64838f419692545f06787e485f` (v0.2.0) | 2026-10-20 01:51:02 +0800 | 生产请求编解码与流式解析器 |
| **vLLM** | `https://github.com/vllm-project/vllm` | `2d40c0609fd672e845ddfef0324ab6d8dd616af4` (main, v0.9.2+) | 2026-10-23 10:41:07 +0800 | 通用推理对比对照组（最新main分支） |
| **SGLang** | `https://github.com/sgl-project/sglang` | `79eb10e182c62af6564e5d8aa2b62e19810b6cc2` (main, v0.5.9+) | 2026-10-22 20:01:42 -0700 | 通用推理对比对照组（最新main分支） |

## 2. DeepInfer 子模块（vendor/ Git Submodules）

| 子模块路径 | 上游项目 | 锁定 Submodule Commit | 说明 |
| --- | --- | --- | --- |
| `vendor/DeepEP` | `deepseek-ai/DeepEP` | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` | NVIDIA 平台高吞吐专家并行通信库 |
| `vendor/DeepEP-Ascend` | `deepseek-ai/DeepEP-Ascend` | `3b25377d04b24fc6154698ded78a2bcb2c59afff` | 华为昇腾 950 专家并行通信库 |
| `vendor/DeepGEMM` | `deepseek-ai/DeepGEMM` | `1fc9c66975ee322d628b3288534ca0e9ce12a64d` | NVIDIA 平台稠密与 MoE FP8 GEMM 内核 |
| `vendor/DeepGEMM-Ascend`| `deepseek-ai/DeepGEMM-Ascend` | `3a35326271b069b8b5bf1a754435acd6cc91575b` | 华为昇腾 950 GEMM 内核 |
| `vendor/DeepSelect` | `deepseek-ai/DeepSelect` | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` | Top-K 与稀疏索引选择算子 |
| `vendor/FlashMLA` | `deepseek-ai/FlashMLA` | `2e5429fc5653bab6e081f09477126f731882a6a9` | 多头潜在注意力（MLA）高效解码内核 |
