# 附录 D 参考资料

本书所有事实与数据均严格遵循五级证据体系进行核验。下表分类列出全书引用的核心一手与二手参考文献（访问与锁定日期截至 2026 年 10 月 23 日）：

### 一、A 级：代码证据（固定 Commit 源码库）
1. **DeepSeek-AI / DeepInfer**，Commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，`https://github.com/deepseek-ai/DeepInfer`
2. **DeepSeek-AI / deepseek-recipe**，Commit `935817c66f5d6e64838f419692545f06787e485f`，`https://github.com/deepseek-ai/deepseek-recipe`
3. **vLLM Project / vLLM**，Commit `2d40c0609fd672e845ddfef0324ab6d8dd616af4`，`https://github.com/vllm-project/vllm`
4. **SGLang Project / sglang**，Commit `79eb10e182c62af6564e5d8aa2b62e19810b6cc2`，`https://github.com/sgl-project/sglang`
5. **DeepSeek-AI / DeepEP**，Commit `f6aa38fe06cd4a0489e7730365d4e0114970a2f3`
6. **DeepSeek-AI / DeepGEMM**，Commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`
7. **DeepSeek-AI / FlashMLA**，Commit `2e5429fc5653bab6e081f09477126f731882a6a9`

### 二、B 级：官方文档与权威发文
1. **DeepSeek 推理系统部署与架构问答（Inference Deployment Q&A）**，DeepInfer 官方文档 `doc/inference_deployment_qa.md`，2026 年 10 月。
2. **DeepSeek-V4.1-Flash 线上推理系统架构揭秘**，DeepSeek 官方知乎专栏文章，`https://zhuanlan.zhihu.com/p/2095560295930508953`，2026 年 10 月。
3. **DeepSeek 官方开源公告**，DeepSeek 官方 X 账号（@deepseek_ai），2026 年 10 月 20 日。
4. **DeepSeek-V4.1-Flash-1017 模型权重卡与配置文件（config.json）**，Hugging Face 模型中心，`https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`，2026 年 10 月。
5. **vLLM 官方声明关于 GB200 解码吞吐**，vLLM Project 官方 X 账号（@vllm_project），2026 年 10 月 21 日。
6. **SGLang v0.5.21 / v0.5.22 Release Notes 与 V4.1 Cookbook**，SGLang 官方文档，2026 年 10 月。

### 三、C 级：同行评审论文与第三方基准
1. **DSpark 变长投机验证论文**：*Dynamic Confidence-Aware Verification for Speculative Decoding*，arXiv:2607.05147，2026 年 7 月。
2. **DFlash 并行草稿生成论文**：*DFlash: Non-Causal Block Parallel Speculative Decoding*，arXiv:2602.06036，2026 年 2 月。
3. **DualPath 显存直通调度论文**：*DualPath: Eliminating PCIe and DMA Bottlenecks for Disaggregated Inference*，ACM SIGCOMM '26 录用，arXiv:2602.21548，2026 年 2 月。
4. **vLLM 社区 YOCO 截断评测与 PR #58132 / PR #62840**，vLLM GitHub Pull Requests，2026 年 10 月。

### 四、D 级：社区评测与媒体报道（仅作背景参考）
1. **GPUStack 单机 8×B200 推理吞吐对比报告**，2026 年 10 月。
2. **warmctl 8×B300 节点 vLLM 与 DeepInfer 基准实测摘要**，2026 年 10 月。
3. **The New Stack: DeepSeek's Open Source Inference Engine Realities**，2026 年 10 月。
