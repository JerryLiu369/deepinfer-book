# 读懂 DeepInfer：DeepSeek 的大规模推理引擎

[中文](#中文) · [English](#english)

---

## 中文

这是一本关于 [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) 推理引擎的中文学习书，同时也是它完整的制作记录。

**成品**：[`dist/deepinfer_book.pdf`](dist/deepinfer_book.pdf)，共 17 章、前言和 5 个附录。

### 这本书讲什么

DeepInfer 不是另一个通用推理框架。它围绕一个模型（DeepSeek-V4.1-Flash）和一套特定的集群拓扑设计，目标是把每个 token 的服务成本压到最低。本书沿着"一个请求的一生"，讲清楚它的每个核心机制**为什么这样设计，又付出了什么代价**：

| 部分 | 章节 | 内容 |
| --- | --- | --- |
| 认识对象 | 1–5 | 推理引擎在做什么；DeepInfer 是什么、不是什么；它服务的模型；系统全貌；代码地图 |
| 核心机制 | 6–12 | 调度、KV 与状态缓存、SWA 有界重放、并行与专家负载均衡、投机解码、内核与双硬件、结果可信 |
| 边界与比较 | 13–15 | 没有开源的那一半、与 vLLM / SGLang 的异同、评价 |
| 上手 | 16–17 | 如何学习；二十个值得问的问题 |
| 附录 | A–E | 术语、配置、源码索引、参考资料、版本信息 |

### 这本书怎么来的

全书由 AI 代理按 `spec/` 里的任务书完成：主代理负责规划和写作，子代理负责读源码、查资料，只交"事实笔记"，不写书稿。过程中的每一步都保留在这个仓库里，方便追溯和复现。

几条需要读者知道的前提：

- **读码，不是实测。** 所有结论来自对固定版本源码和公开资料的阅读，没有人在文中提到的硬件上运行过 DeepInfer。
- **版本固定。** 代码基线是 `deepseek-ai/DeepInfer @ 9bd9cf2`（2026-10-21），完整清单见 [`research/versions.md`](research/versions.md)。之后上游的变化不在本书范围内。
- **证据分级。** 每个事实标注等级（代码证实、官方文档、第三方实测等），规则见 [`spec/00_MASTER_SPEC.md`](spec/00_MASTER_SPEC.md) 第 6 节。"代码里有"和"文档里说"在书中严格区分。
- **可能有错。** 书稿按 `spec/07_REVIEW_CHECKLIST.md` 自查（目前只有第 6 章留有正式审校记录，见 `process/reviews/`），仍是 AI 产出，请对关键数字回到笔记和源码核对。发现问题欢迎提 issue。

### 仓库结构

```
deepinfer-book/
├── dist/            # 成品：deepinfer_book.pdf
├── manuscript/      # 书稿
│   ├── chapters/    #   各章 Markdown 源文件
│   ├── figures/     #   图（mermaid 源文件 .mmd 与渲染后的 .png）
│   └── briefs/      #   每章写作前的简报
├── research/        # 调研
│   ├── notes/       #   R01–R18 事实笔记（带出处和证据等级）
│   ├── versions.md  #   锁定的上游版本
│   ├── open_questions.md  # 调研中发现的矛盾及结论
│   └── sources/     #   上游源码克隆（不入库）
├── spec/            # 任务书：目标、大纲、文风、调研任务、验收清单
├── process/         # 过程记录：开工计划、核心论点、审校记录
├── build/           # 排版模板与 PDF 构建脚本
└── scripts/         # 辅助脚本
```

从哪里开始读：

- **只想读书**：打开 `dist/deepinfer_book.pdf`，或直接读 `manuscript/chapters/`。
- **想核对某个结论**：在 `research/notes/` 找对应章节的笔记，里面有 `路径:行号` 出处。
- **想了解怎么做出来的**：从 [`spec/00_MASTER_SPEC.md`](spec/00_MASTER_SPEC.md) 读起，再看 `process/`。
- **想复用这套方法**：`spec/` 里的文件互相引用，建议按 `00 → 08` 的编号顺序阅读。

### 复现

还原上游源码（用于核对笔记里的 `路径:行号`）：

```bash
bash scripts/fetch_sources.sh        # 克隆并锁定到 research/versions.md 记录的 commit
```

重新构建 PDF，需要 pandoc、WeasyPrint、mermaid-cli（`mmdc`）和 Noto CJK 字体，详见 [`spec/08_PDF_BUILD.md`](spec/08_PDF_BUILD.md)：

```bash
bash build/build_pdf.sh              # 输出 dist/deepinfer_book.pdf
```

### 相关仓库

- [deepinfer-video](https://github.com/JerryLiu369/deepinfer-video)：同一主题的配套视频系列，复用本仓库的 `research/` 作为知识库。

---

## English

A Chinese-language study book on the [DeepSeek DeepInfer](https://github.com/deepseek-ai/DeepInfer) inference engine, published together with the full record of how it was produced.

**Deliverable**: [`dist/deepinfer_book.pdf`](dist/deepinfer_book.pdf) (preface, 17 chapters, 5 appendices; written in Chinese).

### What the book covers

DeepInfer is not a general-purpose serving framework. It is built around a single model (DeepSeek-V4.1-Flash) and a specific cluster topology, with the goal of minimizing the marginal cost per token. Following the life of one request, the book explains **why each core mechanism is designed the way it is, and what it costs**: scheduling, KV and state caches, bounded replay for sliding-window attention, parallelism and expert load balancing, speculative decoding, kernels for two hardware families, and result reproducibility. It also separates what the open-source code actually contains from what only the documentation describes, and compares DeepInfer with vLLM and SGLang.

### How it was made

The book was produced by AI agents following the task package in `spec/`. A lead agent planned and wrote the text. Sub-agents read source code and sources and handed back fact notes only; they did not write the manuscript. Every step is kept in this repository for traceability.

What readers should keep in mind:

- **Code reading, not benchmarking.** Conclusions come from reading a pinned source revision and public material. Nobody ran DeepInfer on the hardware discussed.
- **Pinned versions.** The code baseline is `deepseek-ai/DeepInfer @ 9bd9cf2` (2026-10-21). See [`research/versions.md`](research/versions.md). Later upstream changes are out of scope.
- **Graded evidence.** Each fact carries an evidence level (code-verified, official docs, third-party measurement, and so on). See section 6 of [`spec/00_MASTER_SPEC.md`](spec/00_MASTER_SPEC.md).
- **May contain errors.** Chapters were self-checked against the review checklist (only chapter 6 has a formal review record in `process/reviews/`), and the text is AI-generated. Please check key numbers against the notes and the source. Issues are welcome.

### Repository layout

| Path | Contents |
| --- | --- |
| `dist/` | Final PDF |
| `manuscript/` | Chapters (Markdown), figures (mermaid sources and PNGs), per-chapter briefs |
| `research/` | Fact notes R01–R18, version pins, open questions, upstream source clones (not tracked) |
| `spec/` | The task package: goals, outline, style guide, research tasks, review checklists |
| `process/` | Plan, thesis, review records |
| `build/` | Typesetting templates and the PDF build script |
| `scripts/` | Helper scripts |

### Reproduce

```bash
bash scripts/fetch_sources.sh    # clone upstream repos at the pinned commits into research/sources/
bash build/build_pdf.sh          # rebuild dist/deepinfer_book.pdf (needs pandoc, WeasyPrint, mmdc, Noto CJK)
```

### Related

- [deepinfer-video](https://github.com/JerryLiu369/deepinfer-video): a companion video series on the same subject. It uses `research/` from this repository as its knowledge base.
