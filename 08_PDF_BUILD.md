# 08 PDF 构建

## 1. 推荐方案

**Markdown → HTML（pandoc）→ PDF（WeasyPrint，备选 Chromium 无头模式）**。

理由：中文排版用 CSS 控制最简单；mermaid 图先渲染成 PNG 再嵌入；出错时容易排查。

`templates/` 中的脚本和样式已经实际跑通过一次（pandoc 3.x、WeasyPrint 70、mermaid-cli 12、Noto CJK 字体）：封面、可点击且带页码的目录、页眉章节名、页码、中文正文、表格、代码块、mermaid 中文图都能正常输出。

所需工具（按需安装，任选一种安装方式）：

```bash
# pandoc：系统包，或 pip 安装自带二进制的 pypandoc_binary
sudo apt-get install -y pandoc        # 或：pip install pypandoc_binary
# 中文字体
sudo apt-get install -y fonts-noto-cjk
# PDF 引擎
pip install weasyprint
# mermaid 渲染（需要 Node.js）
npm install -g @mermaid-js/mermaid-cli   # 提供 mmdc 命令
```

安装后先运行 `fc-list :lang=zh` 确认中文字体存在。

如果 mermaid-cli 自带的浏览器下载失败，可以设置 `PUPPETEER_SKIP_DOWNLOAD=1` 再安装，然后用 `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome`（或系统中的 chromium 路径）指定已有浏览器。

## 2. 书稿组织约定

- 书稿文件：`work/chapters/` 下，按文件名排序合并。命名：`00_preface.md`、`01_ch01.md` … `17_ch17.md`、`90_appendix_a.md` … `94_appendix_e.md`。
- 每个文件以一级标题开头（`# 第 1 章 推理引擎在做什么`），小节用二级、三级标题。一级标题会自动另起一页，并出现在页眉。
- 图：mermaid 源码写成独立文件 `work/figures/fig-06-02.mmd`，脚本会渲染为同名 `fig-06-02.png`。书稿中**单独成段**引用（前后各空一行）：

  ```markdown
  ![图 6-2 同步调度下的时间线](../figures/fig-06-02.png)
  ```

  单独成段的图片会生成带图注的图。**不要**在 Markdown 里直接写 mermaid 代码块指望自动渲染。
- 用 PNG，不用 SVG：mermaid 生成的 SVG 把文字放在 `foreignObject` 里，WeasyPrint 不渲染它，图上的字会消失。
- 提示框可以用 pandoc 的 fenced div 写法，样式已定义：

  ```markdown
  ::: warning
  本章内容来自 DeepSeek 的部署文档，相关代码没有开源。
  :::
  ```

- 数学公式少用；需要时用 `$...$`，脚本已加 `--mathml`。

## 3. 构建步骤

把 `templates/` 下的全部文件（`build_pdf.sh`、`book.css`、`cover.html`、`puppeteer.json`）复制到 `work/build/`，先把 `cover.html` 中的日期和版本号改成真实值，然后运行：

```bash
cd work/build
bash build_pdf.sh                        # 默认用 WeasyPrint
PDF_ENGINE=chrome CHROME=google-chrome bash build_pdf.sh   # 改用 Chromium
```

命令不在 PATH 里时，可以用环境变量指定：`PANDOC=… MMDC=… WEASYPRINT=… bash build_pdf.sh`。

脚本做三件事：

1. 把 `work/figures/*.mmd` 渲染为 PNG（只重新渲染有改动的）。
2. 用 pandoc 把全部章节合并成一个带目录的 HTML，封面放在目录之前。
3. 把 HTML 转成 `work/build/deepinfer_book.pdf`。

构建日志中的这些警告可以忽略：`Could not load translations for zh-CN`、`The term Abstract has no translation defined`、`Using fontTools instead of HarfBuzz-Subset`、`Ignored overflow-x / user-select`。

## 4. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 中文显示为方块 | 字体未安装或 CSS 中字体名不对。用 `fc-list :lang=zh` 查实际字体名，改 `book.css` 中的 `font-family` |
| mermaid 渲染失败 | 无头浏览器问题。确认 `puppeteer.json` 在 `work/build/`（已含 `--no-sandbox`）；必要时设置 `PUPPETEER_EXECUTABLE_PATH` |
| 图上没有文字 | 用了 SVG。改用脚本生成的 PNG |
| 图片和文字挤在同一行 | 图片引用没有单独成段，前后加空行 |
| 代码块超出页面 | `book.css` 已设置自动换行；仍不好看就缩短代码行 |
| 表格太宽 | 拆表，或用 `::: wide-table` 包住该表（CSS 中已为该类缩小字号） |
| 目录没有页码 | WeasyPrint 支持 `target-counter`，已配置；Chromium 不支持页码目录，只能点击跳转 |

## 5. 备选方案

如果上述方案在环境中无法工作：

- **pandoc + XeLaTeX**：`pandoc ... --pdf-engine=xelatex -V CJKmainfont="Noto Serif CJK SC" -V geometry:margin=2.5cm --toc`。
- **Typst**：pandoc 支持 `-t typst`，需要配置中文字体。

无论用哪种方案，都要按 `07_REVIEW_CHECKLIST.md` 第 3 节打开 PDF 抽查。可以用 PyMuPDF 把若干页渲染成图片来检查：

```bash
python -c "import pymupdf as f; d=f.open('deepinfer_book.pdf'); [d[i].get_pixmap(dpi=80).save(f'page{i}.png') for i in range(0, len(d), max(1, len(d)//10))]"
```
