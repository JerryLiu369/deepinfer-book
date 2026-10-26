#!/usr/bin/env bash
# 把 manuscript/chapters/*.md 合并构建为 PDF。
# 目录约定：本脚本位于 build/，章节在 manuscript/chapters/，图在 manuscript/figures/，成品写入 dist/。
# 中间 HTML 放在 manuscript/_out/，使章节里的 ../figures/ 相对路径在 HTML 里同样有效。
# 可用环境变量覆盖命令路径：PANDOC、MMDC、WEASYPRINT、CHROME；PDF_ENGINE=weasyprint|chrome。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CH="$ROOT/manuscript/chapters"
FIG="$ROOT/manuscript/figures"
OUTDIR="$ROOT/manuscript/_out"
mkdir -p "$OUTDIR" "$ROOT/dist"

PANDOC="${PANDOC:-pandoc}"
MMDC="${MMDC:-mmdc}"
WEASYPRINT="${WEASYPRINT:-weasyprint}"
CHROME="${CHROME:-chromium}"
PDF_ENGINE="${PDF_ENGINE:-weasyprint}"
TITLE="${TITLE:-读懂 DeepInfer：DeepSeek 的大规模推理引擎}"
HTML="$OUTDIR/book.html"
OUT="${OUT:-$ROOT/dist/deepinfer_book.pdf}"

# 1. 渲染 mermaid 图为 PNG。
#    不用 SVG：mermaid 的 SVG 用 foreignObject 放文字，WeasyPrint 不渲染 foreignObject，文字会丢失。
shopt -s nullglob
mmd_files=("$FIG"/*.mmd)
if [ ${#mmd_files[@]} -gt 0 ]; then
  for f in "${mmd_files[@]}"; do
    png="${f%.mmd}.png"
    if [ ! -f "$png" ] || [ "$f" -nt "$png" ]; then
      echo "渲染 $(basename "$f")"
      "$MMDC" -i "$f" -o "$png" -p "$HERE/puppeteer.json" -b white -s 3
    fi
  done
fi

# 2. 合并章节为 HTML。
chapters=("$CH"/*.md)
if [ ${#chapters[@]} -eq 0 ]; then
  echo "错误：$CH 下没有 .md 文件" >&2
  exit 1
fi
IFS=$'\n' chapters=($(printf '%s\n' "${chapters[@]}" | sort)); unset IFS

# 封面用 --include-before-body 放在目录之前。
"$PANDOC" "${chapters[@]}" \
  --include-before-body "$HERE/cover.html" \
  --from markdown+east_asian_line_breaks+pipe_tables+fenced_divs \
  --to html5 --standalone \
  --toc --toc-depth=2 \
  --metadata title="$TITLE" --metadata lang=zh-CN \
  --css "../../build/book.css" \
  --resource-path="$HERE:$CH:$FIG" \
  --mathml \
  -o "$HTML"

# 3. HTML 转 PDF。
case "$PDF_ENGINE" in
  weasyprint)
    "$WEASYPRINT" --base-url "$OUTDIR/" "$HTML" "$OUT"
    ;;
  chrome)
    "$CHROME" --headless --no-sandbox --disable-gpu \
      --no-pdf-header-footer --print-to-pdf="$OUT" "file://$HTML"
    ;;
  *)
    echo "未知 PDF_ENGINE：$PDF_ENGINE" >&2
    exit 1
    ;;
esac

echo "完成：$OUT"
