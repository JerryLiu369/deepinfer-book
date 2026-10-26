#!/usr/bin/env bash
# 还原研究笔记引用的上游源码到 research/sources/，并锁定到 research/versions.md 记录的 commit。
# 用法：bash scripts/fetch_sources.sh [--no-submodules]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/research/sources"
mkdir -p "$DEST"

# 名称 | 仓库 | commit（与 research/versions.md 保持一致）
REPOS=(
  "DeepInfer|https://github.com/deepseek-ai/DeepInfer|9bd9cf243fc53ef9afbf8f04d391d989117d6ab9"
  "deepseek-recipe|https://github.com/deepseek-ai/deepseek-recipe|935817c66f5d6e64838f419692545f06787e485f"
  "vllm|https://github.com/vllm-project/vllm|2d40c0609fd672e845ddfef0324ab6d8dd616af4"
  "sglang|https://github.com/sgl-project/sglang|79eb10e182c62af6564e5d8aa2b62e19810b6cc2"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<<"$entry"
  dir="$DEST/$name"
  if [ ! -d "$dir/.git" ]; then
    echo "克隆 $name"
    git clone --filter=blob:none "$url" "$dir"
  fi
  echo "锁定 $name @ ${commit:0:7}"
  git -C "$dir" fetch --quiet origin "$commit" || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$commit"
done

if [ "${1:-}" != "--no-submodules" ]; then
  echo "初始化 DeepInfer 子模块（vendor/）"
  git -C "$DEST/DeepInfer" submodule update --init --recursive
fi

echo "完成：$DEST"
