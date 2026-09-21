#!/bin/zsh
# torch の各版で check_mps.py を実行する。結果は results/ に版ごとの JSON。
cd "$(dirname "$0")"
versions=(2.4.1 2.5.1 2.6.0 2.7.1 2.8.0 2.9.1 2.10.0 2.11.0 2.12.1 2.13.0 2.14.0)
(( $# )) && versions=("$@")
for v in $versions; do
  echo "=== torch $v"
  uv run -q --no-project --python 3.12 --with torch==$v --with numpy python check_mps.py 2>&1 | grep -v "^#"
done
