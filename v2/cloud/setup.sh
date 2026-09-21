#!/bin/zsh
# クラウドの Mac(EC2 Mac)でのセットアップ。~/mb に v2/ pyproject.toml uv.lock を展開し、~/mb/.env を置いた後に実行する。
#   zsh ~/mb/v2/cloud/setup.sh
# やること: uv の導入、環境の記録、SSH 越しに MPS が使えるかの確認。走査は始めない。
set -e
cd ~/mb
[[ -f .env ]] && chmod 600 .env || echo "注意: ~/mb/.env が無い(--wandb を使うなら必要)"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
[[ -f ~/.local/bin/env ]] && source ~/.local/bin/env
echo "--- 環境"
sw_vers
sysctl -n machdep.cpu.brand_string
echo "RAM $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
df -h / | tail -1
echo "--- MPS(SSH 越し・画面にログインしていない状態で使えるか)"
for v in ${@:-2.14.0}; do
  echo "torch $v"
  uv run -q --no-project --python 3.12 --with torch==$v --with numpy python -c "
import torch
print(' version', torch.__version__, '| mps available', torch.backends.mps.is_available(), '| built', torch.backends.mps.is_built())
x = torch.arange(8, dtype=torch.float16, device='mps'); y = (x @ x).item()
print(' matmul on mps:', y, '(expected 140.0)')
a = torch.ones(4, 8, 8, device='mps'); print(' bmm on mps:', torch.bmm(a, a)[0, 0, 0].item(), '(expected 8.0)')
" || echo " -> torch $v は動かない(この macOS では対象外として記録する)"
done
