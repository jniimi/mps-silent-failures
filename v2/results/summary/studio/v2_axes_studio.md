# v2 の軸の走査(Studio、2026-09-20): all-256 / bmm_bwd / offset / far

Mac Studio(Apple M2 Ultra、Mac14,14、192 GB)、macOS 27.0 (26A428)、torch 2.14.0、Python 3.12。コードは git 2e2e28a(clean)、`v2/scan.py` sha256 abcd1fbb8854…、`v2/worker.py` a264baa58030…(5 run とも manifest に同じ値)。GPU は 1 走査ずつ、`caffeinate -dims` 付きで順に実行。W&B は project `mps-boundary`、run 名 = run_id。ここに書くのは測定の結果だけで、PyTorch / Metal の中の原因には触れない。

## 1. run の一覧

どれも `v2/` で `uv run --env-file ../.env python scan.py --host-tag studio --wandb --wandb-every 25` に下の引数を足した形。

| # | 項目 | run_id | 足した引数 | 行(dry-run の計画) | 時間 | 分類 |
|---|---|---|---|---|---|---|
| 1 | all-256 | 20260920T113543-studio | `--budget-hours 24 --shapes all-256x256x256 --dtypes fp32,fp16` | 228(246) | 11:35:43–12:01:57(26 分、見積もり 50.9 分) | ok 142 / error 78 / wrong 8 |
| 2 | bmm_bwd | 20260920T120157-studio | `--budget-hours 24 --op bmm_bwd --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16` | 604(652) | 12:01:57–13:21:33(80 分、見積もり 160.7 分) | ok 420 / wrong 40 / error 144 |
| 3 | offset | 20260920T132133-studio | `--budget-hours 24 --layouts offset --shapes out-256x64x256,in-256x256x64 --dtypes fp32,fp16` | 104(96) | 13:21:33–13:25:49(4 分、見積もり 5.2 分) | ok 104 |
| 4 | far | 20260920T132549-studio | `--points far_points_full.json` | 124(較正 12 + 112 点) | 13:25:49–13:56:38(31 分) | ok 52 / wrong 32 / error 40 |
| 5 | all-256 の補足 | 20260920T135650-studio | `--points all256_supp_points.json` | 36(較正 6 + 30 点) | 13:56:50–14:11:05(14 分) | ok 12 / wrong 24 |

- 5 run とも manifest の status = done、rc = 0。timeout / crash / truncated / input_corrupt / skipped_memory / skipped_budget は 0 行。入力の読み戻し確認(`input_mismatch_batches`)は全行 0。dry-run の skipped_memory はどの項目も 0 点。
- 行数と計画の差: 項目 1・2 は二分探索が 1 度も発動しなかった(切り替わりは全部、隣り合う探索点の間)。項目 1 は最終点が計画 48 → 54、項目 3 は最終点(`offset_fin`)8 行が計画の外。
- `--budget-hours 24`: 既定の 2.2 時間を超えると最終点の全要素比較が `skipped_budget` になるため(項目 2 の見積もりは 161 分)。コードは変えていない。
- offset だけを回す形は `--layouts offset`(`--with-offset` は標準 4 レイアウトに足す指定)。
- far: `--far 2` は標準の走査を丸ごと回し直す(dry-run で 1276 実行、93.6 分。far の点はその最後で、既定の予算では `skipped_budget` になる)。そこで far の点だけの points ファイルを使った: shape {out-256x64x256, in-256x256x64} × dtype {fp32, fp16} × レイアウト {contig, bT, aT, slice} × B {131071, 131072, 131073}(乱数入力、seed 0)と、B = 131073 の index 符号化 4 種。計 112 点、全部 `compare = full`(dry-run の見積もりが 1 点あたり最大 33 秒だったので全要素比較にした)。
- 項目 5 は当初の計画に無い補足。項目 1 の走査では aT / bT の誤答点(B = 65537、65538)にサンプル比較しか無い(最終点は ok → error の切り替わり 32767 / 32768 に置かれる)ので、all-256x256x256 × {fp32, fp16} × {aT, bT} × B {65537, 65538} と連続の B = 65537 を、seed 0 / 1 / 2 の全要素比較で測った(30 点)。
- 許容誤差(run ごとの較正): fp32 1.15e-5〜1.28e-5、fp16 3.87e-3〜4.21e-3。

## 2. 項目 1 + 5: all-256x256x256(a・b・出力が同時に B·2**16 要素)

fp32 と fp16 で同じ。探索点は fp32 25 点、fp16 17 点(B = 4096〜65538、各境界の ±2 を含む)。

| レイアウト | B ≤ 32767 | 32768 ≤ B ≤ 65536 | B = 65537、65538 |
|---|---|---|---|
| contig | ok | ok | **ok** |
| aT / bT | ok | error | wrong(全バッチ) |
| slice | ok | error | **ok** |

- error 78 行は全部 `RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX`(stage = bmm)。切り替わりは 32767 / 32768 と 65536 / 65537(どちらも隣り合う点の間)。
- 全要素比較(最終点、seed 0 / 1 / 2): contig B = 65538 ok、aT / bT / slice の 32767 ok・32768 error、slice の 65536 error・65537 ok。
- 補足(項目 5、全要素比較、seed 0 / 1 / 2): aT / bT の B = 65537・65538 は 24 行とも誤答の範囲が [0, B)(全バッチ)。最大誤差 2.05〜2.19。仮説の一致率は stride 1.00、wrap 0.0000〜0.0006、wrap+stride は 65537 で 0.969、65538 で 0.938、nooffset 0。壊れた要素の割合は fp32 0.99997、fp16 0.99026。連続の B = 65537 は 6 行(2 dtype × 3 seed)とも ok(最大誤差 fp32 1.3e-6、fp16 4.8e-4)。
- 項目 1 のサンプル比較の wrong 8 行も、比べたバッチ全部が誤答(fp16 271/271、fp32 281/281)、stride 1.00。
- 朝の単発測定(連続 65537 ok、bT 全バッチ誤答)と一致。

## 3. 項目 2: bmm_bwd(forward + backward、out / grad_a / grad_b を別々に比較)

fp32 と fp16 で同じ。

out-256x64x256(出力と G が B·2**16 要素、a・b は B·2**14):

| レイアウト | B ≤ 65536 | B = 65537、65538 |
|---|---|---|
| contig / slice | out・grad_a・grad_b とも ok | out ok、grad_a wrong、grad_b wrong |
| aT / bT | 同上 | out wrong(全バッチ)、grad_a wrong、grad_b wrong |

- 勾配の誤答: B = 65537 の全要素比較(4 レイアウト × 2 dtype × seed 0 / 1 / 2)で、grad_a・grad_b とも誤答の範囲はちょうど [65536, 65537)(最後の 1 バッチ)。最大誤差 1.16〜1.57。勾配の簡易検定の一致率は wrap 1.00、stride 0.00〜0.01(aT の grad_b と bT の grad_a は wrap+stride も 1.00)。B = 65538 はサンプル比較で、最初の誤答バッチが 65536。
- forward の誤答(aT / bT): 全バッチ、stride 1.00、wrap 0.0、最大誤差 2.09〜2.17(全要素比較)。
- 朝の単発(fp16 連続 65537: forward ok、勾配は最後のバッチが wrap)と一致。fp32・4 レイアウト・3 seed に広がった。

in-256x256x64(a が B·2**16 要素、b・出力・G は B·2**14):

| レイアウト | B ≤ 32767 | B ≥ 32768(65538 まで) |
|---|---|---|
| contig / bT | 全部 ok | error、**stage = bwd** |
| aT / slice | 全部 ok | error、stage = bmm(forward) |

- error 144 行は全部 INT_MAX の同じ文言(bwd 72、bmm 72)。切り替わりは 32767 / 32768。朝の単発(fp16 連続 65535 で bwd の例外)と一致。
- データで言えること: 連続な入力でも、a が 2**31 要素以上になると backward が例外になる(forward は通っている)。out 系列では G が 2**32 要素を超えると、forward が正しいレイアウトでも勾配の最後のバッチが誤答になる。
- **解釈であって、このデータでは確かめられないこと**: 勾配の結果は「backward の bmm で G(連続、2**32 超)が規則 3 の大きい operand にあたる」と読めば規則 3 と、in 系列の bwd の例外は「backward が a の転置 view(2**31 以上)を bmm に渡す」と読めば規則 2 と整合する。ただし行には backward の中の operand の情報が無く、勾配の仮説検定も autograd の計算の形を仮定した簡易版なので、整合するとしか言えない。
- **ハーネスの穴**: stage = bwd の例外の行には forward の比較が残らない(`cls_out` などが None)。in 系列の連続・bT の B ≥ 65537 で forward が規則 3 の誤答になっているかどうかは、この run からは分からない(bmm だけの v1 の走査では誤答)。

## 4. 項目 3: offset(大きい storage_offset の view、B = 4096 固定、全要素比較)

| 系列 | pad_batches O の点 | 結果 |
|---|---|---|
| out-256x64x256 fp32 | 1〜262146 の 25 点(offset の境界 65536 / 131072 / 262144 の ±2 を含む) | 全部 ok |
| out-256x64x256 fp16 | 1〜262146 の 17 点 | 全部 ok |
| in-256x256x64 fp32 | 1〜65538 の 25 点(境界 16384 / 32768 / 65536 の ±2) | 全部 ok |
| in-256x256x64 fp16 | 1〜65538 の 17 点 | 全部 ok |

- 104 行(較正 12、offset 84、offset_fin 8)が全部 ok。最大誤差 4.2e-4(fp16)。storage_offset の最大は 4295098368 要素(2**32 + 2**17)、格納の最大は 4563533824 要素。view が 2**31 / 2**32 をまたぐ点(O0 − 2、O0 − 1)も、view 全体が境界の先にある点も ok。pad の側を含めて入力の読み戻しは不一致 0。
- 朝の 4 点と一致。規則 1〜3 のどれも発動しない条件(view は 2**28 要素以下、出力も小さい)で、規則どおり ok。

## 5. 項目 4: far(B = 131071 / 131072 / 131073、全要素比較、seed 0)

乱数入力。fp32 と fp16 で同じ。

| shape | レイアウト | 131071 | 131072 | 131073 |
|---|---|---|---|---|
| out-256x64x256 | contig / slice | ok | ok | ok |
| out-256x64x256 | aT / bT | wrong [0, B) | wrong [0, B) | wrong [0, B) |
| in-256x256x64 | contig | wrong [65536, B) | wrong [65536, B) | wrong [65536, B) |
| in-256x256x64 | bT | wrong [65536, B) | error | error |
| in-256x256x64 | aT / slice | error | error | error |

- out の aT / bT: 全バッチ誤答、stride 1.00、wrap 0.0、最大誤差 2.16〜2.18。出力が 2 × 2**32 要素をまたいでも範囲は変わらない。
- in の連続: 誤答はちょうど [65536, B)(131071 で 65535 バッチ、131072 で 65536、131073 で 65537)、最大誤差 2.22、wrap 1.00、stride 0.0。壊れた要素の割合は fp32 0.49998〜0.49999、fp16 0.4951。仮説検定は誤答バッチの先頭 16 個と末尾 16 個で行うので、B = 131073 では 2 周目のバッチ(131072、a の平坦 index が 2 × 2**32 以上)が検定に入っており、そこを含めて wrap(mod 2**32)が 1.00。2 周目にあたるバッチはこの走査では 1 run につきこの 1 バッチだけ。
- in の bT: 131071 は連続と同じ誤答([65536, 131071)、wrap 1.00、fp16 は stride 0.01)。131072 から例外(b の転置 view が 131072 × 2**14 = 2**31 要素に達する点)。
- in の aT / slice: 3 点とも例外。例外 40 行(乱数 16 + index 符号化 24)は全部 INT_MAX の同じ文言、stage = bmm。
- index 符号化(B = 131073): out の contig / slice は 4 種とも ok、in の aT / bT / slice は 4 種とも error。out の aT は idx_a_rc・idx_b_rc が wrong(全バッチ、stride 1.00)、idx_a_batch・idx_b_batch が ok。out の bT は idx_b_batch だけ ok、他の 3 種は wrong。in の連続は idx_a_batch が wrong([65536, 131073)、wrap 1.00)、他の 3 種は ok。この ok は 7. の不変性で説明がつく。

## 6. 規則の照合(乱数入力の全行、`phase = calibrate` を除く)

v1 の規則(torch 2.14.0): (1) 出力 > 2**32 で operand が転置 → 全バッチ誤答、(2) view の operand(転置 / slice)が 2**31 要素以上 → 例外、(3) 連続な operand > 2**32 → 2**32 より後ろのバッチだけ誤答。3 つの規則がどう重なるかを 3 通りに読んで、全行と突き合わせた。

- A: 1 → 2 → 3 の順。規則 1 は転置があるときだけ発動し、無ければ 2・3 に進む
- B: **出力 > 2**32 なら「転置あり → 全バッチ誤答、なければ ok」で確定。規則 2・3 は出力 ≤ 2**32 のときだけ**
- C: 出力 > 2**32 のとき規則 2 は外すが、規則 3 は残す

判定: ok / error は分類の一致。全バッチ誤答は「比べたバッチが全部誤答」。後ろだけの誤答は、全要素比較なら誤答の範囲がちょうど [ceil(2**32 / バッチあたり要素数), B)、サンプル比較なら誤答が全部その範囲にあり、ヒストグラムのその範囲の bin が全部誤答。

| データ | 行 | A | B | C |
|---|---|---|---|---|
| v1 の全走査 `v1/results/raw/studio/20260919T130741-studio.jsonl` | 1076 | 1066 | **1076** | 1076 |
| 項目 1(all-256 の走査) | 222 | 202 | **222** | 212 |
| 項目 4(far) | 48 | 44 | **48** | 48 |
| 項目 5(all-256 の補足) | 30 | 24 | **30** | 24 |
| 計 | 1376 | 1336 | **1376** | 1360 |

- **B は 4 つのデータの 1376 行すべてと一致**(内訳: ok 1011、error 244、全バッチ誤答 78 = 全要素 54 + サンプル 24、後ろだけの誤答 43 = 全要素 29 + サンプル 14)。
- A の例外 40 行: 出力 > 2**32 の slice で「予測 error、観測 ok」が 24 行(v1 の out-128x64x512 / out-512x64x128 fp32 slice B = 65537・65538 が 10、all-256 slice 65537・65538 が 10、far の out-256x64x256 slice 131072・131073 が 4)、all-256 の連続で「予測 後ろだけ誤答、観測 ok」が 16 行(項目 1 が 10、項目 5 が 6)。
- C の例外 16 行: 上の all-256 の連続の 16 行。
- つまり v1 の 3 規則に足す最小の修正は「出力が 2**32 要素を超える条件では、規則 2 と規則 3 は効かない(転置があれば規則 1、なければ正しい)」。v1 のデータだけでは B と C を区別できず(v1 には出力と入力が同時に 2**32 を超える条件が無い)、区別したのは all-256 の連続の 16 行。
- 照合の範囲: bmm の標準 4 レイアウトの行だけ。offset(規則はどれも発動せず、104 行とも ok)と bmm_bwd(3. のとおり、forward の out は規則 1 と一致、勾配と bwd の例外は解釈つき)は表に入れていない。版は torch 2.14.0、macOS 27.0、この 1 台だけ。

## 7. index 符号化の行と入力の不変性

index 符号化の入力では、読み違いが起きても積が変わらない組み合わせがある(たとえば値が行・列だけで決まり、バッチに依らない入力は、巻き戻りで別のバッチを読んでも同じ)。`v2/check_idx_invariance.py`(無改変。予測の式は上の B と同じ)で、規則 B が誤答を予測する index 符号化の行について、「読み違いのもとでの積」(`worker.hyp_operand`)が正しい積と一致するかを CPU で計算し、記録された分類と突き合わせた。

```
uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python check_idx_invariance.py --run 20260920T132549-studio --results results
uv run --no-project --python 3.12 --with torch==2.14.0 --with numpy python check_idx_invariance.py --results ../v1/results
```

| データ | 規則 B が誤答を予測する index 行 | 不変 → ok | 変わる → wrong | 食い違い |
|---|---|---|---|---|
| far(20260920T132549-studio) | 24 | 12 | 12 | 0 |
| v1 の全走査 | 64 | 36 | 28 | 0 |

- far の「規則 B は誤答を予測するが ok」12 行(out aT の idx_a_batch・idx_b_batch、out bT の idx_b_batch、in 連続の idx_a_rc・idx_b_batch・idx_b_rc、各 2 dtype)は全部、入力が読み違いに対して不変。wrong の 12 行は全部、入力が変わる組み合わせ。v1 の 36 行も同じ(v1 の時点の確認と同じ数)。
- far の index 行 64 のうち残りの 40 行(予測 ok 16、予測 error 24)は規則 B と一致。項目 1・2・3・5 には index 符号化の行は無い。

## 8. 測っていない条件

- far は seed 0 だけ、k = 2 だけ(3 × 2**32 は無い)。a が 2 × 2**32 を超えるのは B = 131073 の 1 バッチ分だけ。all-256 の far(出力と入力が同時に 2 × 2**32)も無い。
- all-256 は 65538 まで。bf16、追加 shape、all-256 の bmm_bwd と offset は無い。
- offset は「格納 = pad + view」だけ(view の後ろに余分がある形、転置・slice との組み合わせ、view 自体が大きい場合は無い)。B = 4096 固定。
- bmm_bwd は基本 shape の 2 つだけ、index 符号化なし。bwd の例外のとき forward の比較が残らない(3.)。
- torch 2.14.0、macOS 27.0、M2 Ultra 192 GB の 1 台だけ。他の版・他のホストでは規則 B を確かめていない。
