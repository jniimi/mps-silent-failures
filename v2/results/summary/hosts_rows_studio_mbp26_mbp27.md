# ホストをまたぐ行ごとの照合

基準: **studio**。組: studio vs mbp-macos26, studio vs mbp-macos27, mbp-macos26 vs mbp-macos27。`uv run python compare_rows.py --source results:studio --source results:mbp-macos26 --source results:mbp-macos27 --all-pairs` で生成(`v2/` で実行)。

同じ点 = (op, shape, dtype, レイアウト, 入力の種類, B, seed, 比較の方法, pad_batches) が同じ行(op が無い行は bmm、pad_batches が無い行は None)。status = done、scale_log2 = 0 の run の全部を使う。較正の行(phase = calibrate)は照合から除き、数だけ数える。phase は点の定義に入れない。同じ点が 2 回あれば後の行を残す。共通の点について、下の「外した項目」以外の全項目が等しいかを見る(片方にしかない項目も違いに数える)。同じか違うかだけを書き、解釈はしない。

系列 = (op, torch の公開版, mode, 較正以外の行の shape の集合, レイアウトの集合, --points の run なら指定点のリストの sha256)が同じ run の集まり(指定点の sha256 は label を除き、整列した JSON から計算する)。系列の名前は表示用で、source の間の対応づけは signature(表の sig 列はその sha256 の先頭 8 桁)で行う。

外した項目:

- 識別(run ごとに違う): `run_id`, `fingerprint`, `host_tag`, `ts`
- 環境(ホストの記述そのもの): `os_release`, `macos`, `macos_build`, `hw_model`, `chip`, `mem_bytes`, `linux_distro`, `python`, `git_rev`, `git_dirty_code`, `mem_limit_host_bytes`, `mem_limit_dev_bytes`, `mem_limit_note`
- 時間とメモリの実測(実行ごとに揺れる): `time`, `max_rss`, `wall_sec`, `dev_driver_alloc`, `dev_current_alloc`
- crash の stderr(Apple の build root のパスが入る): `stderr_tail`

## 対象

| host-tag | 系列 | sig | run | status | 機種 / チップ / OS | torch | 行数 | うち較正 | 重複 | 比較に使う点 | 分類の内訳(全行) | revision | sha_scan / sha_worker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| studio | main sweep 2.14.0 | 50429b85 | 20260920T150512-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 1596 | 24 | 0 | 1572 | WRONG 97, err 214, ok 1285 | 4423ad9 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.4.1 | bc628586 | 20260920T192435-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.4.1 | 360 | 12 | 0 | 348 | TRUNC 16, WRONG 40, ok 304 | 5441bf1 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.5.1 | cf1c2f70 | 20260920T200227-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.5.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | 396026b | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.6.0 | 76b73696 | 20260920T203904-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.6.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.7.1 | d1515268 | 20260920T211533-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.7.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.8.0 | 4d6ca121 | 20260920T214629-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.8.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.9.1 | e3c699b2 | 20260920T221708-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.9.1 | 360 | 12 | 0 | 348 | WRONG 32, err 54, ok 274 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.10.0 | 6f87aa24 | 20260920T224546-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.10.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.11.0 | d70cfc65 | 20260920T231439-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.11.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.12.1 | 574a7d0c | 20260920T234329-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.12.1 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | reduced sweep 2.13.0 | 0925729a | 20260921T001206-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.13.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | f195052 | abcd1fbb8854 / a264baa58030 |
| studio | backward 2.14.0 | 74e1ba2c | 20260920T120157-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 604 | 12 | 0 | 592 | WRONG 40, err 144, ok 420 | 2e2e28a | abcd1fbb8854 / a264baa58030 |
| studio | all-256 2.14.0 | 6c54d649 | 20260920T113543-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 228 | 6 | 0 | 222 | WRONG 8, err 78, ok 142 | 2e2e28a | abcd1fbb8854 / a264baa58030 |
| studio | all-256 supplement 2.14.0 | e0fe34d7 | 20260920T135650-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 36 | 6 | 0 | 30 | WRONG 24, ok 12 | 2e2e28a | abcd1fbb8854 / a264baa58030 |
| studio | far points 2.14.0 | 46f0408b | 20260920T132549-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 124 | 12 | 0 | 112 | WRONG 32, err 40, ok 52 | 2e2e28a | abcd1fbb8854 / a264baa58030 |
| studio | offset 2.14.0 | e12f206c | 20260920T132133-studio | done | Mac14,14 / Apple M2 Ultra / macOS 27.0 (26A428) | 2.14.0 | 104 | 12 | 0 | 92 | ok 104 | 2e2e28a | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | main sweep 2.14.0 | 50429b85 | 20260920T110142-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 1596 | 24 | 0 | 1572 | WRONG 97, err 214, ok 1285 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.4.1 | bc628586 | 20260920T145220-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.4.1 | 360 | 12 | 0 | 348 | TRUNC 16, WRONG 40, ok 304 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.5.1 | cf1c2f70 | 20260920T193350-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.5.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.6.0 | 76b73696 | 20260920T200732-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.6.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.7.1 | d1515268 | 20260920T204053-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.7.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.8.0 | 4d6ca121 | 20260920T152603-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.8.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.9.1 | e3c699b2 | 20260920T155714-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.9.1 | 360 | 12 | 0 | 348 | WRONG 32, err 54, ok 274 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.10.0 | 6f87aa24 | 20260920T211244-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.10.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.11.0 | d70cfc65 | 20260920T214215-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.11.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.12.1 | 574a7d0c | 20260920T221148-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.12.1 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | reduced sweep 2.13.0 | 0925729a | 20260920T224058-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.13.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | backward 2.14.0 | 74e1ba2c | 20260920T162628-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 604 | 12 | 0 | 592 | WRONG 40, err 144, ok 420 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | all-256 2.14.0 | 6c54d649 | 20260920T183611-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 228 | 6 | 0 | 222 | WRONG 8, err 78, ok 142 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | all-256 supplement 2.14.0 | e0fe34d7 | 20260921T010713-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 36 | 6 | 0 | 30 | WRONG 24, ok 12 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | far points 2.14.0 | 46f0408b | 20260920T231012-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 124 | 12 | 0 | 112 | WRONG 32, err 40, ok 52 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos26 | offset 2.14.0 | e12f206c | 20260920T234618-mbp-macos26 | done | Mac14,6 / Apple M2 Max / macOS 26.6.2 (25G83) | 2.14.0 | 104 | 12 | 0 | 92 | ok 104 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | main sweep 2.14.0 | 50429b85 | 20260921T022455-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 1596 | 24 | 0 | 1572 | WRONG 97, err 214, ok 1285 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.4.1 | bc628586 | 20260921T042254-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.4.1 | 360 | 12 | 0 | 348 | TRUNC 16, WRONG 40, ok 304 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.5.1 | cf1c2f70 | 20260921T045656-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.5.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.6.0 | 76b73696 | 20260921T053027-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.6.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.7.1 | d1515268 | 20260921T060357-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.7.1 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.8.0 | 4d6ca121 | 20260921T063556-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.8.0 | 374 | 12 | 0 | 362 | CRASH 40, WRONG 42, err 10, ok 282 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.9.1 | e3c699b2 | 20260921T070727-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.9.1 | 360 | 12 | 0 | 348 | WRONG 32, err 54, ok 274 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.10.0 | 6f87aa24 | 20260921T073654-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.10.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.11.0 | d70cfc65 | 20260921T080637-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.11.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.12.1 | 574a7d0c | 20260921T083617-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.12.1 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | reduced sweep 2.13.0 | 0925729a | 20260921T090540-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.13.0 | 360 | 12 | 0 | 348 | WRONG 32, err 52, ok 276 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | backward 2.14.0 | 74e1ba2c | 20260921T093501-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 604 | 12 | 0 | 592 | WRONG 40, err 144, ok 420 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | all-256 2.14.0 | 6c54d649 | 20260921T105732-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 228 | 6 | 0 | 222 | WRONG 8, err 78, ok 142 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | all-256 supplement 2.14.0 | e0fe34d7 | 20260921T112505-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 36 | 6 | 0 | 30 | WRONG 24, ok 12 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | far points 2.14.0 | 46f0408b | 20260921T114054-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 124 | 12 | 0 | 112 | WRONG 32, err 40, ok 52 | ed941f3 | abcd1fbb8854 / a264baa58030 |
| mbp-macos27 | offset 2.14.0 | e12f206c | 20260921T121710-mbp-macos27 | done | Mac14,6 / Apple M2 Max / macOS 27.0 (26A428) | 2.14.0 | 104 | 12 | 0 | 92 | ok 104 | ed941f3 | abcd1fbb8854 / a264baa58030 |

行数 = run の行の全部(較正と重複を含む)。重複 = 同じ点が 2 回あった数(後の行を使う)。比較に使う点 = 較正を除き、重複をまとめた後の点の数。

## 照合

A = 組の左、B = 組の右。manifest = その系列の run の sha_scan / sha_worker / tol / torch(公開版)/ points が A と B で同じか(違う項目があれば名前を書く)。

### studio vs mbp-macos26

| 系列 | sig | 共通の点 | A にしかない点 | B にしかない点 | 全項目が同じ点 | 違う点 | 違った項目(点の数) | manifest |
|---|---|---|---|---|---|---|---|---|
| main sweep 2.14.0 | 50429b85 | 1572 | 0 | 0 | 1572 | 0 | - | 同じ |
| reduced sweep 2.4.1 | bc628586 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.5.1 | cf1c2f70 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.6.0 | 76b73696 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.7.1 | d1515268 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.8.0 | 4d6ca121 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.9.1 | e3c699b2 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.10.0 | 6f87aa24 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.11.0 | d70cfc65 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.12.1 | 574a7d0c | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.13.0 | 0925729a | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| backward 2.14.0 | 74e1ba2c | 592 | 0 | 0 | 592 | 0 | - | 同じ |
| all-256 2.14.0 | 6c54d649 | 222 | 0 | 0 | 222 | 0 | - | 同じ |
| all-256 supplement 2.14.0 | e0fe34d7 | 30 | 0 | 0 | 30 | 0 | - | 同じ |
| far points 2.14.0 | 46f0408b | 112 | 0 | 0 | 112 | 0 | - | 同じ |
| offset 2.14.0 | e12f206c | 92 | 0 | 0 | 92 | 0 | - | 同じ |
| **合計(16 系列)** | | 6156 | 0 | 0 | 6156 | 0 | | 同じ |

### studio vs mbp-macos27

| 系列 | sig | 共通の点 | A にしかない点 | B にしかない点 | 全項目が同じ点 | 違う点 | 違った項目(点の数) | manifest |
|---|---|---|---|---|---|---|---|---|
| main sweep 2.14.0 | 50429b85 | 1572 | 0 | 0 | 1572 | 0 | - | 同じ |
| reduced sweep 2.4.1 | bc628586 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.5.1 | cf1c2f70 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.6.0 | 76b73696 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.7.1 | d1515268 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.8.0 | 4d6ca121 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.9.1 | e3c699b2 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.10.0 | 6f87aa24 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.11.0 | d70cfc65 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.12.1 | 574a7d0c | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.13.0 | 0925729a | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| backward 2.14.0 | 74e1ba2c | 592 | 0 | 0 | 592 | 0 | - | 同じ |
| all-256 2.14.0 | 6c54d649 | 222 | 0 | 0 | 222 | 0 | - | 同じ |
| all-256 supplement 2.14.0 | e0fe34d7 | 30 | 0 | 0 | 30 | 0 | - | 同じ |
| far points 2.14.0 | 46f0408b | 112 | 0 | 0 | 112 | 0 | - | 同じ |
| offset 2.14.0 | e12f206c | 92 | 0 | 0 | 92 | 0 | - | 同じ |
| **合計(16 系列)** | | 6156 | 0 | 0 | 6156 | 0 | | 同じ |

### mbp-macos26 vs mbp-macos27

| 系列 | sig | 共通の点 | A にしかない点 | B にしかない点 | 全項目が同じ点 | 違う点 | 違った項目(点の数) | manifest |
|---|---|---|---|---|---|---|---|---|
| main sweep 2.14.0 | 50429b85 | 1572 | 0 | 0 | 1572 | 0 | - | 同じ |
| reduced sweep 2.4.1 | bc628586 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.5.1 | cf1c2f70 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.6.0 | 76b73696 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.7.1 | d1515268 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.8.0 | 4d6ca121 | 362 | 0 | 0 | 362 | 0 | - | 同じ |
| reduced sweep 2.9.1 | e3c699b2 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.10.0 | 6f87aa24 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.11.0 | d70cfc65 | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.12.1 | 574a7d0c | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| reduced sweep 2.13.0 | 0925729a | 348 | 0 | 0 | 348 | 0 | - | 同じ |
| backward 2.14.0 | 74e1ba2c | 592 | 0 | 0 | 592 | 0 | - | 同じ |
| all-256 2.14.0 | 6c54d649 | 222 | 0 | 0 | 222 | 0 | - | 同じ |
| all-256 supplement 2.14.0 | e0fe34d7 | 30 | 0 | 0 | 30 | 0 | - | 同じ |
| far points 2.14.0 | 46f0408b | 112 | 0 | 0 | 112 | 0 | - | 同じ |
| offset 2.14.0 | e12f206c | 92 | 0 | 0 | 92 | 0 | - | 同じ |
| **合計(16 系列)** | | 6156 | 0 | 0 | 6156 | 0 | | 同じ |

## 違う点・片方にしかない点

- なし

## 外した項目のうち、実際に値が違った点の数

共通の点のうち、外した項目の値が A と B で違った点の数(項目ごと。片方にしかない場合も数える)。ここに無い外した項目は、全部の共通の点で値が同じだった。

- studio vs mbp-macos26(共通の 6156 点): `run_id` 6156, `fingerprint` 6156, `host_tag` 6156, `ts` 6156, `os_release` 6156, `macos` 6156, `macos_build` 6156, `hw_model` 6156, `chip` 6156, `mem_bytes` 6156, `python` 5996, `git_rev` 6156, `git_dirty_code` 6156, `mem_limit_host_bytes` 6156, `time` 5996, `max_rss` 5991, `wall_sec` 6084, `dev_driver_alloc` 143, `stderr_tail` 160
- studio vs mbp-macos27(共通の 6156 点): `run_id` 6156, `fingerprint` 6156, `host_tag` 6156, `ts` 6156, `hw_model` 6156, `chip` 6156, `mem_bytes` 6156, `python` 5996, `git_rev` 6156, `git_dirty_code` 6156, `mem_limit_host_bytes` 6156, `time` 5996, `max_rss` 5995, `wall_sec` 6028, `dev_driver_alloc` 65
- mbp-macos26 vs mbp-macos27(共通の 6156 点): `run_id` 6156, `fingerprint` 6156, `host_tag` 6156, `ts` 6156, `os_release` 6156, `macos` 6156, `macos_build` 6156, `time` 5996, `max_rss` 5978, `wall_sec` 5868, `dev_driver_alloc` 92, `stderr_tail` 160

## 使わなかった run、対応のない系列

- 使わなかった run: なし
- 対応のない系列: なし

## まとめ

- **studio vs mbp-macos26**: 対応する系列 16、共通の点 6156、全項目が同じ 6156、違う 0、片方にしかない点 0 + 0、manifest が違う系列 0
- **studio vs mbp-macos27**: 対応する系列 16、共通の点 6156、全項目が同じ 6156、違う 0、片方にしかない点 0 + 0、manifest が違う系列 0
- **mbp-macos26 vs mbp-macos27**: 対応する系列 16、共通の点 6156、全項目が同じ 6156、違う 0、片方にしかない点 0 + 0、manifest が違う系列 0

