# mps-silent-failures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22845374.svg)](https://doi.org/10.5281/zenodo.22845374)

Code and results for the paper

<!-- TODO(arxiv-id): replace "to be added" with the arXiv link once the identifier is assigned -->
> Junichiro Niimi. *Silent Failures Beyond the 32-Bit Index Range: A Differential Characterization of Large-Tensor Matrix Multiplication in PyTorch's MPS Backend.* 2026. [PDF](paper.pdf) (arXiv link: to be added)

PyTorch's MPS backend returns **wrong values without an error or a warning** for batched matrix multiplication on tensors with more than 2**32 elements. This affects `torch.bmm`, and therefore `torch.matmul` and eager attention. We observed it in every PyTorch release we tested, from 2.4.1 to 2.14.0. Upstream issue: [pytorch/pytorch#197636](https://github.com/pytorch/pytorch/issues/197636).

What the paper (`paper.pdf`, the second arXiv version) finds:

- **Three rules account for every outcome on 2.14.0.** When the output exceeds 2**32 elements and an operand is a transposed view, the entire output is wrong and equals a computation that ignores that operand's strides. Otherwise, a view with at least 2**31 elements raises an exception, and a contiguous input above 2**32 elements makes exactly the batches beyond that point wrong, equal to a computation whose index wraps at 2**32. A slightly larger problem can thus turn an explicit error into a silent failure. The rules reproduce all 1376 outcomes with random inputs, including the series in which inputs and output exceed 2**32 elements together and the points near 2 * 2**32 elements.
- **Training is affected where inference is not.** With contiguous operands and an output just above 2**32 elements, the forward pass is correct and both gradients are silently wrong. The same rules, applied to the two products that define the gradients, reproduce all 592 outcomes of the backward sweep.
- **Deterministic across machines and macOS versions.** An M2 Max under macOS 26.6.2, the same machine under macOS 27.0 and an M2 Ultra under macOS 27.0 agree in all 6156 common runs of every series, for correct and wrong results alike and for all 11 PyTorch versions. The same sweeps on an NVIDIA A100, including the backward pass, are correct in all 2530 runs.
- **Real workloads are affected.** In a public sentiment classifier with eager attention, one oversized batch corrupts a third of the outputs, which collapse onto one class.
- **One rule is enough in practice.** Do not create or read an MPS tensor with 2**32 or more elements. Applied to the recorded outcomes of all 11 versions, this rule stops all 481 silent runs, and no run below 2**32 elements is silently wrong.

All findings are obtained from observable behavior: the kernels below PyTorch's MPS backend are closed source, so this repository records what the backend returns compared with reference results and does not analyze the implementation.

## Minimal reproduction

Needs an Apple Silicon Mac with macOS 15 or later and about 40 GB of free unified memory.

```sh
uv run --no-project --python 3.12 --with torch==2.14.0 python v2/firstpass/issue_repro.py
```

The output of `bmm` crosses 2**32 elements at `B = 65536`. Just below, the relative error against the CPU is at rounding level; just above, it is of order one, and nothing is raised. `v2/firstpass/issue_repro_backward.py` is the corresponding reproduction for the backward pass: with contiguous operands and `B = 65537`, the forward output is correct and the last batch of both gradients is wrong.

## A guard you can use today

No PyTorch version we tested is safe above 2**32 elements on MPS, so the practical rule is: *do not create or read an MPS tensor with 2**32 or more elements; split the batch into chunks before it reaches the device.* `guard/mps_guard.py` enforces the rule. It is a single file with no dependencies other than PyTorch; copy it next to your code.

```python
import mps_guard
mps_guard.install()   # from here on, any MPS op touching >= 2**32 elements raises Over2Pow32Error
```

It is a `TorchDispatchMode` that checks the inputs and outputs of every aten op. For `mm`, `bmm`, `addmm`, `baddbmm` and fused SDPA it estimates the size before running the op, because some PyTorch versions crash inside the op. The threshold is inclusive because PyTorch 2.4.1 returns zeros for a view with exactly 2**32 elements; below 2**32 elements, no run of the sweep in any version is silently wrong. For `bmm` the error message states how many batches per chunk stay below the limit. The rule also prevents the silently wrong gradients, because the forward output, which has the size of the upstream gradient, is stopped first. Overhead measured with `guard/bench_guard.py`: 1.73x on a small training loop (2000 steps, 1024 x 120 inputs) and 1.00x on large attention forward passes. Set `MPS_GUARD=0` to disable `install()`.

## Contents

| Path | What it is |
|---|---|
| `v1/` | Code and results as of the first arXiv version (a technical report): one machine, one macOS version. Frozen: these are the files that were at the top level of this repository up to v1.0.x. The first-pass results (`v1/firstpass/results/`), the case-study outputs (`v1/results/workload/`) and the CUDA control on an A100 (`v1/results/raw/a100/`, `v1/results/cuda/`) are here. |
| `v2/` | The current harness, the results reported in the paper, and the analysis scripts. See the next table. |
| `guard/` | `mps_guard.py`, the guard described above, and `bench_guard.py`, its overhead benchmark. |
| `paper.pdf` | The paper (second arXiv version). |
| `pyproject.toml`, `uv.lock` | The `uv` project used by `uv run python ...`. PyTorch is not a dependency (see below). |

### `v2/`

| Path | What it is |
|---|---|
| `scan.py` | Driver of the boundary sweep of `bmm`: calibration, points around each theoretical boundary, bisection, index-encoded inputs, full comparison at the final points. New in v2, all off by default: the backward pass (`--op bmm_bwd`), a shape in which both inputs and the output exceed 2**32 elements together (`--shapes all-256x256x256`), points near multiples of 2**32 elements (`--far`), small views at large storage offsets (`--layouts offset`), bfloat16, and a list of given points (`--points`). With default arguments the plan is the same as in `v1/`. |
| `worker.py` | Executes and judges one configuration (`bmm` or `bmm_bwd`; device, shape, dtype, layout, kind of input, batch size, seed, kind of comparison) in a fresh process with a pinned PyTorch version, on MPS or CUDA. |
| `compare_rows.py` | Compares the raw results of several hosts row by row, over all series and all fields except those that describe the host or vary between executions. Exits with 1 on any difference. Reads only; no GPU. |
| `compare_hosts.py` | Host-by-condition table of the classes for the `bmm` sweeps (machines, macOS versions, CUDA control). Reads only; no GPU. |
| `versions.py` | Version-by-condition table from the sweeps of each PyTorch version. |
| `run_versions.sh` | Reduced sweep for each of the ten earlier PyTorch versions, one run per version. |
| `boundary_map.py` | Draws the boundary map figure of the paper from the raw results of one run. |
| `check_idx_invariance.py` | CPU-only check that the index-encoded runs which the rules predict to be wrong, but which are correct, use inputs that are unchanged by the predicted misreading. |
| `colab_chunks.py` | Runs the sweep on a Colab GPU in chunks, one run per chunk, because a Colab session does not last for the whole sweep. In v2 it also covers the backward, (256, 256, 256), offset and far-point series. |
| `cuda_ref.py` | CUDA control runs on an A100 for the first-pass configurations (unchanged from `v1/`). |
| `demo_workload.py` | Case study on a real workload: a RoBERTa sentiment classifier with eager attention on TweetEval, one large batch against chunked execution (unchanged from `v1/`). |
| `workload_stats.py` | Recomputes the case-study statistics from the saved logits, without rerunning on MPS. |
| `analysis/` | Scripts that check the rules against the recorded outcomes (`rules.py`, `rules_bwd.py`, `analyze.py`; output in `rules_all.txt`), the point lists passed to `scan.py --points` (far points, supplement of the (256, 256, 256) shape), and the queues that were actually run. See `analysis/README.md`. |
| `cloud/` | Helper scripts for running the sweep on a remote Mac over SSH (setup, sweep over macOS versions, pulling results). |
| `firstpass/` | First-pass test suite over 26 test cases (`check_mps.py`, `run_matrix.sh`), the minimal reproductions (`issue_repro.py` for the forward pass, `issue_repro_backward.py` for the gradients), and `sdpa_ops.py`, which records the aten op that fused attention resolves to on MPS. The published first-pass results are in `v1/firstpass/results/`. |
| `results/raw/<host-tag>/` | One JSON line per executed configuration (`<run_id>.jsonl`), and a manifest per run (`<run_id>.manifest.json`: environment, versions, script hashes, tolerances, command line). |
| `results/summary/` | Per-run summaries (`<host-tag>/<run_id>.md`), the overview of the v2 series (`studio/v2_axes_studio.md`), and the comparisons between hosts (`hosts_*.md`). `hosts_rows_studio_mbp26_mbp27.md` is the row-by-row comparison of the three configurations. |

### Hosts in `v2/results/raw/`

| host-tag | Machine | What it contains |
|---|---|---|
| `studio` | Mac Studio, M2 Ultra, macOS 27.0 | 16 runs, 6348 rows |
| `mbp-macos26` | MacBook Pro, M2 Max, macOS 26.6.2 | the same 16 runs |
| `mbp-macos27` | the same MacBook Pro after the upgrade to macOS 27.0 | the same 16 runs |
| `a100` | Google Colab, NVIDIA A100-SXM4-80GB, PyTorch 2.14.0+cu130 (CUDA control) | the same series as `studio` on PyTorch 2.14.0 (no reduced sweep), split into 37 runs because a hosted session is short: main sweep 18, backward 8, (256, 256, 256) 4, its repeated points 2, far points 4, offset 1; 2530 rows, all correct |

The 16 runs of each host, all with the same `scan.py` and `worker.py`:

| Series | Runs | PyTorch | Rows per run | Compared between hosts |
|---|---|---|---|---|
| main sweep | 1 | 2.14.0 | 1596 | 1572 |
| reduced sweep | 10 | 2.4.1, 2.5.1, 2.6.0, 2.7.1, 2.8.0, 2.9.1, 2.10.0, 2.11.0, 2.12.1, 2.13.0 | 360 (374 for 2.5.1 to 2.8.0) | 348 (362) |
| backward pass | 1 | 2.14.0 | 604 | 592 |
| (256, 256, 256) shape | 1 | 2.14.0 | 228 | 222 |
| supplement of the (256, 256, 256) shape | 1 | 2.14.0 | 36 | 30 |
| far points | 1 | 2.14.0 | 124 | 112 |
| offset | 1 | 2.14.0 | 104 | 92 |

Rows per run include the calibration rows; the comparison between hosts excludes them, which leaves 6156 common runs. Only runs whose manifest has `status: done` are published.

## Running the sweep

Run from `v2/`. `--host-tag` names the directory under `results/raw/` and `results/summary/`; use a tag of your own.

```sh
cd v2
uv run python scan.py --host-tag myhost --dry-run    # number of configurations, memory and time estimate
uv run python scan.py --host-tag myhost --budget-hours 24   # main sweep under a new run_id
uv run python scan.py --host-tag myhost --resume <run_id>
uv run python scan.py --host-tag myhost --summary <run_id>
uv run python scan.py --host-tag myhost --budget-hours 24 --op bmm_bwd --shapes out-256x64x256,in-256x256x64   # backward pass
uv run python scan.py --host-tag myhost --budget-hours 24 --shapes all-256x256x256                              # (256, 256, 256) shape
uv run python scan.py --host-tag myhost --budget-hours 24 --layouts offset --shapes out-256x64x256,in-256x256x64 # offset
uv run python scan.py --host-tag myhost --points analysis/far_points_full.json      # far points
uv run python scan.py --host-tag myhost --points analysis/all256_supp_points.json   # supplement of the (256, 256, 256) shape
./run_versions.sh 2.13.0                             # reduced sweep per PyTorch version (writes under host-tag "studio")
cd firstpass && ./run_matrix.sh                      # first-pass suite, all 11 versions
```

The command line of every published run is recorded in its manifest (`invocations[].argv`). PyTorch is not a project dependency. Each run starts `worker.py` through `uv run --no-project --with torch==X`, so every version gets a fresh environment. Configurations whose memory estimate exceeds 60% of physical memory are recorded as `skipped_memory`; the largest estimate in the published series is 53.4 GiB, and no configuration was skipped on the 96 GB machine. For time, the manifests record `elapsed_min`: on the M2 Ultra the main sweep took 114 minutes, a reduced sweep 28 to 38 minutes per version, the backward sweep 80 minutes, and all 16 runs 9.7 hours (9.8 and 9.9 hours on the M2 Max).

How a result is judged: the reference is float64 on the CPU, computed from inputs that are regenerated on the CPU from fixed seeds (inputs are never read back from the device to build the reference). The error is `max|device - fp64| / max|fp64|` per batch. The tolerance per dtype is ten times the largest error measured in a calibration run well below the boundary. Outcomes are `ok`, `error` (raises), `crash` (process aborts), `timeout`, `wrong`, `truncated`, `input_corrupt`, `skipped_memory` and `skipped_budget`. For the backward pass, the output and the two gradients are judged separately (`cls_out`, `cls_grad_a`, `cls_grad_b`), and the class of the run is the worst of the three.

## Checking the published results

**Comparison between machines.** This reads the published raw results only; it needs no GPU and no PyTorch.

```sh
cd v2
uv run python compare_rows.py --source results:studio --source results:mbp-macos26 --source results:mbp-macos27 --all-pairs
echo $?   # 0
```

It exits with 0 and reports, for each of the three pairs, 16 matching series and 6156 common runs, all 6156 identical in every compared field, and none present on one side only. The report is written to `results/summary/hosts_rows.md` (change with `--out`) and should be identical to the published `results/summary/hosts_rows_studio_mbp26_mbp27.md`. The fields that are excluded from the comparison (identifiers, the description of the host, measured time and memory, the stderr of a crash) are listed at the top of the report. A difference in any other field, a run present on one side only, or a manifest that differs in the script hashes, the tolerances, the PyTorch version or the point list makes the script exit with 1.

**Script hashes.** Every manifest records the `sha256` of the scripts that produced it (`code.sha_scan`, `code.sha_worker`). All published manifests in `v2/results/raw/` record the same pair, and it matches the files here (the second command prints each hash with a count; there should be one line per script):

```sh
shasum -a 256 v2/scan.py v2/worker.py
grep -ohE '"sha_(scan|worker)": *"[0-9a-f]{64}"' v2/results/raw/*/*.manifest.json | sort | uniq -c
```

**Rules.** `cd v2 && uv run python analysis/rules.py B results/raw/studio/<run_id>.jsonl` checks the rules of the paper against the recorded classes of a run (see `v2/analysis/README.md`).

## Environment of the published results

Machine A: Mac Studio (Mac14,14), Apple M2 Ultra, 192 GB unified memory, macOS 27.0 (build 26A428), Python 3.12.12. Machine B: MacBook Pro (Mac14,6), Apple M2 Max, 96 GB unified memory, macOS 26.6.2 (build 25G83) and then macOS 27.0 (build 26A428), Python 3.12.11. PyTorch 2.4.1 to 2.14.0 from the official PyPI wheels. Machine C (CUDA control): Google Colab runtime, Intel Xeon @ 2.20 GHz (12 vCPUs), 167 GB host memory, NVIDIA A100-SXM4-80GB, driver 580.82.07, CUDA 13.0, Ubuntu 24.04.1 LTS, Python 3.12.3, PyTorch 2.14.0+cu130, TF32 disabled.

## Notes

- **Raw data.** Result files are published as they were written, with two replacements that remove personal information: the home directory in absolute paths is replaced by `/Users/USER`, and the URL of the private Weights & Biases project is removed (`wandb.url` is `null` in the manifests, and `(private)` in the per-run summaries). Neither touches a field that is compared or analyzed. Scripts are copied unchanged. Runs that did not finish are not published.
- **Provenance.** Manifests and result rows record a git revision. It refers to the private development repository, not to this one. To check that the published scripts are the ones that produced a result, compare the hashes as shown above. For `v1/`, `scan.py` gained options between runs, so earlier runs record a different hash; those earlier revisions are kept in `v1/provenance/` as `scan.<first 12 hex digits of the sha256>.py`.
- **Weights & Biases.** The published runs were backed up to a private W&B project with `--wandb`. This is off by default and is not needed to run the sweep or to check the results; everything is in `results/raw/`.
- **Language.** Comments and docstrings in the code are in Japanese, and so are the per-run summaries and comparison reports. They are left as they are, because editing the scripts would change the recorded hashes.
- **Status.** This is the state of the second arXiv version of the paper: two machines of the M2 generation, macOS 26.6.2 and 27.0. We did not run macOS 14 or 15, and other chip generations are untested.

## Citation

Please cite the paper. To refer to the code and the results themselves, use the Zenodo record, [doi:10.5281/zenodo.22845374](https://doi.org/10.5281/zenodo.22845374), which always resolves to the latest version.

<!-- TODO(arxiv-id): replace the note with eprint / archivePrefix fields once the identifier is assigned -->
```bibtex
@misc{niimi2026mps,
  author = {Niimi, Junichiro},
  title  = {Silent Failures Beyond the 32-Bit Index Range: A Differential Characterization of Large-Tensor Matrix Multiplication in {PyTorch}'s {MPS} Backend},
  year   = {2026},
  note   = {arXiv identifier to be added}
}
```

## License

The code, the guard and the result files are released under the MIT License (see `LICENSE`). The paper (`paper.pdf`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
