# mps-silent-failures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22845374.svg)](https://doi.org/10.5281/zenodo.22845374)

Code and results for the technical report

> Junichiro Niimi. *Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in PyTorch's Apple MPS Backend.* 2026. [PDF](paper.pdf) (arXiv link: to be added)

On macOS 15 and later, PyTorch's MPS backend returns **wrong values without an error or a warning** for batched matrix multiplication on tensors with more than 2**32 elements. This affects `torch.bmm`, and therefore `torch.matmul` and eager attention. We observed it in every PyTorch release we tested, from 2.4.1 to 2.14.0. Upstream issue: [pytorch/pytorch#197636](https://github.com/pytorch/pytorch/issues/197636).

The study is black-box. This repository records what the backend returns compared with reference results; it does not analyze the implementation.

## Minimal reproduction

Needs an Apple Silicon Mac with macOS 15 or later and about 40 GB of free unified memory.

```sh
uv run --no-project --python 3.12 --with torch==2.14.0 python firstpass/issue_repro.py
```

The output of `bmm` crosses 2**32 elements at `B = 65536`. Just below, the relative error against the CPU is at rounding level; just above, it is of order one, and nothing is raised.

## A guard you can use today

No PyTorch version we tested is safe above 2**32 elements on MPS, so the practical rule is: *do not create or read an MPS tensor with 2**32 or more elements; split the batch into chunks instead.* `firstpass/mps_guard.py` enforces the rule. It is a single file with no dependencies other than PyTorch.

```python
import mps_guard
mps_guard.install()   # from here on, any MPS op touching >= 2**32 elements raises Over2Pow32Error
```

It is a `TorchDispatchMode` that checks the inputs and outputs of every aten op. For `mm`, `bmm`, `addmm`, `baddbmm` and fused SDPA it estimates the size before running the op, because some PyTorch versions crash inside the op. The threshold is inclusive because PyTorch 2.4.1 is silently wrong for a view with exactly 2**32 elements; below 2**32 elements, no run of the sweep in any version is silently wrong. For `bmm` the error message states how many batches per chunk stay below the limit. Overhead measured with `firstpass/bench_guard.py`: 1.73x on a small training loop (2000 steps, 1024 x 120 inputs) and 1.00x on large attention forward passes. Set `MPS_GUARD=0` to disable `install()`.

## Contents

| Path | What it is |
|---|---|
| `scan.py`, `worker.py` | Boundary sweep of `bmm`. `scan.py` is the driver (calibration, points around each theoretical boundary, bisection, index-encoded inputs, full comparison at the final points). `worker.py` executes one configuration in a fresh process with a pinned PyTorch version, on MPS or CUDA. |
| `run_versions.sh`, `versions.py` | Reduced sweep for each PyTorch version, and the version-by-condition table. |
| `demo_workload.py` | Case study on a real workload: a RoBERTa sentiment classifier with eager attention on TweetEval, one large batch against chunked execution. |
| `boundary_map.py` | Draws the boundary map figure of the report from the raw results of one run. |
| `check_idx_invariance.py` | CPU-only check that the index-encoded runs which the rules predict to be wrong, but which are correct, use inputs that are unchanged by the predicted misreading. |
| `workload_stats.py` | Recomputes the case-study statistics from the saved logits, without rerunning on MPS (separation of errors around the boundary position, accuracy and exact McNemar test on the affected positions, predicted-class breakdown). |
| `firstpass/` | First-pass test suite over 26 operations and 11 PyTorch versions (`check_mps.py`, `run_matrix.sh`), the guard, its benchmark, and the minimal reproduction. |
| `colab_chunks.py` | Runs the sweep on a Colab GPU in chunks of (shape, dtype, layout), one run per chunk, because a Colab session does not last for the whole sweep. |
| `cuda_ref.py` | CUDA control runs on an A100 for the first-pass configurations (output in `results/cuda/a100_run1.log`). |
| `results/summary/` | Per-run summaries and the version table. |
| `results/raw/` | One JSON line per executed configuration, and a manifest per run (environment, versions, script hashes, tolerances). |
| `results/workload/` | Outputs of the case study (logits, first-layer intermediates, figure, abort log at L = 256). |
| `results/summary/a100/`, `results/raw/a100/` | CUDA control: the same sweep on an A100 80 GB with PyTorch 2.14.0 (18 chunks, 1464 runs, all correct; overview in `cuda_control.md`). These runs finished after the first version of the report and are not described in it. |
| `results/cuda/` | CUDA control output and a note on `torch.arange` on CUDA. |
| `provenance/` | Earlier revisions of `scan.py` whose hashes appear in the manifests. |

## Running the sweep

```sh
uv run python scan.py --host-tag myhost --dry-run    # number of configurations, memory and time estimate
uv run python scan.py --host-tag myhost --no-wandb   # sweep under a new run_id
uv run python scan.py --host-tag myhost --no-wandb --resume <run_id>
uv run python scan.py --host-tag myhost --summary <run_id>
./run_versions.sh 2.13.0 2.14.0                      # reduced sweep per PyTorch version
cd firstpass && ./run_matrix.sh                      # first-pass suite, all 11 versions
```

PyTorch is not a project dependency. Each run starts `worker.py` through `uv run --no-project --with torch==X`, so every version gets a fresh environment. Configurations whose memory estimate exceeds 60% of physical memory are recorded as `skipped_memory`.

How a result is judged: the reference is float64 on the CPU, computed from inputs that are regenerated on the CPU from fixed seeds (inputs are never read back from the device to build the reference). The error is `max|device - fp64| / max|fp64|` per batch. The tolerance per dtype is ten times the largest error measured in a calibration run well below the boundary. Outcomes are `ok`, `error` (raises), `crash` (process aborts), `timeout`, `wrong`, `truncated`, `input_corrupt`, `skipped_memory` and `skipped_budget`.

## Environment of the published results

Mac Studio (Mac14,14), Apple M2 Ultra, 192 GB unified memory, macOS 27.0 (build 26A428), Python 3.12.12, PyTorch 2.4.1 to 2.14.0 from the official PyPI wheels. CUDA control runs come from an A100 on Google Colab with TF32 disabled.

## Notes

- **Provenance.** Manifests and result rows record a git revision. It refers to the private development repository, not to this one. To check that the published scripts are the ones that produced a result, compare the `sha256` of `scan.py` and `worker.py` recorded in the manifest (`code.sha_scan`, `code.sha_worker`) with the files here. `scan.py` gained options between runs, so earlier runs record a different hash; those earlier revisions are kept in `provenance/` as `scan.<first 12 hex digits of the sha256>.py`. Files are copied unchanged; in result files the home directory in absolute paths is replaced by `/Users/USER`.
- **Language.** Comments and docstrings in the code are in Japanese, and so are the per-run summaries. They are left as they are, because editing the scripts would change the recorded hashes.
- **Status.** This is the state of the first version of the report: one machine and one macOS version. Other macOS versions are planned for the next version.

## Citation

Please cite the report. To refer to the code and the results themselves, use the Zenodo record, [doi:10.5281/zenodo.22845374](https://doi.org/10.5281/zenodo.22845374), which always resolves to the latest version.

```bibtex
@misc{niimi2026mps,
  author = {Niimi, Junichiro},
  title  = {Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in {PyTorch}'s {Apple} {MPS} Backend},
  year   = {2026},
  note   = {arXiv identifier to be added}
}
```

## License

The code, the guard and the result files are released under the MIT License (see `LICENSE`). The report (`paper.pdf`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
