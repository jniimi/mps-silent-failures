# mps-silent-failures

Code and results for the technical report

> Junichiro Niimi. *Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in PyTorch's Apple MPS Backend.* 2026. (arXiv link: to be added)

On macOS 15 and later, PyTorch's MPS backend returns **wrong values without an error or a warning** for batched matrix multiplication on tensors with more than 2**32 elements. This affects `torch.bmm`, and therefore `torch.matmul` and eager attention. We observed it in every PyTorch release we tested, from 2.4.1 to 2.14.0. Upstream issue: [pytorch/pytorch#197636](https://github.com/pytorch/pytorch/issues/197636).

The study is black-box. This repository records what the backend returns compared with reference results; it does not analyse the implementation.

## Minimal reproduction

Needs an Apple Silicon Mac with macOS 15 or later and about 40 GB of free unified memory.

```sh
uv run --no-project --python 3.12 --with torch==2.14.0 python firstpass/issue_repro.py
```

The output of `bmm` crosses 2**32 elements at `B = 65536`. Just below, the relative error against the CPU is at rounding level; just above, it is of order one, and nothing is raised.

## A guard you can use today

No PyTorch version we tested is safe above 2**32 elements on MPS, so the practical rule is: *do not create or read an MPS tensor with more than 2**32 elements; split the batch into chunks instead.* `firstpass/mps_guard.py` enforces the rule. It is a single file with no dependencies other than PyTorch.

```python
import mps_guard
mps_guard.install()   # from here on, any MPS op touching > 2**32 elements raises Over2Pow32Error
```

It is a `TorchDispatchMode` that checks the inputs and outputs of every aten op. For `mm`, `bmm`, `addmm`, `baddbmm` and fused SDPA it estimates the size before running the op, because some PyTorch versions crash inside the op. Overhead measured with `firstpass/bench_guard.py`: 1.73x on a small training loop (2000 steps, 1024 x 120 inputs) and 1.00x on large attention forwards. Set `MPS_GUARD=0` to disable `install()`.

## Contents

| Path | What it is |
|---|---|
| `scan.py`, `worker.py` | Boundary sweep of `bmm`. `scan.py` is the driver (calibration, points around each theoretical boundary, bisection, index-encoded inputs, full comparison at the final points). `worker.py` executes one configuration in a fresh process with a pinned PyTorch version, on MPS or CUDA. |
| `run_versions.sh`, `versions.py` | Reduced sweep for each PyTorch version, and the version-by-condition table. |
| `demo_workload.py` | Case study on a real workload: a RoBERTa sentiment classifier with eager attention on TweetEval, one large batch against chunked execution. |
| `firstpass/` | First-pass test suite over 26 operations and 11 PyTorch versions (`check_mps.py`, `run_matrix.sh`), the guard, its benchmark, and the minimal reproduction. |
| `cuda_ref.py` | CUDA reference values on an A100 for the first-pass configurations (output in `results/cuda/a100_run1.log`). |
| `wandb_log.py` | Optional logging of each run to Weights & Biases. Local JSONL is always written first. |
| `results/summary/` | Per-run summaries and the version table. |
| `results/raw/` | One JSON line per executed configuration, and a manifest per run (environment, versions, script hashes, tolerances). |
| `results/workload/` | Outputs of the case study (logits, first-layer intermediates, figure, abort log at L = 256). |
| `results/cuda/` | CUDA reference output and a note on `torch.arange` on CUDA. |
| `provenance/` | Earlier revisions of `scan.py` whose hashes appear in the manifests. |

## Running the sweep

```sh
uv run python scan.py --host-tag myhost --dry-run    # number of configurations, memory and time estimate
uv run python scan.py --host-tag myhost              # sweep under a new run_id
uv run python scan.py --host-tag myhost --resume <run_id>
uv run python scan.py --host-tag myhost --summary <run_id>
./run_versions.sh 2.13.0 2.14.0                      # reduced sweep per PyTorch version
cd firstpass && ./run_matrix.sh                      # first-pass suite, all 11 versions
```

PyTorch is not a project dependency. Each run starts `worker.py` through `uv run --no-project --with torch==X`, so every version gets a fresh environment. Configurations whose memory estimate exceeds 60% of physical memory are recorded as `skipped_memory`.

How a result is judged: the reference is float64 on the CPU, computed from inputs that are regenerated on the CPU from fixed seeds (inputs are never read back from the device to build the reference). The error is `max|device - fp64| / max|fp64|` per batch. The tolerance per dtype is ten times the largest error measured in a calibration run well below the boundary. Outcomes are `ok`, `error` (raises), `crash` (process aborts), `timeout`, `wrong`, `truncated`, `input_corrupt`, `skipped_memory` and `skipped_budget`.

## Environment of the published results

Mac Studio (Mac14,14), Apple M2 Ultra, 192 GB unified memory, macOS 27.0 (build 26A428), Python 3.12.12, PyTorch 2.4.1 to 2.14.0 from the official PyPI wheels. CUDA reference values come from an A100 on Google Colab with TF32 disabled.

## Notes

- **Provenance.** Manifests and result rows record a git revision. It refers to the private development repository, not to this one. To check that the published scripts are the ones that produced a result, compare the `sha256` of `scan.py` and `worker.py` recorded in the manifest (`code.sha_scan`, `code.sha_worker`) with the files here. `scan.py` gained options between runs, so earlier runs record a different hash; those earlier revisions are kept in `provenance/` as `scan.<first 12 hex digits of the sha256>.py`. Files are copied unchanged; in result files the home directory in absolute paths is replaced by `/Users/USER`.
- **Language.** Comments and docstrings in the code are in Japanese, and so are the per-run summaries. They are left as they are, because editing the scripts would change the recorded hashes.
- **Status.** This is the state of the first version of the report: one machine and one macOS version. Other macOS versions are planned for the next version.

## Citation

```bibtex
@misc{niimi2026mps,
  author = {Niimi, Junichiro},
  title  = {Silent Failures at the $2^{32}$ Boundary: A Technical Report on Large-Tensor Matrix Multiplication in {PyTorch}'s {Apple} {MPS} Backend},
  year   = {2026},
  note   = {arXiv identifier to be added}
}
```

## License

MIT. See `LICENSE`.
