# Independent adjudication: DINOtxt tempfile repairs

**Candidate:** `scripts/pipeline_model_reports/benchmarks/dinotxt_replica_test5.py`  
**Root evaluation reviewed:** `root-evaluation-writer-v4.md`  
**Scope:** Read-only source review. No imports, tests, benchmark, CUDA, or model execution.

## Verified claims

- `write_json_no_overwrite` calls `tempfile.mkstemp` with supported `prefix` and `dir` arguments. Its unique same-directory temporary file is written as bytes, flushed and fsync'd, closed, then hard-linked to the destination. `os.link` preserves no-overwrite semantics. The `finally` block closes an unclaimed descriptor and removes the temporary path.
- `save_embedding_cache` writes NPZ data to the binary file object returned by `os.fdopen`, flushes and fsyncs before `os.replace`, and closes/removes temporary files in cleanup blocks. Metadata uses a separate same-directory `mkstemp` file and the same flush/fsync/replace and cleanup pattern.
- `_require_cuda` resolves bare `cuda` through `torch.cuda.current_device()`, rejects non-CUDA devices, negative indices, and indices greater than or equal to `device_count()` before model construction. The default is indexed `cuda:0`.
- The corresponding source review statements in the root evaluation are supported by the candidate.

## Blocking issue

The runner's stated invariant that all outputs and caches stay under `/tmp` is not enforced. `resolve_output_path` rejects repository paths and paths inside protected inputs, but permits an otherwise arbitrary `--output` path. `--embedding-cache` is also unrestricted; its parent directory is passed to `mkstemp` and the cache is replaced there. Defaults are under `/tmp`, but explicit arguments can write elsewhere. This makes the module-level guarantee inaccurate and leaves the output/cache location policy unenforced.

**Adjudication: do not pass yet.** Either enforce `/tmp` containment for the evidence output and embedding cache, or revise the stated contract and evaluation criteria if arbitrary external destinations are intended. No other blocking defect was found in the reviewed tempfile or CUDA-index repairs.
