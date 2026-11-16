# Modernized component experiments

The modernized pipeline has two matched modes. `masked` is the default and
uses YOLOE-26-L-seg with SigLIP2 Base mask-isolated image features.
`region-proposal` uses standard YOLOE-26-L with FG-CLIP2 So400m padded-proposal
features and rectangular box geometry for RGB-D mapping. Both use the
configured local VLM annotation baseline. Keep each component comparison
paired and change one setting per run. Never mix feature vectors from different
encoders in a map.

Final caption consolidation defaults to Gemma 4 12B QAT Q4; frame annotation
continues to use the configured VLM (Qwen3.8 by default). Gemma starts after
mapping and stops after consolidation. Pass `--consolidation-model qwen3.8` to
retain the earlier Ollama-backed consolidator.

## Run a paired component comparison

Prepare local checkpoints first:

```bash
./.venv/bin/python scripts/download_models.py --download --only yoloe26l yoloe26l_detect
./.venv/bin/python scripts/download_models.py --download --only modernized_features
```

The YOLOE text-prompt tokenizer is a declared project dependency. Run
`uv sync` before offline inference if the readiness check reports it missing;
the modernized detector disables Ultralytics' implicit package installation.

Use the paired launcher with the same scene, stride, data config, prompts,
runtime, and decoding settings for both arms. It creates separate detection,
annotation, and mapping suffixes for the baseline and candidate.

```bash
scripts/run_modernized_component_benchmark.sh \
  --component detector \
  --baseline /path/to/yoloe-26l-seg.pt \
  --candidate /path/to/yoloe-26m-seg.pt \
  --scene room0 --stride 50 --no-consolidation
```

Supported slots are `detector`, `mask-encoder`, `proposal-encoder`, and `vlm`.
Mask-encoder comparisons use `masked`; proposal-encoder comparisons default to
`region-proposal`. Pass `--input-mode` only when an explicit different mode is
needed.
Encoder values are Hugging Face model IDs. Detector values are local checkpoint
paths. VLM values name models already served by the configured local endpoint.
Do not change two slots in the same comparison. The YAML config snapshots and
separate output suffixes record which component produced each arm.
For interrupted VLM annotation, pass `--resume-from-caches` to the local
pipeline launcher. It verifies a complete detection cache, skips detection,
then resumes annotation before mapping.

## Score each component separately

### Detector

Prepare both official segmentation checkpoints, then run the deterministic
500-image COCO val subset benchmark on CUDA:

```bash
./.venv/bin/python scripts/download_models.py --download --only yoloe26l yoloe26m
./.venv/bin/python -m conceptgraph.scripts.yoloe_component_benchmark \
  --models yoloe-26l yoloe-26m --max-images 500 \
  --output .agentic_workflow/workspace/modernized_yoloe_seg_component.json
```

It reports instance-mask AP@0.5, box AP@0.5, latency, and peak CUDA allocation.
Mask AP uses native-resolution predicted polygons and COCO polygon annotations.
For map-level follow-up, use identical VLM and feature encoders and compare
object counts, association quality, and graph recovery. Keep association
thresholds fixed.

### Mask and proposal encoders

Run the existing real-data object-retrieval benchmark one model at a time:

```bash
./.venv/bin/python -m conceptgraph.scripts.clip_component --models siglip2-so400m
./.venv/bin/python -m conceptgraph.scripts.clip_component --models fg-clip2-so400m
./.venv/bin/python -m conceptgraph.scripts.clip_component --models clip
```

Use the same annotated crop bank and report masked-object and padded-proposal
retrieval metrics separately, including macro AP, Recall@K, category accuracy,
cross-view identity, throughput, and peak memory. The modernized pipeline
selects the mask and proposal encoder independently. For map-level follow-up,
run one selected encoder in its matching slot and keep the other slot fixed.
SigLIP2 and FG-CLIP2 features are separate spaces and must not be combined.

### VLM annotation

Use the same held-out frame manifest, YOLOE caches, object IDs, masks, prompt,
output schema, and decoding settings for each served model. Score useful visible
attributes, unsupported claims, object-ID coverage, parse success, and physical
relation precision/recall. Review disagreements against reference annotations.
Record model revision or quantization, runtime, reasoning mode, prompt version,
latency, and peak GPU memory. Keep the current Qwen3.8 local model as baseline;
LLaVA-OneVision-2 and Qwen2.5-VL are candidates, not assumed winners.

### Caption consolidation model pilot

Caption consolidation is a text-only stage and is evaluated separately from
per-frame VLM annotation. First verify the configured Qwen model on the four
room0 object groups that previously returned malformed output. Save those
successful outputs as the baseline. Then compare the same source captions with
smaller local edge-compatible models. This is a failure-focused pilot, not a
dataset coverage benchmark.

When the GPU is available, release the local model in the workflow checkout
and confirm model-control has no active managed or unmanaged GPU jobs:

```bash
cd /home/nils/Project_Code/codex_opencode_pipeline
python3 src/executor.py release-local
```

Caption consolidation disables Qwen reasoning so the 256-token response
budget is used for the caption rather than a hidden reasoning trace. Verify the
configured Qwen model on the four real failed groups:

```bash
LLM_MODEL=qwen3.8:27b-mtp4-ud_q4_K_XL-128k \
CONCEPTGRAPHS_CAPTION_TEST_MAP=/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/pcd_r_modernized_yoloe26l_stride50.pkl.gz \
CONCEPTGRAPHS_CAPTION_TEST_OUTPUT=/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/qwen_caption_baseline_failed_groups.json \
  ./.venv/bin/python -m pytest -q tests/test_modernized_caption_consolidation.py
```

Proceed only if all four Qwen outputs are nonempty. For each smaller candidate,
start its CUDA server in a separate terminal. Set the model directory to the
local `hub/local_models` directory inspected for this experiment:

```bash
CG_EDGE_MODEL_DIR=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub/local_models/edge-compatible-gguf \
LLAMA_SERVER_BIN=/path/to/llama-server \
  scripts/edge-compatible/serve_model.sh qwen35-9b-q4
```

The edge server launcher also disables reasoning by default. Set
`EDGE_COMPATIBLE_REASONING=on` or `auto` only when a task needs it.

For a broader scene comparison, use the deterministic 40-track sample in
`.agentic_workflow/workspace/caption-consolidation-benchmark/room0-40-case-sample.json`.
It includes every multi-caption track in the map and 35 evenly spaced
single-caption tracks. Single-caption cases measure output behavior but do not
test multi-caption merging. Its paired Qwen3.8 baseline combines cached outputs
from the 40-frame run with the four repaired results.

With the server running, use the repository `.venv` to compare only text
consolidation against the saved Qwen outputs. The benchmark derives case
indices from the baseline, then checks the exact caption inputs and source-map
hash before sending requests:

```bash
./.venv/bin/python scripts/benchmark_caption_consolidation.py \
  --map /mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/pcd_r_modernized_yoloe26l_stride50.pkl.gz \
  --baseline-results .agentic_workflow/workspace/caption-consolidation-benchmark/room0-qwen3.8-baseline-40.json \
  --candidate-id qwen35-9b-q4
```

The benchmark records `server_reasoning_mode=off`; pass a matching
`--server-reasoning-mode` value if the server was started with another mode.

Stop the candidate server before repeating with `qwen35-9b-q8` and
`gemma4-12b`. The launcher derives and verifies a unique server alias from the
model ID; the benchmark requires that exact alias and rejects a mismatch. The
three edge candidates have about 6.9 GB, 13.9 GB, and 6.9 GB of weights plus
projector, respectively. The report saves each output, latency, agreement with
Qwen, case counts by caption multiplicity, and fields for manual faithfulness,
unsupported-claim, and concision review. Text overlap with Qwen is not
ground-truth accuracy.

### Query-time VLM

Query answering is separate from per-frame annotation. Use a fixed set of
retrieved templates, evidence crops, object attributes, relations, questions,
and answer schema. Compare answer correctness, object attribution, evidence
support, affordance reasoning, abstention, latency, and peak memory. The
configured Qwen3-VL 30B Thinking model is an experiment candidate only; the
annotation launcher never loads it per frame.

## Results record

Store each run's config snapshot, source manifest, checkpoint or model revision,
metrics, latency and peak-memory measurements, and a brief error analysis
beside its experiment outputs. Compare paired scene/query results and confidence
intervals where the held-out sample supports them. Leave unmeasured fields
empty; do not infer model quality from checkpoint names or published scores.
