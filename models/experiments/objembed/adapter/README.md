# ObjEmbed ReplicaSSG adapter

`prepare_replica_manifest.py` freezes the local-complete ReplicaSSG test subset and oracle instance crops. `benchmark_gt_object_embeddings.py` evaluates the official ObjEmbed object-token encoder using those same oracle boxes. It does not evaluate WeDetect proposal recall or detector-conditioned AP.

Run `python3 src/executor.py release-local` from `.agentic_workflow`, verify the model-control app reports no unrelated model/job, and launch from host context. Run one checkpoint at a time:

```bash
models/experiments/envs/objembed/bin/python \
  models/experiments/objembed/adapter/benchmark_gt_object_embeddings.py \
  --models 2b --attention sdpa
```

The environment must use the pinned ObjEmbed repository at `models/experiments/objembed/repo` and cached HF snapshot IDs documented in `scripts/pipeline_model_reports/objembed.md`. The runner sets HF offline mode, requires complete model shards, validates every frozen input file against its manifest hash, and refuses existing output directories. It reads `use_global_caption`, `use_two_tokens`, and `use_two_captions` from the loaded checkpoint and supplies the matching `<global>` and `<global_text>` token IDs/placeholders used by official inference. It has no CPU fallback.

The 2B and 4B runs use the same five ReplicaSSG test scenes, 50 evenly sampled positions per scene, 2,204 ground-truth object views, 49 category queries, and full view gallery. Scores include category-text macro AP, category-query Recall@K, zero-shot category accuracy, leave-frame-out repeated-ID association, and oracle-predicate graph identity recovery. The report must keep these scores separate from the existing SigLIP2/FG-CLIP2 crop-control scores and the WeDetect proposal-conditioned benchmark. Oracle boxes isolate the encoder and do not measure detector or map association quality.

The runner reports the HF revision and shard hashes, frozen manifest digest, exact model dtype and attention backend, elapsed model/text/image encoding times, GPU UUID/name, resident/peak allocated and reserved VRAM, and per-model raw metrics JSON. Each run uses a fresh output directory; failed artifacts remain for diagnosis.
