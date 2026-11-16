# Open-scene YOLOE-26 prompt modes: accuracy and complete-pipeline latency

**Status:** proposed experiment; no open-scene inference, VLM prompt generation, model acquisition, or export has run. Existing COCO component results are reported separately and are not open-scene pipeline results.

## Decision this experiment should support

When the scene categories are not known in advance, does prompt-free YOLOE-26-L segmentation or a scene-description VLM followed by text-prompted YOLOE-26-L produce a more useful map than the current fixed ScanNet200 labels? What is the wall-clock cost of each complete path, including mask handling, features, annotations, mapping, and scene consolidation?

Treat this as a three-arm paired comparison. Keep the current fixed ScanNet200 approach as the operational baseline. Keep ONNX and TensorRT as a later backend study; they do not answer the vocabulary question and should not be mixed into this first comparison.

## Arms

| Arm | Detector and prompt behavior | What is fixed or learned before each scene? |
|---|---|---|
| A. Fixed-label baseline | Current `yoloe-26l-seg.pt`, text prompts via `set_classes`, using all 200 `conceptgraph/scannet200_classes.txt` names. | Existing ScanNet list and MobileCLIP2 text encoder. |
| B. Prompt-free | Official `yoloe-26l-seg-pf.pt`; do not call `set_classes`. | The checkpoint's built-in prompt-free vocabulary. |
| C. VLM scene vocabulary | Lightweight image VLM describes calibration frames and returns a bounded list of object/actor category nouns. Pass this list to `yoloe-26l-seg.pt` via `set_classes` and reuse it for the full scene. | Five evenly spaced calibration frames per scene; one frozen list per scene. |

Use one VLM to return the validated category schema directly; do not add a second LLM by default. Screen the candidate set below on the same 25 calibration frames (five per scene), then select one candidate using schema validity, category precision/recall, generation latency, and actual co-resident memory. Keep the 225 held-out frames for final evaluation only.

| Candidate | Why screen it | Constraint to verify |
|---|---|---|
| SmolVLM2-500M-Video-Instruct | Smallest candidate; its model card reports 1.8 GB GPU RAM for video inference and supports image questions/captioning. | That memory figure is not a guarantee for this prompt, runtime, or co-resident pipeline. |
| Qwen3.5-0.8B and 2B | Compact vision-language variants and a reasonable structured-output candidate. | Disable thinking for the bounded extraction task; the 0.8B card warns that thinking mode may loop. Confirm exact serving/JSON-constraint support. |
| Florence-2-base; try Florence-2-large if base misses quality | `<OD>` returns structured boxes and object labels. `<DENSE_REGION_CAPTION>` also returns boxes and region labels, but its descriptive phrases may need a deterministic noun normalizer. This could provide scene categories without a second LLM. Base is 0.23B; Large is 0.77B. | It predates the newest candidates; its model-card benchmarks do not establish current open-scene category quality or our complete-pipeline latency. Measure `<OD>` and dense captions on the same calibration frames. |
| Gemma 4 E2B | Designed for on-device multimodal use and supports images. | E2B means effective parameters: its card lists 5.1B total parameters with embeddings. Check actual weight and runtime memory. |
| LFM3-VL-2B | Existing local BF16 Transformers benchmark makes it a relevant baseline; it handles images and video. | Official GPU checkpoint is BF16, not FP16. Existing quality data is general image-task evidence only. Measure category extraction and co-residency directly. |
| Moondream 3.1 | Native caption, query, detection, and point outputs with structured responses. | It has 9B total parameters and 2B active; memory depends on stored weights and runtime, not active count alone. Review its model-specific license. |

Treat all published memory numbers as screening hints, not acceptance. Measure actual peak GPU use with YOLOE and the fixed feature/map models resident, including runtime workspaces, graph pools, image tokens, and output buffers. A candidate that cannot remain resident or misses the latency gate is excluded. Use the smallest candidate that clears the category-quality and response-time thresholds. The existing [LFM3 BF16 report](../vlm_lfm3_vl_2b_transformers_bf16.md) measures general image tasks and does not validate scene-category extraction.

The VLM is asked for visible object and actor *categories*, such as “desk”, “plant”, or “person”. It returns a strict JSON object with separate `objects` and `actors` arrays containing category strings and optional confidence values. A deterministic validator parses the schema, removes duplicates, normalizes spelling and plural forms, rejects actions/relationships/attributes/prose, and caps the final list at 50 unique categories using a frozen salience rule. Do not first generate a free-form paragraph and then rely on fragile noun-extraction heuristics. Allow names outside ScanNet200 because discovering out-of-taxonomy categories is the test. Preserve the raw response and validated list. Do not use GT labels, test labels, or hidden frames to compose the prompt list.

If the selected VLM cannot reliably return a valid category schema, add one small LLM to convert its scene description into that same JSON schema. In that fallback, the VLM, extractor LLM, and YOLOE weights must all fit in GPU memory together and remain resident for the experiment. Run their inference steps sequentially on the GPU; record memory use and stage latency. The LLM must not see GT data. Do not accept an extraction fallback that requires unloading or swapping model weights between frames or scenes.

For prompt-free and VLM outputs, save the raw names and apply a separately reviewed, versioned category crosswalk after inference. Preserve exact matches, accepted aliases, ambiguous matches, and unmatched names as distinct statuses. Do not silently force a novel name to the nearest ScanNet class.

Visual prompts are out of scope. They require example boxes and generally return generic object IDs, so they do not solve open-world category naming without another labeling mechanism.

## Dataset and split

Use the frozen ReplicaSSG five-scene local-complete test subset in `models/experiments/objembed/data/replicassg-five-scene-v1.json`: office_1, office_3, office_4, room_1, and room_2. It has 50 sampled frames per scene (250 total), 2,204 visible GT object views, 215 tracks, and 49 category strings. Six of the original eleven test scenes lack local RGB/pose inputs and remain excluded.

For arm C, choose five evenly spaced frames from each scene as vocabulary calibration inputs. Generate and freeze one list per scene. Run all 50 sampled frames through the full pipeline and include calibration, prompt setup, and model switching in scene latency. For the primary paired accuracy comparison, score only the other 45 frames per scene (225 frames). Report the five calibration frames separately as an explicitly seen-frame diagnostic; do not include them in held-out quality aggregates. Arms A and B use the same 225 primary scoring frames and also process the same 250-frame sequence for the operational latency comparison.

A preliminary exact-string audit found 19/49 GT category strings absent from the ScanNet200 list, corresponding to 43/215 tracks and 279/2,204 object views. This is only a string mismatch audit. Some labels may be valid aliases or synonyms. Before scoring, freeze a crosswalk and report results for (1) exact ScanNet matches, (2) reviewed alias matches, and (3) genuinely unmatched or unscorable labels. The main score must not count unresolved labels as detector misses. Provide both a common scorable subset comparison and a separate coverage/discovery analysis for labels outside the baseline vocabulary.

This is a controlled indoor-scene benchmark, not evidence of general open-world performance. Five scenes are a limited sample, and the calibration design allows each VLM arm to inspect scene context before scoring later frames.

## Hold the rest of the pipeline constant

Change only detector prompt mode and the scene-level VLM prepass in arm C. Fix the image order and decode path, YOLOE size L, 640-pixel input, confidence/IoU thresholds, maximum detections, mask resizing/retina setting, and downstream settings. Record config and checkpoint hashes.

Keep the selected masked feature encoder (SigLIP2 So400m), padded proposal encoder (FG-CLIP2 So400m), per-frame annotation model and prompt, mapper parameters, and caption consolidation model/config fixed across arms. Use separate caches and map outputs per arm. The dataset currently does not have a ready-to-use map-level ground-truth evaluator; validate this before interpreting map or relation scores as accuracy. Do not tune thresholds separately per arm on the test scenes.

For arm C, require the scene VLM, optional extractor LLM, and YOLOE to fit in GPU memory together. Load the required weights once at process start and keep them resident; execute each inference stage sequentially so only one model is actively generating at a time. Include loading all weights in cold-start time, and report warm scene time separately. If co-residency fails, arm C does not meet the requested deployment design; do not quietly unload and reload models to make its throughput look acceptable. Complete arms sequentially, with no other GPU job sharing the device.

## Accuracy and coverage measures

Report the detector metrics first, paired by scene and frame:

- GT category coverage before inference: fraction of distinct GT categories expressible by each arm's available/generated prompt vocabulary, plus counts of object views and tracks covered.
- Instance box and mask precision/recall and AP at IoU 0.50 and 0.50:0.95, using a frozen evaluator and the same 225 held-out frames.
- Per-scene and per-category results, false positives, duplicate detections, invalid/empty masks, and class assignment accuracy.
- Exact-match, accepted-alias, and unmatched/ambiguous-label strata. Include ScanNet200-supported classes and out-of-list classes separately.
- For C, generated category count, prompt-list precision/recall against scene GT categories, calibration-frame list stability across repeated deterministic runs, and held-out category coverage. Do not treat a VLM description as ground truth.

For the complete pipeline, report map object/track fragmentation and merges, category consistency, and relation triplet precision/recall/F1 only if an evaluator can validly align predictions with ReplicaSSG IDs and predicates. Otherwise provide the map outputs as qualitative artifacts and label these measures unavailable. Detector AP alone cannot establish map quality.

Use paired per-scene deltas and confidence intervals clustered by scene (five scene clusters). Also report the raw per-scene values; do not imply frame-level independence or broad statistical generalization.

## Response time and throughput

For every arm, report two distinct speed views:

1. **Detector-stage throughput:** full-resolution segmentation path at batch one, including image decode/load, preprocessing, model, mask generation and upsampling, device-to-host transfer, CPU extraction, and output serialization. Give cold-start, warm per-frame latency (median and p95), and steady-state images/s. Also profile internal stages to locate the cost.
2. **Complete-pipeline effective throughput:** `250 sampled frames / elapsed wall-clock seconds` from scene start until all scene outputs are written. Include checkpoint/model loads and switches, VLM calibration and prompt creation for C, YOLOE prompt setup, all detector and mask work, image/region features, frame annotations, map updates, and final scene consolidation. Report each scene separately, total five-scene elapsed time, time to first mapped result, median and p95 frame service time where meaningful, and peak GPU memory.

A secondary amortized figure may exclude one-time checkpoint loading but must still include per-scene VLM calibration and prompt setup. State exactly which operations are included in each number. Do not compare the old 9.64 seg images/s and 79.49 detect-only images/s as if they were model-kernel measurements: the detect-only arm omits masks, and the historical windows also include different CPU output work. Rerun arm A with the same paired frames and timing boundary as B and C.

For operational response time, include prompt generation in arm C even though five calibration frames are not in the held-out accuracy aggregate. Report detector-only and complete-pipeline measurements together so a fast detector cannot hide a slow scene prepass or downstream stage. The prompt-free arm is the speed reference. The initial real-time guardrail is at least 10 effective end-to-end frames/s after scene-level calibration, with arm C no more than 10% slower than prompt-free on warm full-scene wall time and p95 frame service time. Report a failure if prompt-free itself misses 10 frames/s; do not relabel detector-only FPS as pipeline FPS.

## Execution gates and decision rule

1. **Asset and evaluator preflight:** obtain the official prompt-free checkpoint through the approved asset workflow, verify its source, checksum, license/usage terms, and local loading route. Confirm the exact segmentation variant, not a detect-only checkpoint. Check whether the existing ReplicaSSG manifest can drive all three detectors and whether the pipeline can consume each arm's class outputs without an arm-specific downstream change. Build and freeze the GT category crosswalk and evaluator before model scoring.
2. **VLM screen and residency:** screen SmolVLM2-500M, Qwen3.5-0.8B/2B, Florence-2-base, Gemma 4 E2B, LFM3-VL-2B, and Moondream 3.1 using the same 25 calibration frames and a fixed category-list prompt/schema. Select one candidate before opening the held-out results. For LFM3, use the documented full-weight BF16 GPU checkpoint unless a separate FP16 conversion is verified. Before selection, verify the candidate plus YOLOE and fixed feature/map components can remain resident together. Record model revision, dtype/quantization, runtime, schema-valid rate, category precision/recall, latency, and peak GPU memory. Use one VLM only; add a separate extractor LLM only if the direct structured-output candidates fail and the added model also meets every residency and latency gate. If no candidate qualifies, mark arm C as not deployable under the requested design.
3. **Paired run:** execute all arms on the same scenes, frames, hardware, thresholds, downstream model versions, and output paths isolated by arm. Run one GPU model at a time. Keep any runtime failures and incomplete scenes visible in the report; mark partial outcomes preliminary.
4. **Interpretation and go/no-go:** compare quality and total scene latency together. For the VLM arm, require either (a) at least +2 absolute mask-AP points or +5 percentage points of GT object-view recall on the out-of-ScanNet subset versus prompt-free, with no more than a 1-point mask-AP or common-label recall loss, or (b) at least 10% faster full-pipeline response than prompt-free with quality non-inferior within 1 point. In either case, require co-residency and the latency guardrail above. These are proposed thresholds to freeze before inference, not results. If no arm meets the guardrail, report measured tradeoffs and mark the real-time objective unmet; do not claim a winner from detector-stage FPS alone.

The prompt-free checkpoint is not currently listed in `scripts/download_models.py`, and no local `yoloe-26l-seg-pf.pt` asset was found during plan preparation. Arm B therefore has an explicit asset gate and has not been run. The selected VLM candidate and any optional extractor LLM require a GPU co-residency preflight with YOLOE and downstream pipeline models before arm C is eligible. This plan does not authorize model download or GPU execution.

## Separate follow-up: ONNX and TensorRT

After choosing a prompt mode, profile whether segmentation postprocessing or model execution dominates. Then compare PyTorch FP32/FP16, ONNX Runtime CUDA FP16, and TensorRT FP16 on the same fixed prompt profile and images, with identical full-resolution masks and downstream output. YOLOE export can bake prompt embeddings into the artifact, so an exported engine cannot be assumed to accept scene-varying VLM lists. Adaptive prompt use in arm C should remain on the supported PyTorch route unless a documented runtime path for dynamic prompts is verified. Include engine build/startup and prompt-embedding preparation in any operational comparison.

## Current evidence and documentation

The current modernized masked pipeline uses `yoloe-26l-seg.pt` with `set_classes` and the 200-name ScanNet list. The completed COCO comparison uses 80 COCO text prompts and is a detector component benchmark, not an open-scene or end-to-end result. No arm in this experiment has been benchmarked yet. The VLM route now screens small candidates including Florence-2; LFM3-VL-2B is documented for full-weight BF16 GPU inference, and its existing benchmark covers general image tasks only.

- [YOLOE prompting modes](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md#choosing-a-prompting-mode): text, visual, and prompt-free checkpoint behavior.
- [YOLOE export usage](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md#export-usage): prompt embeddings and exported models.
- [YOLOE limitations](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md#limitations): prompt-set size and class-name guidance.
- [Ultralytics ONNX integration](https://docs.ultralytics.com/integrations/onnx/).
- [Ultralytics TensorRT integration](https://docs.ultralytics.com/integrations/tensorrt/).

## Research candidate: DINOv3-L feature injection into YOLOE-26-L-seg

**Status:** architecture hypothesis only; no hybrid has been implemented or benchmarked. Keep this behind the open-scene prompt-mode benchmark and an available, disjoint training split.

The DINO-YOLO preprint reports gains from injecting frozen DINOv3 features into a YOLOv12 detector on several small, domain-specific detection datasets. It also reports materially higher inference latency for the hybrid configurations. This makes feature injection worth a scoped study, but does not establish a benefit for YOLOE-26, open-vocabulary detection, or instance masks. YOLOE has prompt-conditioned detection and segmentation paths that a plain backbone swap could disrupt.

### Hypothesis and first implementation

Test whether frozen DINOv3 ViT-L/16 patch features improve YOLOE-26-L-seg's open-scene box and mask quality when fused into its multi-scale neck. Preserve YOLOE's native backbone, text and prompt-free vocabulary paths, detection head, and mask head. Add trainable feature projections and gated residual fusion at stride-matched neck levels; begin with one level, then test an additional level only if the first arm trains stably. Do not start with a full backbone replacement or copy the paper's P0 image-projection path without a YOLOE-specific ablation.

Use three arms: (A) unmodified YOLOE-26-L-seg evaluated with the selected prompt mode; (B) the same YOLOE initialization and training schedule with neck/segmentation components adapted but no DINO features, to control for fine-tuning; and (C) the same schedule with frozen DINOv3-L features and trainable projections/fusion. Keep prompt lists, image resolution, thresholds, optimizer budget, training examples, and evaluation code paired. If a text-alignment component is later tested, make DINOtxt a separate factor; do not conflate its text encoder or global/local losses with the DINO-YOLO feature-injection hypothesis.

### Feasibility gate and evaluation

Before training, confirm the exact YOLOE-L feature taps and strides, DINOv3-L/16 weight provenance and license, checkpoint compatibility, GPU memory headroom, and a legal training set disjoint by scene from the frozen ReplicaSSG five-scene evaluation subset. If no suitable disjoint instance-mask training data exists, stop at a paper-backed proposal; do not tune on the held-out scenes. Freeze prompt-free and text-prompt evaluation protocols before training.

Evaluate box AP, mask AP, per-category recall, and the existing exact/alias/unmatched open-scene strata on the same held-out frames. Report prompt-free and text-prompt modes separately, since retaining YOLOE's interface does not guarantee retaining its pretrained vocabulary quality after adaptation. Measure cold load, warm batch-one full segmentation latency, end-to-end pipeline throughput, and peak GPU memory with DINOv3 resident. Include training cost and checkpoint size. Compare against both A and B so gains cannot be attributed only to extra adaptation.

Proceed beyond feasibility only if the fusion arm improves a predeclared open-scene quality measure over both controls, while maintaining the existing end-to-end real-time guardrail and required GPU co-residency. Otherwise retain vanilla YOLOE. The DINO-YOLO paper's reported accuracy gains and latency are external results, not expected outcomes for this experiment.

- [DINO-YOLO preprint](https://arxiv.org/abs/2510.25140): frozen DINOv3 feature injection into YOLOv12 for domain-specific detection; reported accuracy and latency vary by configuration.
- [DINOv3 repository and model documentation](https://github.com/facebookresearch/dinov3): backbone variants, feature outputs, weights, and license information.
- [YOLOE prompting modes and architecture](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md): text, visual, and prompt-free paths that the hybrid must preserve and re-evaluate.

## Follow-up: scoped RF-DETR-Seg M/L comparison

RF-DETR is worth benchmarking as a closed-set segmentation control, but it is not a direct substitute for the open-scene arms above. The documented COCO-pretrained RF-DETR checkpoints classify a fixed COCO label set; the inference API resolves COCO class IDs or class names stored in a fine-tuned checkpoint. The documented YOLOE text-prompt path can change categories at inference. Therefore, RF-DETR cannot currently test the same unknown-class discovery question without fine-tuning on a suitable, disjoint training set.

Use `RFDETRSegMedium` and `RFDETRSegLarge` on the exact 500 COCO image IDs already used for YOLOE-26-M/L-seg. Compare each against YOLOE-26-M/L-seg with the same COCO category mapping, frozen scorer, confidence sweep/AP calculation, and full-resolution instance masks. Keep batch size one and run on the same GPU and host. Record checkpoint load/capture warmup separately; time image input, preprocessing, forward pass, mask upsampling, transfer, CPU extraction, and serialization. Report mask AP and box AP with paired per-image differences, warm latency distribution, end-to-end detector images/s, peak GPU memory, and failures. Do not compare Roboflow's published T4 TensorRT timings directly to the local 5090 measurements: they use different hardware/backend, and the documentation cautions that segmentation CUDA Graph performance is not yet measured.

Only promote RF-DETR as a possible fast path for scenes where the known COCO classes are adequate. It does not replace the prompt-free or VLM-prompted YOLOE experiment for unknown categories. If the project later wants a fair custom-class RF-DETR arm, first establish a legal, disjoint training set with instance masks and freeze its taxonomy; fine-tuned performance is a separate question from pretrained inference.

- [RF-DETR detection checkpoints](https://rfdetr.roboflow.com/latest/learn/run/detection/#pre-trained-checkpoints): M/L COCO accuracy and T4 TensorRT latency.
- [RF-DETR segmentation checkpoints](https://rfdetr.roboflow.com/latest/learn/run/segmentation/#pre-trained-checkpoints): M/L mask models, COCO scores, and published latency conditions.
- [RF-DETR segmentation inference](https://rfdetr.roboflow.com/latest/learn/run/segmentation/): fixed COCO class mapping, mask output behavior, and CUDA Graph caveats.


## Candidate references

- [SmolVLM2-500M-Video-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct): image/video use and reported GPU memory.
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B): multimodal model details and thinking-mode caveat.
- [Gemma 4 E2B model card](https://ai.google.dev/gemma/docs/core/model_card_4): supported modalities and effective-versus-total parameters.
- [LFM3-VL-2B model card](https://huggingface.co/LiquidAI/LFM3-VL-2B): checkpoint precision, inputs, and serving routes.
- [Moondream 3.1 model card](https://huggingface.co/moondream/moondream3.1-9B-A2B): structured skills, total/active parameter counts, and model license.
- [Florence-2-base model card](https://huggingface.co/microsoft/Florence-2-base): 0.23B checkpoint, object detection and dense-region-caption tasks with structured boxes/labels, and benchmark references. The Large variant is 0.77B.
