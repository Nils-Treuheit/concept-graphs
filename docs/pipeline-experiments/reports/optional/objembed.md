# ObjEmbed object-region embedding benchmark

The official ObjEmbed 2B and 4B object-token paths were evaluated on the same frozen five-scene ReplicaSSG split (250 frames, 2,204 object views, 49 categories). Both runs completed in BF16 on CUDA:0.

| Model | Identity top-1 | Identity R@5 | Oracle triplet recall | Category-text mAP | Wall time | Peak allocated memory |
|---|---:|---:|---:|---:|---:|---:|
| ObjEmbed 2B | 0.38047 | 0.71948 | 0.142857 | 0.02554 | 106.706 s | 4.716 GiB |
| ObjEmbed 4B | 0.37817 | 0.73422 | 0.183673 | 0.02537 | 122.549 s | 9.417 GiB |

Results are mixed; neither replaces the selected encoder on this nonidentical crop protocol. These are oracle-box object-token measurements, not detector proposal recall or autonomous map association. See the [2B report](../../../../scripts/pipeline_model_reports/objembed_2b_test5.md), [4B report](../../../../scripts/pipeline_model_reports/objembed_4b_test5.md), and [aggregate comparison](../../../../scripts/pipeline_model_reports/objembed.md) for exact configurations, artifacts, and limitations.
