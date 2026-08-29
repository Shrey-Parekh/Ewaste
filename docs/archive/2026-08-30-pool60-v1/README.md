# Archived results — first pool60 round (superseded)

These are the summaries of the three detectors trained on `dataset_pool60`
before the multi-architecture round. The run directories, weights and
evaluation images were deleted on 2026-08-30; only these files remain.

They are superseded because every model was retrained under a single uniform
schedule (common batch size, `cache='ram'`, identical epochs and seed) so that
the eight-way architecture comparison is fair. Numbers here are **not**
comparable with the current results and must not be quoted in the paper.

| Model | Detection rate | False-alarm rate | F1 |
|---|---|---|---|
| YOLOv8s | 78.5% | 7.0% | 0.820 |
| YOLOv11s | 84.2% | 10.0% | 0.830 |
| EDNet-S | 82.5% | 11.0% | 0.813 |

EDNet was VisDrone-pretrained while the YOLO arms were COCO-pretrained, so its
figures confound architecture with pretraining corpus.
