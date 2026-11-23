# ImageNet-100 visual-feature probe smoke on server 247

Date: 2026-11-23

Status: **PASS**

The bounded smoke used the existing pinned Base Qwen2.5-VL-3B-Instruct model,
the existing `/data2/ykli/imagenet100` ImageFolder tree, and one shared RTX PRO
5000 GPU selected for low utilization immediately before launch. All outputs
were written below a unique `/tmp` directory; no model, dataset, or feature
cache was copied onto the nearly full `/data1` or `/data2` filesystems.

Validated dataset identity:

- 100 train/validation synsets;
- 130,000 train images (1,300 per class);
- 5,000 validation images (50 per class);
- `Labels.json` SHA256
  `0adf3838574e9f51ae33ab9097a78898cb05017c62df29b18109b6525e11641b`;
- sorted synset-list SHA256
  `86bc5c0b92179267729b4b1efc27c5cf14a321fdc3a689b3cc217a5c492a8a6b`.

Smoke subset and feature checks:

- four deterministically selected classes;
- 32 train and 16 validation images;
- bicubic resize to short side 256 and center crop to 224;
- final-ViT/pre-merger features: 256 tokens per image, dimension 1,280;
- post-merger/pre-LLM features: 64 tokens per image, dimension 2,048;
- the complete model was frozen; only the two linear heads were optimized;
- per-split feature, label, and source-index arrays were written with SHA256
  records;
- both heads completed 30 epochs with finite, decreasing training loss;
- the final process released its GPU memory.

The four-class smoke accuracies are deliberately not reportable. Their only
purpose was to validate model loading, the merger pre-hook, visual-token
accounting, mean pooling, immutable feature caches, and linear-head training.

One compatibility issue was found and fixed before the successful run:
Transformers 4.55 exposes `model.visual` as a read-only property, so the script
must retain the visual-module reference and delete the top-level model object
rather than assigning `model.visual = None`.

Full Base/Rotation/Position extraction and three-seed fitting were not run on
247. They must use the target H20 job's durable output root as specified in
`PROBING.md`.
