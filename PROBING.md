# ImageNet-100 visual-feature linear probes

## Objective

Compare frozen visual representations from the pinned Base model and all four
normal unfrozen paper-recipe checkpoints: Rotation, Position, Jigsaw, and
Contrastive. The probe measures two feature locations without running the
language model:

- `pre_merger`: final ViT patch features immediately before `visual.merger`;
- `post_merger`: projected visual tokens immediately before they enter the LLM.

All spatial tokens are mean pooled. Only a linear classifier is trained.

## Dataset identity

The selected dataset is the existing ImageFolder tree at
`/data2/ykli/imagenet100`, not `PKU-ML/ImageNet-Subset`:

```text
train: 100 classes, 130,000 images, 1,300/class
val:   100 classes,   5,000 images,    50/class
Labels.json SHA256:
  0adf3838574e9f51ae33ab9097a78898cb05017c62df29b18109b6525e11641b
sorted synset list SHA256:
  86bc5c0b92179267729b4b1efc27c5cf14a321fdc3a689b3cc217a5c492a8a6b
```

The extraction command refuses any other class or row inventory.

## Protocol

- models: Base, normal Rotation, Position, Jigsaw, and Contrastive;
- image transform: bicubic short-side resize to 256, center crop to 224;
- frozen model, no language-model forward pass;
- feature dimensions for Qwen2.5-VL-3B: 1,280 pre-merger, 2,048 post-merger;
- L2-normalized cached features;
- linear head: SGD, momentum 0.9, no weight decay, cosine schedule;
- 90 epochs and seeds 0, 1, 2;
- primary metric: ImageNet-100 validation Top-1 accuracy.

## Smoke

Use a unique temporary directory. This loads the real Base visual stack but
extracts only 32 train and 16 validation images across four classes:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/probing/imagenet100_linear_probe.py smoke \
  --model-path <pinned-base-model> --model-role base \
  --data-root /data2/ykli/imagenet100 \
  --output-dir /tmp/imagenet100-probe-smoke-<id> \
  --device cuda:0 --batch-size 4 --workers 0 \
  --classes 4 --train-per-class 8 --val-per-class 4 --epochs 30
```

The smoke is a wiring check only; its accuracy is not reportable.

## Full extraction and fitting

Run these commands separately for Base, Rotation, Position, Jigsaw, and
Contrastive. Every normal trained model must have a matching
`ssl4rl_export_manifest.json` whose `model_role` matches the CLI role.

```bash
python scripts/probing/imagenet100_linear_probe.py extract \
  --model-path <model> --model-role <base|rotation|position|jigsaw|contrastive> \
  --data-root <imagenet100> --output-dir <feature-cache> \
  --device cuda:0 --batch-size 32 --workers 8 \
  --attn-implementation flash_attention_2

python scripts/probing/imagenet100_linear_probe.py fit \
  --cache-dir <feature-cache> --output-dir <probe-results> \
  --device cuda:0 --epochs 90 --batch-size 4096 \
  --learning-rate 0.1 --seeds 0,1,2
```

Each model cache is approximately 0.9 GB in float16. Store full caches under
the target job's durable output root, not the nearly full `/data1` or `/data2`
filesystems on server 247.

## Aggregation

```bash
python scripts/probing/aggregate_imagenet100_probes.py \
  --base-result <base-linear_probe_results.json> \
  --rotation-result <rotation-linear_probe_results.json> \
  --position-result <position-linear_probe_results.json> \
  --jigsaw-result <jigsaw-linear_probe_results.json> \
  --contrastive-result <contrastive-linear_probe_results.json> \
  --output-dir <summary-dir>
```

Aggregation rejects smoke subsets, task-identity mismatches, changed artifacts,
and differences in dataset, feature extraction, or head-training protocols.
