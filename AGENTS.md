# Repository instructions for coding agents

## Scope and objective

This checkout contains the Frozen Vision Encoder (FVE) GRPO workflow built on
SSL4RL and the bundled `verl 0.4.1.dev` source. The immediate deployment target
is one Linux job with 8 NVIDIA H20 GPUs (compute capability `sm_90`) and a
durable output root at:

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
```

Do not begin a full training run until the environment and all three FVE smoke
gates below pass. Do not change global batch semantics merely because the H20s
have more memory.

## Environment source of truth

Use a Python 3.10 image and prefer the following validated core stack:

| Package | Version |
|---|---:|
| Python | 3.10 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| vLLM | 0.10.2 |
| FlashAttention | 2.8.3.post2 |
| xFormers | 0.0.32.post1 |
| Triton | 3.4.0 |
| TensorDict | 0.9.1 |
| torchdata | 0.11.0 |
| Transformers | 4.55.2 |
| tokenizers | 0.21.1 |
| Accelerate | 1.8.1 |
| PEFT | 0.21.2 |
| Ray | 2.60.0 |
| datasets | 4.0.0 |
| PyArrow | 20.0.0 |
| qwen-vl-utils | 0.0.11 |
| NumPy | 1.26.4 |
| Hydra Core | 1.3.8 |
| OmegaConf | 2.3.1 |

This exact family was checked on 2026-11-15 in the existing SSL4RL environment
on `ryzhou@222.29.2.247`: `pip check` was clean; six CPU repository tests
passed; FVE, FSDP, reward-trace, and vLLM integration modules imported together;
and real CUDA plus FlashAttention kernels ran successfully. The PyTorch cu128
wheel reports both `sm_90` (H20) and `sm_120` support. This is strong package
compatibility evidence, but it is not a substitute for the required H20 smoke
run: no complete FVE optimizer step has yet succeeded on the target machine.

The host NVIDIA driver must support CUDA 12.8 containers. If it does not, stop
and report the driver and image constraints. Do not silently substitute an
older package family.

## Known dependency conflicts: do not mix these stacks

The generic upstream documentation and installer predate this FVE handoff:

- `README.md` describes `verl 0.3.0`, while this checkout bundles
  `verl 0.4.1.dev`.
- `scripts/install_vllm_sglang_mcore.sh` pins Torch 2.6.0, vLLM 0.8.5.post1,
  TensorDict 0.6.2, FlashAttention 2.7.4, and CUDA 12.4-era wheels.
- `setup.py` requires TensorDict 0.8.0 through 0.9.1, so the old install
  script's TensorDict 0.6.2 conflicts with the checked-in code.
- The optional `setup.py` extra named `vllm` caps vLLM at 0.8.5 and therefore
  does not describe the validated vLLM 0.10.2 stack.

Consequently:

1. Do **not** run `scripts/install_vllm_sglang_mcore.sh`.
2. Do **not** install `.[vllm]` or `.[sglang]`.
3. Install the validated Torch/vLLM/core versions first, then install this
   repository with plain `pip install -e .`.
4. Finish with `pip check` and the gates below. A successful import alone is
   insufficient evidence of CUDA compatibility.

## Suggested image-build order

Adapt package-index and wheel locations to the image service, but preserve the
versions and order unless a failed gate provides concrete evidence to change
them:

```bash
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm==0.10.2 xformers==0.0.32.post1 tensordict==0.9.1 torchdata==0.11.0
python -m pip install transformers==4.55.2 tokenizers==0.21.1 accelerate==1.8.1 peft==0.21.2 datasets==4.0.0 pyarrow==20.0.0 qwen-vl-utils==0.0.11 numpy==1.26.4 hydra-core==1.3.8 omegaconf==2.3.1 ray[default]==2.60.0 pytest
MAX_JOBS=8 python -m pip install flash-attn==2.8.3.post2 --no-build-isolation
python -m pip install -e .
python -m pip check
```

Do not add a separate FlashInfer pin unless vLLM reports that it is required.
The validated vLLM 0.10.2 environment had no standalone FlashInfer package;
the separately installed FlashAttention package passed a real CUDA kernel test.

## Mandatory environment gates

Run these checks inside the final image, with the selected environment first on
`PATH`. Preserve their output in the job log.

1. Confirm versions and H20 kernel coverage:

   ```bash
   python -c "import torch,vllm,tensordict,transformers,ray; print(torch.__version__,torch.version.cuda,torch.cuda.get_arch_list()); print(vllm.__version__,tensordict.__version__,transformers.__version__,ray.__version__); assert 'sm_90' in torch.cuda.get_arch_list()"
   ```

2. Run a real CUDA operation on every visible GPU, not just an import:

   ```bash
   python -c "import torch; assert torch.cuda.device_count()==8; print([(i,torch.cuda.get_device_name(i),(torch.ones(1,device=f'cuda:{i}')+1).item()) for i in range(8)])"
   ```

3. Run a real FlashAttention kernel:

   ```bash
   python -c "import torch; from flash_attn import flash_attn_func; q=torch.randn(1,4,2,64,device='cuda',dtype=torch.float16); o=flash_attn_func(q,q,q); torch.cuda.synchronize(); print(o.shape,torch.isfinite(o).all().item())"
   ```

4. Run the lightweight repository tests:

   ```bash
   python -m pytest -q tests/utils/test_reward_trace.py tests/special_sanity/test_import.py tests/utils/test_config_on_cpu.py
   ```

5. Confirm the Ray CLI belongs to the same environment. Use the executable,
   not `python -m ray` (Ray has no `ray.__main__`):

   ```bash
   "$(dirname "$(command -v python)")/ray" --version
   "$(dirname "$(command -v python)")/ray" stop --help >/dev/null
   ```

If any gate fails, record the complete command, versions, and error before
changing packages. Change one compatibility family at a time and repeat all
gates.

## FVE runtime invariants

- Base model: Qwen2.5-VL-3B-Instruct.
- Freeze the complete `visual.*` module, including the merger.
- Keep `actor.fsdp_config.use_orig_params=True`.
- Expected parameters with the real checkpoint: 668,684,288 frozen visual;
  3,085,938,688 trainable; 3,754,622,976 total.
- For 8 GPUs, retain 512 prompts, 5 rollouts per prompt, global rollout batch
  2560, global PPO mini-batch 640, per-GPU PPO mini-batch 80, and micro-batch 10.
- Use a short local Ray temp root such as `/tmp/fve-ray`; Ray UNIX sockets have
  a 107-byte path limit.
- Write durable checkpoints, TensorBoard events, reward traces, state markers,
  and queue logs below `FVE_OUTPUT_ROOT`.

## Required execution order on the target job

After model and processed data paths exist:

```bash
export FVE_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export FVE_DATA_ROOT=/path/to/our_datasets
export FVE_OUTPUT_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/jzzhang/runyv
export FVE_PYTHON="$(command -v python)"
python scripts/frozen_vision/prepare_smoke_data.py
```

Then run a dry-run and inspect the printed paths and batch values:

```bash
python scripts/frozen_vision/run_frozen_vision.py --task position --mode train --gpus 0,1,2,3,4,5,6,7 --run-id h20-config-check --dry-run
```

Finally, use a new explicit run ID and execute the queue. It must pass, in
order, `quick`, `quick-resume`, and `production-smoke` before any of the four
full training tasks starts:

```bash
python scripts/frozen_vision/run_queue.py --gpus 0,1,2,3,4,5,6,7 --run-id <unique-run-id>
```

Do not bypass a failed smoke gate. Inspect the corresponding log and
`state/<run-id>/queue.failed` first. Do not claim H20 readiness until the
production-shape smoke completes and the log contains the frozen-vision runtime
audit confirming no visual gradients and an unchanged visual checksum.

## Editing and validation rules

- Keep generated models, datasets, caches, logs, traces, and checkpoints out of
  Git.
- Do not restore old `/data2/ryzhou/...` paths or machine-specific proxies.
- Preserve dry-run purity: `--dry-run` must not create directories or files.
- Use `python3` or the selected environment's explicit Python, never assume a
  `python` alias outside the image.
- After changing FVE launch code, run `python -m py_compile` on all files under
  `scripts/frozen_vision`, run the lightweight tests above, and dry-run the
  8-GPU quick and train modes.
- Never start full training merely because imports pass. Real CUDA,
  FlashAttention, quick-resume, and production-shape tests are mandatory.
