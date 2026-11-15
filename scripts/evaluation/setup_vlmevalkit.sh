#!/usr/bin/env bash
set -euo pipefail

VLMEVAL_COMMIT=0a11917eede83742389b03d7f11fc252aabdf6ef
EVAL_ROOT=${1:-"${FVE_OUTPUT_ROOT:?Set FVE_OUTPUT_ROOT}/tools/vlmevalkit"}
BASE_PYTHON=${FVE_PYTHON:-$(command -v python3)}
SOURCE_DIR="$EVAL_ROOT/VLMEvalKit-$VLMEVAL_COMMIT"
ENV_DIR="$EVAL_ROOT/env"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_PATCHER="$SCRIPT_DIR/patch_vlmeval_qwen.py"
MCQ_PROMPT_PATCHER="$SCRIPT_DIR/patch_vlmeval_mcq_prompt.py"

mkdir -p "$EVAL_ROOT"
if [[ ! -d "$SOURCE_DIR" ]]; then
  git clone https://github.com/open-compass/VLMEvalKit.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$VLMEVAL_COMMIT"
elif [[ -d "$SOURCE_DIR/.git" ]]; then
  actual_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD)
  [[ "$actual_commit" == "$VLMEVAL_COMMIT" ]] || {
    echo "Unexpected VLMEvalKit revision: $actual_commit" >&2
    exit 1
  }
elif [[ ! -f "$SOURCE_DIR/.pinned-commit" ]] || [[ $(<"$SOURCE_DIR/.pinned-commit") != "$VLMEVAL_COMMIT" ]]; then
  echo "Existing VLMEvalKit source is not pinned to $VLMEVAL_COMMIT" >&2
  exit 1
fi

if ! grep -q 'FVE patch: exported checkpoints' "$SOURCE_DIR/vlmeval/vlm/qwen2_vl/model.py"; then
  "$BASE_PYTHON" "$QWEN_PATCHER" "$SOURCE_DIR/vlmeval/vlm/qwen2_vl/model.py"
fi
"$BASE_PYTHON" "$MCQ_PROMPT_PATCHER" "$SOURCE_DIR/vlmeval/dataset/image_mcq.py"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install -r "$SOURCE_DIR/requirements.txt"
"$ENV_DIR/bin/python" -m pip uninstall -y opencv-python || true
"$ENV_DIR/bin/python" -m pip install \
  numpy==1.26.4 \
  transformers==4.55.2 \
  tokenizers==0.21.1 \
  qwen-vl-utils==0.0.11 \
  opencv-python==4.11.0.86 \
  gradio==5.49.1
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$SOURCE_DIR"
printf '%s\n' "$VLMEVAL_COMMIT" > "$SOURCE_DIR/.pinned-commit"

cd "$SOURCE_DIR"
"$ENV_DIR/bin/python" - <<'PY'
import numpy
import torch
import transformers
import vlmeval
from vlmeval.dataset import ImageMCQDataset
from vlmeval.vlm import Qwen2VLChat

assert numpy.__version__ == "1.26.4"
assert transformers.__version__ == "4.55.2"
print({"numpy": numpy.__version__, "torch": torch.__version__, "transformers": transformers.__version__})
print("VLMEvalKit Qwen2.5-VL and MMBench imports OK")
PY

echo "VLMEvalKit environment ready: $ENV_DIR/bin/python"
