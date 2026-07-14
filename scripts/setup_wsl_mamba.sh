#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Setting up Ubuntu packages..."
sudo apt-get update
sudo apt-get install -y build-essential git wget python3 python3-venv python3-dev

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Installing the CUDA 12.8 compiler toolkit inside WSL..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt
python -m pip install -e .

echo "Building the pinned causal-conv1d CUDA extension..."
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  "causal-conv1d==1.4.0"

echo "Building the pinned Mamba CUDA extension..."
MAMBA_FORCE_BUILD=TRUE python -m pip install \
  --no-cache-dir \
  --no-build-isolation \
  "mamba-ssm==2.2.6.post3"

python - <<'PY'
import torch
from mamba_ssm import Mamba

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Architectures:", torch.cuda.get_arch_list())

model = Mamba(d_model=128, d_state=16, d_conv=3, expand=2).cuda()
x = torch.randn(2, 197, 128, device="cuda")
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print("Fused Mamba verification passed:", tuple(y.shape))
PY

echo
echo "WSL Mamba environment is ready."
echo "Activate it with: source .venv-wsl/bin/activate"
echo "Resume with: python scripts/run_corrected_experiment.py --device cuda --skip-existing"
