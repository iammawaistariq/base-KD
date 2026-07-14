# Reproducible installation on a second PC

This project targets two matching Linux or WSL2 PCs with an NVIDIA GPU,
Python 3.10, PyTorch 2.11, and CUDA 12.8. Recreate the environment on each PC;
do not copy a Conda environment or virtual-environment directory. Those
directories contain absolute paths and native extensions built for one
machine.

## 1. Copy the portable files

Transfer the Git repository, required checkpoints under `runs/`, and any data
that cannot be downloaded again. Do not transfer `.venv`, Conda `envs/`,
`__pycache__`, `.so` files, `.triton_cache`, or `.torchinductor_cache`.

Use the same Git commit on both computers:

```bash
git rev-parse HEAD
```

## 2. Check the host GPU

Install the NVIDIA driver on the host, reboot, and then check it from Linux or
WSL2:

```bash
nvidia-smi
```

Do not install a Linux NVIDIA display driver inside WSL. WSL uses the Windows
host driver. The command must work before creating the Python environment.

## 3. Install the CUDA 12.8 compiler

`nvidia-smi` reports driver capability; it does not install `nvcc`. Mamba and
causal-conv1d need the CUDA compiler.

For WSL2 Ubuntu:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y build-essential git cuda-toolkit-12-8
```

For native Ubuntu, use NVIDIA's repository instructions for that exact Ubuntu
release, then install `cuda-toolkit-12-8`. Do not install the generic
`nvidia-cuda-toolkit` Ubuntu package because it may select a different version.

Configure the shell:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

`nvcc` must report CUDA 12.8.

## 4. Create a clean Conda environment

Run these commands from the repository root:

```bash
conda create -n vimkd-mamba python=3.10 -y
conda activate vimkd-mamba
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Confirm that the Python interpreter belongs to the new environment:

```bash
which python
python --version
python -m pip --version
```

Never use `sudo pip` and never mix packages from the Conda `base` environment.

## 5. Verify PyTorch before installing Mamba

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Architectures:", torch.cuda.get_arch_list())
PY
```

Expected results are PyTorch `2.11.0+cu128`, CUDA `12.8`, and `True`. RTX
50-series GPUs should also show `sm_120` in the architecture list. Stop here if
CUDA is unavailable; installing Mamba cannot repair a driver or PyTorch issue.

## 6. Build causal-conv1d and Mamba

Build each extension independently so a failure identifies the responsible
package:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export MAX_JOBS="$(nproc)"

CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install \
  causal-conv1d==1.4.0 \
  --no-build-isolation --no-cache-dir

MAMBA_FORCE_BUILD=TRUE python -m pip install \
  mamba-ssm==2.2.6.post3 \
  --no-build-isolation --no-cache-dir
```

`--no-build-isolation` is required: the build must see the already installed
CUDA-enabled PyTorch. A source build can take several minutes and may appear
quiet while compiling.

## 7. Run an actual CUDA verification

An import alone does not prove that the fused CUDA operation works:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba

assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
model = Mamba(d_model=128, d_state=16, d_conv=3, expand=2).cuda()
x = torch.randn(2, 197, 128, device="cuda")
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
print("Mamba CUDA verification passed:", tuple(y.shape))
PY

pytest -q
```

Run this verification on both PCs before starting or resuming training.

## Common Mamba failures

### `No module named torch` during the build

The packages were installed in the wrong order or pip used an isolated build.
Activate `vimkd-mamba`, install `requirements.txt` first, and retain
`--no-build-isolation` on both extension commands.

### `nvcc: command not found`

Install `cuda-toolkit-12-8`, set `CUDA_HOME`, and verify `nvcc --version`. The
CUDA libraries bundled with a PyTorch wheel do not include the compiler.

### `undefined symbol` when importing a `.so` file

The extension was built for another PyTorch/CUDA ABI. Reinstall it locally:

```bash
python -m pip uninstall -y mamba-ssm causal-conv1d
python -m pip cache purge
```

Then repeat sections 5 through 7. Do not copy wheels or `.so` files built on a
different environment.

### Compiler process is killed

The machine probably ran out of RAM. Retry the build with fewer parallel jobs:

```bash
export MAX_JOBS=2
```

### PyTorch CUDA and nvcc versions differ

These commands should both report 12.8:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

Fix `CUDA_HOME` and `PATH` before rebuilding both extensions.

## Record a successful installation

After verification, record the environment and machine identity for debugging:

```bash
mkdir -p reports/environment
python -m pip freeze > reports/environment/pip-freeze.txt
conda list --explicit > reports/environment/conda-explicit.txt
nvidia-smi > reports/environment/nvidia-smi.txt
nvcc --version > reports/environment/nvcc-version.txt
git rev-parse HEAD > reports/environment/git-commit.txt
```

`requirements.txt` remains the maintained installation input. The generated
reports are diagnostic snapshots and should only be reused on the same OS,
architecture, Python, CUDA, and GPU class.
