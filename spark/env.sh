set -ex
cd ~/kev
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
uv venv -p 3.13 .venv-spark
. .venv-spark/bin/activate
uv pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu129
uv pip install -e ".[serve]"
uv pip install "flash-linear-attention==0.5.2" "triton>=3.7.1"
python -c "import torch,triton;print(torch.__version__,torch.version.cuda,triton.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability(),torch.cuda.get_arch_list())"
