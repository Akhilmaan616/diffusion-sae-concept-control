# Loaded via PYTHONPATH=tests/pretend_gpu (see tests/run_all.sh).
# Pretend a GPU is visible (as on Kaggle). This machine has no CUDA, so any
# tensor actually sent to "cuda" raises -- exposing device leaks on CPU.
# Queries a real GPU answers trivially are stubbed with the real GPU's answer.
import torch
torch.cuda.is_available = lambda: True
torch.cuda.is_current_stream_capturing = lambda: False
# Adam's graph-capture check is a no-op on a real GPU when no capture is active.
torch.optim.Optimizer._accelerator_graph_capture_health_check = lambda self: None
