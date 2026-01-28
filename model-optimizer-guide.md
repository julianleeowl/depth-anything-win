# NVIDIA Model Optimizer Guide for Vision Model Quantization

A practical guide for post-training quantization (PTQ) and quantization-aware training (QAT) on vision models like Depth-Anything V2.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Two Quantization Paths](#two-quantization-paths)
- [PyTorch vs ONNX Comparison](#pytorch-vs-onnx-comparison)
- [Quantization Formats](#quantization-formats)
- [Accuracy Improvement Techniques](#accuracy-improvement-techniques)
- [Sensitivity Analysis](#sensitivity-analysis)
- [PTQ to QAT Workflow](#ptq-to-qat-workflow)
- [Speed/Accuracy Tradeoff Analysis](#speedaccuracy-tradeoff-analysis)
- [Recommendations for ViT-based Models](#recommendations-for-vit-based-models)

---

## Overview

NVIDIA Model Optimizer provides:
- **PTQ**: Post-training quantization (no training data needed)
- **QAT**: Quantization-aware training (fine-tune with fake quantizers)
- **Formats**: INT8, FP8, INT4, NVFP4, W4A8, etc.
- **Deployment**: TensorRT, TensorRT-LLM, vLLM, ONNX Runtime

---

## Installation

```bash
# Basic installation
pip install nvidia-modelopt

# With ONNX support
pip install "nvidia-modelopt[onnx]"

# With HuggingFace support
pip install "nvidia-modelopt[hf]"

# Full installation (for dev)
pip install "nvidia-modelopt[all]"
```

Note: In zsh, quote the brackets: `pip install "nvidia-modelopt[onnx]"`

---

## Two Quantization Paths

### Path 1: PyTorch-based PTQ

```python
import torch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# Load model
model = MyModel().cuda().eval()

# Define calibration loop
def forward_loop(model):
    for img in calibration_images[:512]:
        with torch.no_grad():
            model(img.cuda())

# Quantize with SmoothQuant (best for transformers)
mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop)

# Save checkpoint
mto.save(model, "model_ptq.pt")

# Export to ONNX
torch.onnx.export(model, dummy_input, "model_quantized.onnx", opset_version=17)
```

### Path 2: ONNX-based PTQ

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=model.onnx \
    --quantize_mode=int8 \
    --calibration_method=entropy \
    --calibration_data=calib.npy \
    --output_path=model_int8.onnx
```

```python
from modelopt.onnx.quantization import quantize

quantize(
    onnx_path="model.onnx",
    quantize_mode="int8",
    calibration_data=calibration_numpy_array,
    calibration_method="entropy",  # or "max"
    nodes_to_exclude=[".*sensitive_layer.*"],
    output_path="model_int8.onnx"
)
```

---

## PyTorch vs ONNX Comparison

| Aspect | PyTorch Path | ONNX Path |
|--------|--------------|-----------|
| **SmoothQuant** | Yes | No |
| **AWQ variants** | Yes (lite, clip, full) | Yes (lite, clip) |
| **MSE calibration** | Yes | No |
| **AutoQuantize** | Yes | No |
| **Debug/iterate** | Easier | Harder |
| **QAT path** | Direct | Must go back to PyTorch |
| **TensorRT deploy** | Export to ONNX first | Direct |

**Recommendation**: Use **PyTorch path** for better accuracy control and QAT transition.

---

## Quantization Formats

### Available Configs (PyTorch)

```python
import modelopt.torch.quantization as mtq

# INT8
mtq.INT8_DEFAULT_CFG          # Basic INT8
mtq.INT8_SMOOTHQUANT_CFG      # SmoothQuant (best for transformers)
mtq.INT8_WEIGHT_ONLY_CFG      # Weight-only INT8

# FP8 (requires Ada/Hopper GPU)
mtq.FP8_DEFAULT_CFG           # Per-tensor FP8

# INT4
mtq.INT4_AWQ_CFG              # AWQ weight-only
mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG

# NVFP4 (requires Blackwell GPU)
mtq.NVFP4_DEFAULT_CFG
mtq.NVFP4_AWQ_LITE_CFG

# Mixed precision
mtq.W4A8_AWQ_BETA_CFG         # INT4 weights, FP8 activations
```

### Custom Config Example

```python
from copy import deepcopy

custom_cfg = deepcopy(mtq.INT8_SMOOTHQUANT_CFG)

# Disable specific layers
custom_cfg["quant_cfg"]["*patch_embed*"] = {"enable": False}
custom_cfg["quant_cfg"]["*norm*"] = {"enable": False}
custom_cfg["quant_cfg"]["*head*"] = {"enable": False}

# Use different precision for attention
custom_cfg["quant_cfg"]["*attn*"] = {"enable": False}  # Keep FP16
# Or use FP8 for attention:
# custom_cfg["quant_cfg"]["*attn*weight_quantizer"] = {"num_bits": (4, 3), "axis": None}

mtq.quantize(model, custom_cfg, forward_loop)
```

---

## Accuracy Improvement Techniques

### Without QAT

| Technique | How | Best For |
|-----------|-----|----------|
| **SmoothQuant** | `mtq.INT8_SMOOTHQUANT_CFG` | Transformers/ViT |
| **FP8 instead of INT8** | `mtq.FP8_DEFAULT_CFG` | Better accuracy, needs Ada/Hopper |
| **Entropy calibration** | `calibration_method="entropy"` | General |
| **More calibration data** | 500-1000 samples | All models |
| **Skip sensitive layers** | `{"enable": False}` for layer | High-sensitivity layers |
| **AWQ** | `mtq.INT4_AWQ_CFG` | Weight-only quantization |

### Common Sensitive Layers (ViT/Transformers)

```python
SKIP_PATTERNS = [
    "*patch_embed*",    # First embedding layer
    "*blocks.0.*",      # First transformer block
    "*blocks.1.*",      # Second transformer block
    "*norm*",           # All LayerNorms
    "*head*",           # Output head
    "*decoder*",        # Decoder layers
]
```

---

## Sensitivity Analysis

### Purpose

Identify which layers cause the most accuracy degradation when quantized.

### Manual Layer-by-Layer Analysis

```python
import torch
import modelopt.torch.quantization as mtq
from copy import deepcopy

def layer_sensitivity_analysis(model, calibration_data, eval_func):
    """
    Test each layer's impact on accuracy when quantized to INT8.

    Args:
        model: Your model
        calibration_data: List of input tensors
        eval_func: Function that returns accuracy metric (lower is better)
    """
    # Baseline (FP16)
    baseline_score = eval_func(model)
    print(f"Baseline: {baseline_score:.4f}")

    # Get quantizable layers
    quantizable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            quantizable_layers.append(name)

    results = {}

    for layer_name in quantizable_layers:
        # Config that only quantizes this one layer
        cfg = deepcopy(mtq.INT8_DEFAULT_CFG)
        cfg["quant_cfg"]["*"] = {"enable": False}
        cfg["quant_cfg"][f"*{layer_name}*weight_quantizer"] = {"num_bits": 8, "axis": 0}
        cfg["quant_cfg"][f"*{layer_name}*input_quantizer"] = {"num_bits": 8, "axis": None}

        model_copy = deepcopy(model)

        def fwd_loop(m):
            for img in calibration_data[:64]:
                with torch.no_grad():
                    m(img.cuda())

        mtq.quantize(model_copy, cfg, fwd_loop)

        score = eval_func(model_copy)
        degradation = score - baseline_score

        results[layer_name] = {
            "score": score,
            "degradation": degradation,
            "relative": degradation / baseline_score * 100
        }

        print(f"{layer_name}: degradation={degradation:+.4f} ({degradation/baseline_score*100:+.1f}%)")

        del model_copy
        torch.cuda.empty_cache()

    # Rank by sensitivity
    ranked = sorted(results.items(), key=lambda x: x[1]["degradation"], reverse=True)

    print("\nMost Sensitive Layers:")
    for name, r in ranked[:10]:
        print(f"  {name}: {r['degradation']:+.4f}")

    return results
```

### AutoQuantize (Built-in, LLM-focused)

```python
# Requires loss function - designed for LLMs
model, _ = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 8.0},
    data_loader=calib_loader,
    forward_step=lambda m, b: m(**b),
    loss_func=lambda out, batch: out.loss,
    quantization_formats=[mtq.INT8_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
    verbose=True,
)
```

---

## PTQ to QAT Workflow

### Step 1: PTQ with Sensitivity-Informed Config

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# After sensitivity analysis, create config
SKIP_LAYERS = ["*patch_embed*", "*blocks.0.*", "*norm*", "*head*"]

cfg = deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
for pattern in SKIP_LAYERS:
    cfg["quant_cfg"][pattern] = {"enable": False}

# Quantize (inserts fake quantizers)
mtq.quantize(model, cfg, forward_loop)

# Save PTQ checkpoint
mto.save(model, "model_ptq.pt")
```

### Step 2: QAT (When Training Data Available)

```python
import modelopt.torch.opt as mto

# Load fresh model
model = MyModel().cuda()

# Restore quantized state (same fake quantizers)
mto.restore(model, "model_ptq.pt")

# Enable training
model.train()
model.requires_grad_(True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for images, targets in train_dataloader:
        optimizer.zero_grad()
        output = model(images.cuda())  # Forward through fake quantizers
        loss = criterion(output, targets.cuda())
        loss.backward()
        optimizer.step()

# Save QAT checkpoint
mto.save(model, "model_qat.pt")
```

### Step 3: Export

```python
model = MyModel().cuda()
mto.restore(model, "model_qat.pt")
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "model_qat.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
)
```

---

## Speed/Accuracy Tradeoff Analysis

### Key Insight

PyTorch fake quantization does NOT speed up inference. Speed benefits come only from TensorRT INT8 kernels.

### Combined Analysis Script

```python
import subprocess
import re

def get_trt_latency(onnx_path, iterations=50):
    """Build TRT engine and measure latency."""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        "--int8", "--fp16",
        f"--iterations={iterations}",
        "--warmUp=200",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"mean = ([\d.]+) ms", result.stdout)
    return float(match.group(1)) if match else float('inf')

def analyze_tradeoff(configs, eval_func):
    """Analyze accuracy vs speed for different configs."""
    results = []

    for cfg in configs:
        # Quantize with this config
        quantize_model(cfg)

        accuracy = eval_func("temp.onnx")
        latency = get_trt_latency("temp.onnx")

        results.append({
            "config": cfg["name"],
            "accuracy": accuracy,
            "latency_ms": latency,
            "efficiency": accuracy / latency  # Lower is better for both
        })

    return results
```

### Decision Framework

```python
# Option A: Target accuracy, minimize latency
def best_for_accuracy(configs, target_accuracy):
    valid = [c for c in configs if c["accuracy"] <= target_accuracy]
    return min(valid, key=lambda c: c["latency_ms"])

# Option B: Target latency, maximize accuracy
def best_for_latency(configs, target_latency_ms):
    valid = [c for c in configs if c["latency_ms"] <= target_latency_ms]
    return min(valid, key=lambda c: c["accuracy"])
```

---

## Recommendations for ViT-based Models

### Layer Sensitivity Pattern

| Layer Group | Sensitivity | Speed Impact | Recommendation |
|-------------|-------------|--------------|----------------|
| patch_embed | High | Low (~2%) | Skip |
| Early blocks (0-3) | High | Medium (~10%) | Skip or test |
| Middle blocks (4-7) | Low | Medium (~10%) | Quantize |
| Late blocks (8-11) | Medium | Medium (~10%) | Test |
| Attention (all) | Medium-High | High (~30%) | Use FP8 or skip |
| MLP (all) | Low | High (~40%) | Quantize |
| LayerNorm | N/A | Very Low | Always FP16 |
| Output head | High | Low (~5%) | Skip |

### Recommended Starting Config

```python
from copy import deepcopy
import modelopt.torch.quantization as mtq

# For ViT-based models (Depth-Anything, DINOv2, etc.)
VIT_INT8_CFG = deepcopy(mtq.INT8_SMOOTHQUANT_CFG)

# Skip sensitive layers
VIT_INT8_CFG["quant_cfg"]["*patch_embed*"] = {"enable": False}
VIT_INT8_CFG["quant_cfg"]["*blocks.0.*"] = {"enable": False}
VIT_INT8_CFG["quant_cfg"]["*blocks.1.*"] = {"enable": False}
VIT_INT8_CFG["quant_cfg"]["*norm*"] = {"enable": False}
VIT_INT8_CFG["quant_cfg"]["*head*"] = {"enable": False}
VIT_INT8_CFG["quant_cfg"]["*decoder*"] = {"enable": False}
```

### Hybrid FP8/INT8 Config (Ada/Hopper GPUs)

```python
HYBRID_CFG = {
    "quant_cfg": {
        # INT8 for MLP (tolerant, big speedup)
        "*mlp*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*mlp*input_quantizer": {"num_bits": 8, "axis": None},

        # FP8 for attention (better accuracy than INT8)
        "*attn*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*attn*input_quantizer": {"num_bits": (4, 3), "axis": None},

        # Skip sensitive layers
        "*patch_embed*": {"enable": False},
        "*blocks.0.*": {"enable": False},
        "*blocks.1.*": {"enable": False},
        "*norm*": {"enable": False},
        "*head*": {"enable": False},

        "default": {"enable": False},
    },
    "algorithm": "smoothquant",
}
```

---

## Quick Reference

### Common Issues

| Issue | Solution |
|-------|----------|
| Backbone not quantizing | Check `nodes_to_exclude`, verify Q/DQ nodes with Netron |
| Poor accuracy | Try SmoothQuant, skip sensitive layers, use FP8 |
| ONNX export fails | Check opset version, dynamic shapes |
| TRT build fails | Check TRT version compatibility, use `--verbose` |

### Useful Commands

```bash
# ONNX quantization
python -m modelopt.onnx.quantization --onnx_path=model.onnx --quantize_mode=int8 ...

# TensorRT build with profiling
trtexec --onnx=model.onnx --int8 --fp16 --verbose --dumpProfile

# Check quantized ONNX
# Use Netron: https://netron.app/
```

### Key APIs

```python
# PyTorch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

mtq.quantize(model, config, forward_loop)  # PTQ
mtq.print_quant_summary(model)             # Inspect
mto.save(model, "checkpoint.pt")           # Save
mto.restore(model, "checkpoint.pt")        # Load

# ONNX
from modelopt.onnx.quantization import quantize
quantize(onnx_path, quantize_mode, calibration_data, ...)
```

---

## References

- [Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [Model Optimizer Docs](https://nvidia.github.io/Model-Optimizer/)
- [Quantization Guide](https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
