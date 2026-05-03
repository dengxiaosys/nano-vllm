# How to Run Nano-vLLM

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU，Compute Capability ≥ 6.0（Pascal 及以上） |
| CUDA | 12.4+ |
| Python | 3.10 ~ 3.12 |
| Conda | 用于管理 Python 环境 |

> **注意：** Flash Attention 和 `torch.compile` 需要 Compute Capability ≥ 8.0（Ampere 及以上，如 RTX 3090/A100 等）。如果你的 GPU 为旧架构（如 GTX 10xx / RTX 20xx），项目会自动回退到兼容模式（PyTorch SDPA + float16），无需额外操作。

## 2. 创建 Conda 环境

```bash
conda create -n nanovllm python=3.11 -y
conda activate nanovllm
```

## 3. 安装依赖

### 3.1 安装 PyTorch

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3.2 安装 Flash Attention

**方式 A：预编译 Wheel（推荐，速度快）**

根据你的 PyTorch 版本和 Python 版本选择对应的 wheel。以下命令适用于 PyTorch 2.6 + CUDA 12 + Python 3.11：

```bash
# 先检查你的 PyTorch CXX11_ABI
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

- 输出 `True` → 安装 `cxx11abiTRUE` 版本
- 输出 `False` → 安装 `cxx11abiFALSE` 版本

```bash
# cxx11abiFALSE 版本（PyTorch pip 安装的默认情况）
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# cxx11abiTRUE 版本
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```

> 其他 Python 版本请将 `cp311` 替换为对应版本（如 `cp310`、`cp312`），更多版本见 [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases)。

**方式 B：从源码编译（耗时较长，约 10-20 分钟）**

```bash
pip install flash-attn --no-build-isolation
```

### 3.3 安装其他依赖

```bash
pip install transformers xxhash
```

### 3.4 安装 nano-vllm

在项目根目录下执行：

```bash
pip install -e .
```

## 4. 下载模型

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

模型默认存放路径为 `~/huggingface/Qwen3-0.6B/`，与 `example.py` 中的路径一致。

## 5. 运行

### 5.1 基础推理示例

```bash
conda activate nanovllm
cd /home/dengxiao/code_repos/nano-vllm
python example.py
```

### 5.2 代码示例

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello!"}],
        tokenize=False,
        add_generation_prompt=True,
    )
]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### 5.3 性能基准测试

```bash
python bench.py
```

## 6. 关键参数说明

### LLM 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | 必填 | 模型路径 |
| `enforce_eager` | `False` | 是否禁用 CUDA Graph，调试时建议设为 `True` |
| `tensor_parallel_size` | `1` | 张量并行数 |
| `max_model_len` | `4096` | 最大序列长度 |
| `gpu_memory_utilization` | `0.9` | GPU 显存利用率 |

### SamplingParams 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | `1.0` | 采样温度 |
| `max_tokens` | `16` | 最大生成 token 数 |
| `ignore_eos` | `False` | 是否忽略 EOS 继续生成 |

## 7. 旧 GPU 兼容性说明

如果你的 GPU 架构早于 Ampere（如 GTX 1060 / RTX 2080 等），项目已内置自动适配：

- **Flash Attention** → 自动回退到 PyTorch `scaled_dot_product_attention`
- **torch.compile** → 自动跳过，使用 eager 模式
- **bfloat16** → 自动降级为 float16

无需手动配置，代码会根据 `torch.cuda.get_device_capability()` 自动判断。

## 8. 常见问题

### Q: `ImportError: cannot import name 'get_num_sms' from 'torch._inductor.utils'`

PyTorch 版本冲突，确保只安装了一个版本的 PyTorch：

```bash
pip uninstall torch -y
rm -rf /path/to/env/lib/python3.11/site-packages/torch/
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Q: `undefined symbol` 报错（flash_attn_2_cuda.so）

Flash Attention wheel 与 PyTorch 的 CXX11_ABI 不匹配，参见 [3.2 节](#32-安装-flash-attention)选择正确的 wheel。

### Q: `FlashAttention only supports Ampere GPUs or newer`

GPU 架构不支持 Flash Attention，项目已自动回退到 SDPA，无需处理。

### Q: `Triton only supports devices of CUDA Capability >= 7.0`

GPU 架构不支持 `torch.compile`，项目已自动跳过编译，无需处理。

### Q: CUDA Out of Memory

尝试降低以下参数：
- 减小 `max_model_len`（如设为 2048）
- 减小 `gpu_memory_utilization`（如设为 0.8）
- 使用 `enforce_eager=True` 禁用 CUDA Graph
