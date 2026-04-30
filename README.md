# talkie-mlx-server

OpenAI-compatible API server for the [Talkie 1930 13B IT](https://talkie-lm.com/introducing-talkie) model running on Apple MLX with 8-bit quantization.

Talkie is a 13B parameter language model trained exclusively on pre-1931 English text. It has no knowledge of post-1930 events -- it writes in early 20th century prose about steam engines, telegraphs, and aeroplanes.

## Features

- **OpenAI-compatible API**: Drop-in `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints
- **8-bit quantization**: ~13 GB on disk, ~15 GB peak memory (vs 27 GB BF16)
- **Extended context**: 4096 tokens (2x the 2048 training length)
- **Apple Silicon native**: Runs on M-series Macs via MLX
- **Custom tokenizer support**: Handles Talkie's tiktoken tokenizer and chat template natively

## Requirements

- Apple Silicon Mac (M1+, M5 Pro 48 GB recommended)
- Python 3.13+
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) with Talkie support

### Installing mlx-lm with Talkie support

Talkie support is available via an open PR on mlx-lm:

```bash
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
```

Once [PR #1220](https://github.com/ml-explore/mlx-lm/pull/1220) is merged, a standard `pip install mlx-lm` will suffice.

### Downloading model weights

Pre-converted 8-bit MLX weights are needed. You can either:

1. **Download pre-converted BF16 and quantize yourself** (recommended):

```bash
pip install huggingface-hub
hf download zimengxiong/talkie-1930-13b-it-mlx --local-dir ./models/talkie-1930-13b-it-mlx

# Quantize to 8-bit
python -c "
from mlx_lm import load
from mlx_lm.utils import quantize_model, save_model, save_config
import json
from pathlib import Path

model, tokenizer = load('./models/talkie-1930-13b-it-mlx')
model, config = quantize_model(model, {}, bits=8, group_size=64)
out = Path('./models/talkie-1930-13b-it-mlx-8bit')
save_model(str(out), model, donate_model=True)
# Merge config
with open('./models/talkie-1930-13b-it-mlx/config.json') as f:
    orig = json.load(f)
orig.update(config)
orig['max_seq_len'] = 4096
orig['original_max_seq_len'] = 2048
with open(out / 'config.json', 'w') as f:
    json.dump(orig, f, indent=2)
# Copy tokenizer files
import shutil
for name in ['vocab.txt', 'tokenizer.json']:
    if (Path('./models/talkie-1930-13b-it-mlx') / name).exists():
        shutil.copy2(Path('./models/talkie-1930-13b-it-mlx') / name, out / name)
if (Path('./models/talkie-1930-13b-it-mlx') / 'mlx_talkie').exists():
    shutil.copytree(Path('./models/talkie-1930-13b-it-mlx') / 'mlx_talkie', out / 'mlx_talkie', dirs_exist_ok=True)
print('Done!')
"
```

2. **Use BF16 directly** (slower, ~10 tok/s, ~27 GB memory) by pointing the server to the BF16 model directory.

## Usage

### Starting the server

```bash
python server.py --port 8080 --host 127.0.0.1
```

The model loads at startup (~5s for 8-bit). Memory usage at idle: ~15 GB.

### API Endpoints

#### List models

```bash
curl http://127.0.0.1:8080/v1/models
```

#### Chat completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie",
    "messages": [
      {"role": "system", "content": "You are a helpful Edwardian-era assistant."},
      {"role": "user", "content": "What is the wireless telegraph?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

#### Text completions

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie",
    "prompt": "uctorDescribe the aeroplane.<|end|>\nector",
    "max_tokens": 100
  }'
```

### OpenAI provider configuration

Add to your OpenAI-compatible client config:

```yaml
base_url: http://127.0.0.1:8080/v1
api_key: local-no-key-needed
model: talkie-1930-13b-it-mlx-8bit
```

## Performance

Tested on M5 Pro 48 GB with 8-bit quantization:

| Metric | Value |
|--------|-------|
| Generation speed | ~16.8 tok/s |
| Prefill speed | ~600 tok/s |
| Peak memory (2K ctx) | ~15 GB |
| Peak memory (4K ctx) | ~17 GB |
| Model size on disk | ~13 GB |

## Context Window

| Context | Memory | Quality | Notes |
|---------|--------|---------|-------|
| 2K | 15.7 GB | Perfect | Trained context length |
| 4K | 17.0 GB | Good | 2x training, mostly reliable |
| 8K | 21.1 GB | Degraded | Loses factual recall |
| 12K+ | 24+ GB | Unreliable | Beyond training range |

The model was trained at 2048 tokens. Quality degrades beyond 4K due to attention confusion, not memory limits. The 8-bit quantization frees enough memory for 16K+ context, but the model's training limits useful context to ~4K.

To change the context window, edit `max_seq_len` in the model's `config.json`. The server reads this at load time.

## Chat Template

Talkie IT uses custom tokens:

| Token | Role |
|-------|------|
| `_segmentor` | System turn |
| `uctor` | User turn |
| `ECTOR` | Assistant turn |
| `<|end|>` | End of turn |

The server handles chat template rendering automatically via `mlx_lm`'s tokenizer wrapper.

## Architecture

Talkie uses a custom GPT variant not based on Llama/Qwen/Mistral:

- Per-head scalar gains on queries (TalkieHeadGain)
- Scalar weight gain on lm_head (TalkieWeightGain)
- Per-layer activation gains at (2*n_layer)^-0.5
- Embedding skip connections per layer
- QK-Norm (RMS norm on Q and K after RoPE)
- Custom tiktoken tokenizer (not HuggingFace)

This is why `mlx_lm server` doesn't work out of the box -- it expects standard HF tokenizers with `apply_chat_template`. This server uses `mlx_lm.load()` which properly wraps the tiktoken tokenizer via the PR branch.

## Known Issues

- **4-bit quantization broken**: The `sanitize()` method in the PR branch can't handle `.scales`/`.biases` weight names. Use 8-bit instead.
- **oMLX unsupported**: oMLX's model registry doesn't recognize `model_type: "talkie"`. Use mlx-lm directly.
- **`mlx_lm server` hangs**: The bundled `mlx_talkie/model.py` uses raw `matmul` which crashes on quantized weights. This server uses `mlx_lm.load()` with `QuantizedLinear` auto-conversion instead.

## Credits

- **Talkie model**: Alec Radford, Nick Levine, David Duvenaud -- [talkie-lm.com](https://talkie-lm.com)
- **MLX Talkie support**: ZimengXiong -- [PR #1220](https://github.com/ml-explore/mlx-lm/pull/1220)
- **Pre-converted MLX weights**: [zimengxiong/talkie-1930-13b-it-mlx](https://huggingface.co/zimengxiong/talkie-1930-13b-it-mlx)

## License

The Talkie model and this server code are provided for research and personal use. See the [Talkie repository](https://github.com/talkie-lm/talkie) for model license details.
