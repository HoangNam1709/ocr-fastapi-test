# Installation

Install dependencies using `uv`:

```bash
uv sync
```

Activate virtual environment:

```bash
source .venv/bin/activate
```

# Configuration

Example `config.yaml`:

```yaml
pipeline:
  model_dir: /path/to/layout_model

  model_ocr_dir: /tmp/vgg_transformer.pth

  backend: onnx
  onnx_variant: alex

  device: cpu
  visiable_device_idx: "0"
  batch_size: 8

  threshold: 0.35

  threshold_by_class:
    text: 0.3
    table: 0.5
    figure: 0.4

  layout_nms: true

  benchmark_stats: false
```

---

# Run

Run pipeline using `uv`:

```bash
uv run python main.py \
    --config "path config" \
    --image "path image"
```

Or using activated environment:

```bash
python main.py \
    --config config.yaml \
    --image id.png
```

---

# Example

```bash
uv run python main.py \
    --config config.yaml \
    --image id.png
```

---

# Output

The pipeline returns extracted layout/card information from the input document image.