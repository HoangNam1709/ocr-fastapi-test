import os
import io 
import math
import base64
from typing import Union

import numpy as np
from PIL import Image

def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 15000,
):
    """
    Smart resize for images.

    Ensures:
    1. Height and width are divisible by the given factors
    2. Total pixels are within [min_pixels, max_pixels]
    3. Keeps aspect ratio as much as possible

    Args:
        t: Temporal dimension.
        h: Height.
        w: Width.
        t_factor: Temporal factor.
        h_factor: Height factor.
        w_factor: Width factor.
        min_pixels: Minimum pixels.
        max_pixels: Maximum pixels.

    Returns:
        (new_h, new_w)
    """
    assert t >= t_factor, "Temporal dimension must be greater than the factor."

    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor

    return h_bar, w_bar

def load_image_to_pil(
    image: Union[Image.Image, str, bytes, np.ndarray]
) -> Image.Image:
    import io, base64

    def _try_decode_base64_to_image_bytes(s: str) -> bytes | None:
        candidate = "".join(str(s).split())
        if len(candidate) < 32:
            return None
        if candidate.startswith("<|base64|>"):
            candidate = candidate[len("<|base64|>"):]
        if "." in candidate and len(candidate.rsplit(".", 1)[-1]) <= 5:
            return None
        pad = (-len(candidate)) % 4
        if pad:
            candidate = candidate + ("=" * pad)
        try:
            return base64.b64decode(candidate, validate=True)
        except Exception:
            return None

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        if image.startswith("file://"):
            image = image[7:]
        if os.path.isfile(image):
            with open(image, "rb") as f:
                image_data = f.read()
            pil_image = Image.open(io.BytesIO(image_data))
        elif image.startswith("data:image/"):
            image_data = base64.b64decode(image.split(",")[1])
            pil_image = Image.open(io.BytesIO(image_data))
        else:   
            decoded = _try_decode_base64_to_image_bytes(image)
            if decoded is None:
                raise ValueError(f"Invalid image source: {image}")
            pil_image = Image.open(io.BytesIO(decoded))
    elif isinstance(image, bytes):
        pil_image = Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image source type: {type(image)}")

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    return pil_image

def load_image_to_base64(
    image: Union[Image.Image, str, bytes, np.ndarray],
    t_patch_size: int,
    max_pixels: int,
    image_format: str,
    patch_expand_factor: int = 1,
    min_pixels: int = 112 * 112,
):
    image = load_image_to_pil(image)
    
    # Original size
    w, h = image.size

    # Compute new size
    h_bar, w_bar = smart_resize(
        t=t_patch_size,
        h=h,
        w=w,
        t_factor=t_patch_size,
        h_factor=14 * 2 * patch_expand_factor,
        w_factor=14 * 2 * patch_expand_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

    # Encode as bytes
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    buffered.seek(0)
    image_data = buffered.getvalue()

    # Convert bytes to base64
    base64_encoded_data = base64.b64encode(image_data)
    image_base64 = base64_encoded_data.decode("utf-8")

    return image_base64
