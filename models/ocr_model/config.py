from typing import Tuple, Union, List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    

class LayoutConfig(_BaseConfig):
    model_dir: str = None
    model_ocr_dir: str | None = None
    
    backend: str = "onnx"
    onnx_variant: str = "alex"
    
    device: str = "cpu"
    visiable_device_idx: str = "0"
    batch_size: int = 8
    
    threshold: float = 0.35
    threshold_by_class: Dict = None
    
    layout_nms: bool = True
    layout_unclip_ratio: Union[float, Tuple[float, float], Dict] = None
    layout_merge_bboxes_mode: Union[str, Dict] = None
    
    benchmark_stats: bool = False