import os 
import gc
import time
from typing import Tuple, Union, List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from models.ocr_model.config import LayoutConfig
from models.ocr_model.utils import (
    compute_stats,
    load_image_to_pil,
    get_process_memory_mb,
    apply_layout_postprocess,
    get_gpu_memory_nvidia_smi_mb,
)

DEFAULT_IMAGE_SIZE: Tuple = (800, 800)

RESCALE_FACTOR: float = 1.0 / 255.0

PP_DOCLAYOUT_V3_LABELS = [
    "abstract",           # 0 
    "algorithm",          # 1 
    "aside_text",         # 2 
    "chart",              # 3 
    "content",            # 4 
    "display_formula",    # 5
    "doc_title",          # 6 
    "figure_title",       # 7 
    "footer",             # 8 
    "footer_image",       # 9 
    "footnote",           # 10  
    "formula_number",     # 11 
    "header",             # 12 
    "header_image",       # 13 
    "image",              # 14 
    "inline_formula",     # 15 
    "number",             # 16 
    "paragraph_title",    # 17 
    "reference",          # 18 
    "reference_content",  # 19  
    "seal",               # 20
    "table",              # 21 
    "text",               # 22 
    "vertical_text",      # 23 
    "vision_footnote",    # 24 
]

_LABEL_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#4169e1", "#ff7f50", "#00ced1", "#ff1493",
]


class PPDocLayoutV3Model:

    BACKEND_MODEL_MAP = {
        "paddle": "PaddlePaddle/PP-DocLayoutV3",
        "hf": "PaddlePaddle/PP-DocLayoutV3_safetensors",
        "huggingface": "PaddlePaddle/PP-DocLayoutV3_safetensors",
        "safetensors": "PaddlePaddle/PP-DocLayoutV3_safetensors",
    }
    
    ONNX_MODEL_MAP = {
        "bei": "Bei0001/PP-DocLayoutV3-ONNX",
        "alexdinh": "alex-dinh/PP-DocLayoutV3-ONNX",
        "alex": "alex-dinh/PP-DocLayoutV3-ONNX",
        "dinh": "alex-dinh/PP-DocLayoutV3-ONNX",
    }

    def __init__(self, config: "LayoutConfig"):
        
        self.config = config
        
        self.model_dir = config.model_dir
        self.model_ocr_dir = config.model_ocr_dir
        self.backend = config.backend.lower()
        self.onnx_variant = config.onnx_variant.lower()
        self.config_device = config.device
        self.visiable_device_idx = config.visiable_device_idx
        self.batch_size = config.batch_size
        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class or {}
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.benchmark_stats = config.benchmark_stats

        self.id2label: Dict[int, str] = {
            i: label for i, label in enumerate(PP_DOCLAYOUT_V3_LABELS)
        }
        
        self._model = None
        self._image_processor = None

    def _resolve_model_path(self) -> str:
        if self.backend == "onnx":
            if self.onnx_variant not in self.ONNX_MODEL_MAP:
                raise ValueError(f"Unsupported ONNX variant: {self.onnx_variant}")
            return self.ONNX_MODEL_MAP[self.onnx_variant]
        if self.backend not in self.BACKEND_MODEL_MAP:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return self.BACKEND_MODEL_MAP[self.backend]
    
    def _setup_device(self) -> Tuple[str, torch.dtype]:

        if self.config_device is not None:
            device = self.config_device
            dtype = torch.float32
        elif torch.cuda.is_available() and self.visiable_device_idx:
            device = f"cuda:{self.visiable_device_idx}"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        return device, dtype

    def _label_to_color(self, cls_id: int) -> str:
        return _LABEL_COLORS[cls_id % len(_LABEL_COLORS)]
    
    def _preprocess_single_image(
        self, 
        image: Image.Image,
        target_size: Tuple = DEFAULT_IMAGE_SIZE,
        rescale_factor: float = RESCALE_FACTOR
    ) -> Tuple:

        if self.backend in ("hf", "huggingface", "safetensors"):
            import torchvision.transforms.v2.functional as tvF
            from torchvision.transforms import InterpolationMode

            target_sizes = image.size[1], image.size[0]
            pixel_values = tvF.pil_to_tensor(image)
            pixel_values = tvF.resize(
                pixel_values,
                size=target_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=False,
            )
            pixel_values = pixel_values.to(dtype=self._dtype) * rescale_factor

            return pixel_values, target_sizes

        elif self.backend in ("onnx", "paddle"):
            import cv2

            np_image = np.array(image)
            orig_h, orig_w = np_image.shape[:2]
            scale_h, scale_w = target_size[1] / orig_h, target_size[0] / orig_w
            target_sizes = [scale_h, scale_w]

            resized = cv2.resize(np_image, target_size, interpolation=cv2.INTER_LINEAR)
            blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) * rescale_factor
            pixel_values = blob.transpose(2, 0, 1)[np.newaxis, ...]

            if self.backend == "paddle":
                target_sizes= np.array([target_sizes], dtype=np.float32)
            
            return pixel_values, target_sizes

        else:
            raise ValueError(f"Unsupport backend: {self.backend}")

    def _apply_per_class_threshold(self, raw_results: List[Dict]) -> List[Dict]:
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}
        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            if isinstance(key, str):
                if key in label2id:
                    class_thresholds[label2id[key]] = float(value)
                else:
                    logger.warning(
                        "Unknown class name '%s' in threshold_by_class; "
                        "this entry will be ignored. Known classes: %s",
                        key,
                        ", ".join(sorted(label2id.keys())),
                    )
            else:
                class_thresholds[int(key)] = float(value)
        fallback = self.threshold
        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]
            thresholds = torch.full_like(scores, fallback)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh
            keep = scores >= thresholds
            new_result = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                new_result["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                new_result["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list) if k
                ]
            filtered.append(new_result)
        return filtered
    
    # ------------------------ API ------------------------ #
    def load_model(self) -> None:
        from huggingface_hub import snapshot_download

        self.mem_before = get_process_memory_mb()
        self.gpu_before = get_gpu_memory_nvidia_smi_mb()
    
        model_path = None
        if self.model_dir is not None and os.path.exists(self.model_dir):
            model_path = self.model_dir
        else:
            logger.warning(f"Model weights does not exists {model_path}")
            logger.debug(f"Download model from HuggingFace")
        
        model_path = self._resolve_model_path()
        self._device, self._dtype = self._setup_device()
        
        logger.info(f"Initializing PP-DocLayoutV3 from {self.backend}")
        logger.info(f"Using device: {self._device}")

        if self.backend in ("hf", "huggingface", "safetensors"):
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._image_processor = AutoImageProcessor.from_pretrained(model_path)

            self._model = AutoModelForObjectDetection.from_pretrained(
                model_path
            ).to(self._device, dtype=self._dtype).eval()

        elif self.backend == "onnx":
            import onnxruntime as ort
            from transformers import AutoImageProcessor

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count()
            sess_options.inter_op_num_threads = 1
            sess_options.enable_mem_pattern = True
            sess_options.enable_mem_reuse = True
            
            if self._device == "cpu":
                providers, provider_options = ["CPUExecutionProvider"], [{}]
            
            elif self._device.startswith("cuda"):
                providers, provider_options = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    [{"device_id": self.visiable_device_idx}, {}],
                )

            elif self._device == "tensorrt":
                providers, provider_options = (
                    ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                    [{"device_id": self.visiable_device_idx}, {"device_id": self.visiable_device_idx}, {}],
                )

            elif self._device == "openvino":
                providers, provider_options = (
                    ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
                    [{"device_type": "CPU_FP32"}, {}]
                )

            elif self._device == "directml":
                providers, provider_options = (
                    ["DmlExecutionProvider", "CPUExecutionProvider"],
                    [{"device_id": self.visiable_device_idx}, {}]
                )

            else:
                ValueError(f"Unknown device: {self._device!r}")

            if self.onnx_variant == "bei":
                
                downloaded_model = snapshot_download(model_path)

                self._image_processor = AutoImageProcessor.from_pretrained(downloaded_model)

                
                self._model = ort.InferenceSession(
                    downloaded_model + "/PP-DocLayoutV3.onnx",
                    sess_options=sess_options, 
                    providers=providers, 
                    provider_options=provider_options
                )
                logger.info(
                    f"ONNX providers: {self._model.get_providers()}\n "
                    f"ONNX provider options: {self._model.get_provider_options()}"
                )
                self.batch_size = 1
            
            elif self.onnx_variant in ("alex", "alexdinh", "dinh"):

                downloaded_model = snapshot_download(model_path)
                self._image_processor = None
                self._model = ort.InferenceSession(
                    downloaded_model + "/PP-DocLayoutV3.onnx",
                    sess_options=sess_options, 
                    providers=providers, 
                    provider_options=provider_options
                )

        elif self.backend == "paddle":
            from paddle.inference import Config, create_predictor

            downloaded_model = snapshot_download(model_path)
            
            config = Config(
                f"{downloaded_model}/inference.json",
                f"{downloaded_model}/inference.pdiparams",
            )

            if torch.cuda.is_available():
                config.enable_use_gpu(512, 0)
            config.disable_mkldnn()
            config.enable_memory_optim(False)
            config.switch_ir_optim(True)
            

            self._image_processor = None
            self._model = create_predictor(config)

        else:
            raise ValueError(f"Unsupport backend: {self.backend}")

        self.mem_after_load = get_process_memory_mb()
        self.gpu_after_load = get_gpu_memory_nvidia_smi_mb()
    
    def predict(self, image: Union[Image.Image, str, bytes, np.ndarray]) -> List[Dict]:
        original_batch_size = self.batch_size
        self.batch_size = 1
        try:
            return self.batch_predict(images=[image])[0]
        finally:
            self.batch_size = original_batch_size
        
    def batch_predict(self, images: List[Union[Image.Image, str, bytes, np.ndarray]]) -> List[List[Dict]]:

        if len(images) == 0:
            return []
        
        results: List[List[Dict]] = []
        fwd_latencies = []
        e2e_latencies = []
        
        with torch.no_grad():
            with tqdm(total=len(images), desc="Layout Predict") as pbar:                
                for start in range(0, len(images), self.batch_size):
                    batch_images = images[start : start + self.batch_size]

                    e2e_start = time.perf_counter()
                    
                    pixel_values_list = []
                    model_input_sizes = []
                    orig_sizes = [] 

                    for image in batch_images:
                        pil_image = load_image_to_pil(image)
                        orig_h, orig_w = pil_image.size[1], pil_image.size[0]
                        orig_sizes.append((orig_h, orig_w))

                        pixel_values, model_input_size = self._preprocess_single_image(pil_image)
                        pixel_values_list.append(pixel_values)
                        model_input_sizes.append(model_input_size)
                    
                    predictions = []

                    if self.backend in ("hf", "huggingface", "safetensors"):
                        batch_tensor = torch.stack(pixel_values_list, dim=0).to(self._device)
                        
                        # Forward
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        fwd_start = time.perf_counter()
                        outputs = self._model(pixel_values=batch_tensor)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        fwd_latencies.append(time.perf_counter() - fwd_start)

                        predictions = self._image_processor.post_process_object_detection(
                            outputs, target_sizes=model_input_sizes, threshold=self.threshold
                        )
                        e2e_latencies.append(time.perf_counter() - e2e_start)
                        
                    elif self.backend == "onnx":
                        if self.onnx_variant in ("alex", "alexdinh", "dinh"):
                            blobs = np.concatenate(pixel_values_list, axis=0)

                            input_names = [i.name for i in self._model.get_inputs()]
                            output_names = [o.name for o in self._model.get_outputs()]
                            input_feed = {
                                input_names[0]: [np.array(list(DEFAULT_IMAGE_SIZE), dtype=np.float32)],
                                input_names[1]: blobs,
                                input_names[2]: model_input_sizes,
                            }

                            fwd_start = time.perf_counter()
                            output = self._model.run(output_names, input_feed)[0]
                            fwd_latencies.append(time.perf_counter() - fwd_start)
                            e2e_latencies.append(time.perf_counter() - e2e_start)
                            
                            raw = output[output[:, 1] > self.threshold]
                            raw = raw[np.argsort(raw[:, 6])]

                            boxes_int = raw[:, 2:6].astype(np.int32)
                            polygon_points = [
                                np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
                                for x1, y1, x2, y2 in boxes_int
                            ]
                            predictions.append({
                                "scores": torch.as_tensor(raw[:, 1]),
                                "labels": torch.as_tensor(raw[:, 0], dtype=torch.int64),
                                "boxes": torch.as_tensor(raw[:, 2:6]),
                                "polygon_points": polygon_points,
                                "order_seq": torch.as_tensor(raw[:, 6], dtype=torch.int64)
                            })
                        
                        elif self.onnx_variant == "bei":
                            pil_batch = [load_image_to_pil(img) for img in batch_images]
                            
                            inputs = self._image_processor(images=pil_batch, return_tensors="np")
                            batch_pixel_values = inputs["pixel_values"]  
                            target_sizes = [(pil.size[1], pil.size[0]) for pil in pil_batch]  # (orig_h, orig_w)

                            fwd_start = time.perf_counter()
                            raw_outputs = self._model.run(
                                ["logits", "pred_boxes", "out_masks", "order_logits"],
                                {"pixel_values": batch_pixel_values},
                            )
                            fwd_latencies.append(time.perf_counter() - fwd_start)
                            e2e_latencies.append(time.perf_counter() - e2e_start)
                                                        
                            class _Outputs:
                                __slots__ = ("logits", "pred_boxes", "out_masks", "order_logits")
                                def __init__(self, *tensors):
                                    for attr, t in zip(self.__slots__, tensors):
                                        setattr(self, attr, torch.from_numpy(t))

                            predictions = self._image_processor.post_process_object_detection(
                                _Outputs(*raw_outputs),
                                target_sizes=target_sizes,
                                threshold=self.threshold,
                            )

                        else:
                            raise ValueError(f"Unsupport ONNX variant: {self.onnx_variant}")

                    elif self.backend == "paddle":
                        blobs = np.concatenate(pixel_values_list, axis=0)

                        data = {
                            "im_shape": np.tile(
                                np.array(list(DEFAULT_IMAGE_SIZE), dtype=np.float32),
                                (len(batch_images), 1)
                            ),
                            "image": blobs,
                            "scale_factor": np.concatenate(model_input_sizes, axis=0).astype(np.float32),
                        }

                        input_names = self._model.get_input_names()
                        output_names = self._model.get_output_names()
                        for name in input_names:
                            handle = self._model.get_input_handle(name)
                            handle.reshape(data[name].shape)
                            handle.copy_from_cpu(data[name])

                        fwd_start = time.perf_counter()
                        self._model.run()
                        paddle_output = {}
                        for name in output_names:
                            handle = self._model.get_output_handle(name)
                            paddle_output[name] = handle.copy_to_cpu()

                        fwd_latencies.append(time.perf_counter() - fwd_start)
                        e2e_latencies.append(time.perf_counter() - e2e_start)
                                                
                        raw = paddle_output[output_names[0]]
                        raw = raw[raw[:, 1] > self.threshold]
                        raw = raw[np.argsort(raw[:, 6])] if raw.shape[1] > 6 else raw

                        boxes_int = raw[:, 2:6].astype(np.int32)
                        polygon_points = [
                            np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int32)
                            for x1, y1, x2, y2 in boxes_int
                        ]

                        predictions.append({
                            "scores": torch.as_tensor(raw[:, 1]),
                            "labels": torch.as_tensor(raw[:, 0], dtype=torch.int64),
                            "boxes": torch.as_tensor(raw[:, 2:6]),
                            "polygon_points": polygon_points,
                            "order_seq": torch.as_tensor(raw[:, 6], dtype=torch.int64)
                        })

                    for pred in predictions:
                        if "order_seq" not in pred:
                            pred["order_seq"] = torch.arange(len(pred["scores"]))
                        if "polygon_points" not in pred:
                            pred["polygon_points"] = []
                            
                    if self.threshold_by_class:
                        predictions = self._apply_per_class_threshold(predictions)
 
                    batch_img_sizes_wh = [(w, h) for h, w in orig_sizes]
                    batch_results = apply_layout_postprocess(
                        predictions,
                        id2label=self.id2label,
                        img_sizes=batch_img_sizes_wh,
                        layout_nms=self.layout_nms,
                        layout_unclip_ratio=self.layout_unclip_ratio,
                        layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
                    )
                    results.extend(batch_results)
 
                    pbar.update(len(batch_images))

            mem_after = get_process_memory_mb()
            gpu_after = get_gpu_memory_nvidia_smi_mb()
            
        if  self.benchmark_stats:
            fwd_stats = compute_stats(fwd_latencies)
            e2e_stats = compute_stats(e2e_latencies)
            
            stats = {
                **e2e_stats,
                "fwd_mean_ms": fwd_stats["mean_ms"],
                "fwd_median_ms": fwd_stats["median_ms"],
                "ram_model_mb": self.mem_after_load - self.mem_before,
                "ram_total_mb": mem_after,
                "gpu_model_mb": self.gpu_after_load - self.gpu_before,
                "gpu_peak_mb": gpu_after,
            }
            logger.info(f"Benckmark statistic of {self.backend}:\n {stats}")
        
        #del self._model, self._image_processor
        
        #if torch.cuda.is_available():
        #   torch.cuda.empty_cache()
        #gc.collect()

        return results

    def visualize(
        self,
        image: Union[Image.Image, str, bytes, np.ndarray],
        results: List[Dict],
    ) -> Image.Image:
        image = load_image_to_pil(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        sorted_results = sorted(
            results,
            key=lambda item: (
                item.get("order") or 10 ** 9,
                (item.get("coordinate") or [0, 0, 0, 0])[1],
            ),
        )
        for read_idx, res in enumerate(sorted_results, start=1):
            xmin, ymin, xmax, ymax = res["coordinate"]
            color = self._label_to_color(res["cls_id"])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            text = f"{read_idx}: {res['label']} {res['score']:.2f}"
            text_top = int(round(ymin))
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            pad = 3
            if text_top - text_height - pad * 2 >= 0:
                text_bg_top = text_top - text_height - pad * 2
            else:
                text_bg_top = text_top
            text_bg_bottom = text_bg_top + text_height + pad * 2
            text_bg_right = int(round(xmax))
            text_bg_left = text_bg_right - text_width - pad * 2
            draw.rectangle(
                [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom],
                fill=color,
            )
            draw.text(
                (text_bg_left + pad, text_bg_top + pad),
                text,
                fill="white",
                font=font,
            )
        return image

    def crop_image_region(
        self, 
        image: Union[Image.Image, str, bytes, np.ndarray],
        results: List[Dict],
        labels: Optional[Union[str, List[str]]] = None
    ) -> List[Image.Image]:
        
        image = load_image_to_pil(image)
        
        img_w, img_h = image.size

        if labels is not None:
            if isinstance(labels, str):
                labels = {labels}
            else:
                labels = set(labels)
                
        sorted_results = sorted(
            results,
            key=lambda item: (
                item.get("order") or 10 ** 9,
                (item.get("coordinate") or [0, 0, 0, 0])[1],
            ),
        )
        
        crops: List[Dict] = []
        read_idx = 0
        for res in sorted_results:
            read_idx += 1

            # Apply label filter
            if labels is not None and res.get("label") not in labels:
                continue
 
            xmin, ymin, xmax, ymax = res["coordinate"]
            box_w = xmax - xmin
            box_h = ymax - ymin
 
            # Expand by padding 10% fraction of box dimensions on each side
            pad_x = int(round(box_w * 0.10))
            pad_y = int(round(box_h * 0.10))
 
            cx1 = max(0, xmin - pad_x)
            cy1 = max(0, ymin - pad_y)
            cx2 = min(img_w, xmax + pad_x)
            cy2 = min(img_h, ymax + pad_y)
 
            crop_img = image.crop((cx1, cy1, cx2, cy2))
 
            crops.append({
                "label": res["label"],
                "pil_image": crop_img
            })
 
        return crops

    def extract_card(self, image: Union[Image.Image, str, bytes, np.ndarray]) -> Dict:
        import re
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
        
        if self._model is None:
            self.load_model()
        
        config = Cfg.load_config_from_name('vgg_transformer')
        if self.model_ocr_dir is not None:
            config['weights'] = self.model_ocr_dir
            
        config['device'] =  self._device
        detector = Predictor(config)
        
        results = self.predict(image)
        crops = self.crop_image_region(image, results, labels=["text", "image"])

        text_pil_images = []
        image_crop = None

        for crop in crops:
            if crop["label"] == "text":
                text_pil_images.append(crop["pil_image"])
            else:
                image_crop = crop["pil_image"]

        text_lines = detector.predict_batch(text_pil_images)
        full_text = "\n".join(text_lines)
    
        result = {
            "image": image_crop,
            "id":     None,
            "name":   None,
            "birth":  None,
            "sex":    None,
            "place": None,
        }

        # ID - 12 chữ số liên tiếp
        m = re.search(r"\b(\d{12})\b", full_text)
        if m:
            result["id"] = m.group(1)

        # Tên - dòng ngay sau dòng chứa "Full name"
        m = re.search(r"Full\s*name[^\n]*\n([^\n]+)", full_text, re.I)
        if m:
            result["name"] = m.group(1).strip()

        # Ngày sinh
        m = re.search(r"(\d{2}/\d{2}/\d{4})", full_text)
        if m:
            result["birth"] = m.group(1)

        # Giới tính - bắt cả "Male"/"Female"/"Nam"/"Nữ"
        m = re.search(r"\bSex\s+(Nữ|Nam|Female|Male)\b", full_text, re.I)
        if m:
            result["sex"] = m.group(1)

        # Quê quán
        m = re.search(r"(?:Quê\s+qu\S*|Place\s+of\s+or?\w*)[^\n]*\n([^\n]+)", full_text, re.I)
        if m:
            result["place"] = m.group(1).strip()
        
        return result
    # ------------------------ /API ------------------------ #

if __name__ == "__main__":
    
    image_path = [
        "/mnt/sda1/home/bmestaging/duong/anh/id.png",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0001.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0002.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0003.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0004.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0005.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0006.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0007.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0008.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0009.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0010.jpg",
        "/mnt/sda1/home/bmestaging/duong/anh/QT.KT.03_page-0011.jpg",
    ]

    layout_model = PPDocLayoutV3Model(config=LayoutConfig(
        backend="onnx",
        onnx_variant="alex",
        benchmark_stats=True
    ))

    in4 = layout_model.extract_card(
        image_path[0]
    )
    