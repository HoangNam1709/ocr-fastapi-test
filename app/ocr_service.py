from models.ocr_model.layout import PPDocLayoutV3Model
from models.ocr_model.config import LayoutConfig

model = PPDocLayoutV3Model(
    LayoutConfig(
        backend="onnx",
        onnx_variant="alex",
        device="cpu",
        model_ocr_dir="models/weights/vgg_transformer.pth"
    )
)

def extract_cccd(image_path: str):
    result = model.extract_card(image_path)

    return {
        "id": result["id"],
        "name": result["name"],
        "birth": result["birth"],
        "sex": result["sex"],
        "place": result["place"]
    }