# OCR FastAPI Test

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

POST /ocr/cccd

Input:
- image

Output:

```json
{
  "status": "SUCCESS",
  "data": {
    "id": "",
    "name": "",
    "birth": "",
    "sex": "",
    "place": ""
  }
}
```
Download vgg_transformer.pth from:
https://vocr.vn/data/vietocr/vgg_transformer.pth

Place it in:
models/weights/