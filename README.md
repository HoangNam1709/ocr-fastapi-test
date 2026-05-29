# OCR FastAPI Test

FastAPI service dùng để nhận ảnh CCCD, xử lý OCR và trả về JSON gồm số CCCD, họ tên, ngày sinh, giới tính và nơi ở.

## 1. Cấu trúc chính

```text
ocr-fastapi-test/
├── app/
│   ├── main.py
│   └── ocr_service.py
├── models/
│   ├── ocr_model/
│   └── weights/
│       └── vgg_transformer.pth
├── uploads/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 2. Tải OCR weight

Tải file:

```text
https://vocr.vn/data/vietocr/vgg_transformer.pth
```

Đặt vào:

```text
models/weights/vgg_transformer.pth
```

Không nên push file `.pth` lên GitHub.

## 3. Chạy bằng Docker

```bash
docker compose up --build
```

API chạy tại:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## 4. Test OCR API

Endpoint:

```http
POST /ocr/cccd
```

Form-data:

```text
image = file ảnh CCCD
```

Ví dụ curl:

```bash
curl -X POST "http://localhost:8000/ocr/cccd" \
  -F "image=@test-images/cccd.jpg"
```

Response mẫu:

```json
{
  "status": "SUCCESS",
  "data": {
    "id": "031304005685",
    "name": "NGUYEN VAN A",
    "birth": "22/09/2004",
    "sex": "Nam",
    "place": "Ha Noi"
  }
}
```

## 5. Chạy local không dùng Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run with Docker Compose V1

```bash
docker-compose up --build
```

Run in background:

```bash
docker-compose up -d --build
```

Stop:

```bash
docker-compose down
```

## Swagger

```text
http://localhost:8000/docs
```

## 6. Ghi chú

Một số ảnh mờ, lóa, nghiêng hoặc thiếu sáng có thể khiến OCR trả về `null` hoặc sai field. Backend nên cho nhân viên xác nhận/sửa thủ công trước khi lưu DB.
