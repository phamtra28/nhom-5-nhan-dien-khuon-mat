import gdown

# tải thư viện gdown trước khi chạy chương trình để cài model
# Danh sách file ONNX trên Google Drive
files = {
    "models/1k3d68.onnx": "1zbolbthizUlDuQ2n5fQeKGmBCG02cylr",  # Thay bằng ID thực tế
    "models/w600k_r50.onnx": "1eiYko9KdjWNP",  # Thay bằng ID thực tế
    "models/2d106det.onnxx": "1lL5GfKF4FI3YhCVK9nZpr6Z92Qap7O73",
    "models/det_10g.onnx": "1R-Ztl7wB-PzMhqLMwhYvlQRvqbIXGKf2",
    "models/genderage.onnx": "1dMymFGU0neo8F3gcKJcrLXW1tG6ximN1"
}

# Tải từng file về thư mục models/
for file_name, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Đang tải {file_name} từ Google Drive...")
    gdown.download(url, file_name, quiet=False)

print("Tải xuống hoàn tất!")
