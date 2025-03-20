<h1 align="center">NHẬN DIỆN KHUÔN MẶT VỚI INSIGHT FACE </h1>

<div align="center">

</p>

![Made by AIoTLab](https://github.com/chinhliki/Nhan-dien-khuon-mat/blob/main/Screenshot%202025-03-17%20205953.png?raw=true)
</div>
## 📌 Giới thiệu  
Dự án này sử dụng **InsightFace** – một thư viện nhận diện khuôn mặt mạnh mẽ dựa trên Deep Learning, được tối ưu hóa cho GPU. InsightFace cung cấp các mô hình hiện đại để phát hiện, nhận dạng và so khớp khuôn mặt với độ chính xác cao.  

---  
## 🎯 Tính năng  
- 📸 **Phát hiện khuôn mặt** trong hình ảnh hoặc video.  
- 🔍 **Nhận dạng và so khớp khuôn mặt** với dữ liệu đã lưu trữ.  
- ⚡ **Hỗ trợ chạy trên GPU** để tăng tốc độ xử lý.  
- 🔗 **Tích hợp dễ dàng** với các ứng dụng nhận diện khuôn mặt khác.  

---  

## 🛠️ Cài đặt  

### 1️⃣ Yêu cầu hệ thống  
- 🐍 **Python** >= 3.8  
- 🎮 **CUDA** (nếu chạy trên GPU)  
- 📷 **OpenCV**  
- 🤖 **InsightFace**
## 📌 Mô Hình
![Face Detection](https://github.com/chinhliki/Nhan-dien-khuon-mat/blob/main/Screenshot%202025-03-17%20211520.png#:~:text=17%20205953.png-,Screenshot%202025%2D03%2D17%20211520.png,-Tri_tue_nhan_tao.pptx)  

### 2️⃣ Cài đặt thư viện  
Chạy lệnh sau để cài đặt các thư viện cần thiết:  

```bash  
pip install insightface opencv-python numpy matplotlib onnxruntime  
```

Nếu sử dụng GPU (CUDA), cài đặt **onnxruntime-gpu** thay vì **onnxruntime**:  

```bash  
pip install onnxruntime-gpu  
```

---  

## 🚀 Hướng dẫn sử dụng  

### 1️⃣ Nạp mô hình InsightFace  
```python  
from insightface.app import FaceAnalysis  

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # Chạy trên GPU  
app.prepare(ctx_id=0)  # ctx_id=0 nghĩa là sử dụng GPU  
```

### 2️⃣ Phát hiện khuôn mặt  
```python  
import cv2  

img = cv2.imread("test.jpg")  # Đọc ảnh đầu vào  
faces = app.get(img)  # Phát hiện khuôn mặt  

for face in faces:  
    bbox = face.bbox.astype(int)  # Lấy tọa độ khuôn mặt  
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Vẽ khung  

cv2.imshow("Result", img)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
```

### 3️⃣ So khớp khuôn mặt  
```python  
import numpy as np  

# Trích xuất vector đặc trưng khuôn mặt  
face_embedding1 = faces[0].normed_embedding  
face_embedding2 = faces[1].normed_embedding  

# Tính độ tương đồng  
similarity = np.dot(face_embedding1, face_embedding2)  
print(f"Độ tương đồng giữa hai khuôn mặt: {similarity:.2f}")  
```

---  

## 📌 Ứng dụng thực tế  
- 🏢 **Chấm công nhân viên** bằng khuôn mặt.  
- 🚪 **Kiểm soát ra vào** trong tòa nhà.  
- 🔎 **Tìm kiếm khuôn mặt** trong cơ sở dữ liệu.  

---  

## 🔥 Demo  
![Face Detection](https://github.com/chinhliki/Nhan-dien-khuon-mat/blob/main/Screenshot%202025-03-05%20020151.png#:~:text=README.md-,Screenshot%202025%2D03%2D05%20020151,-.png)  

---  

## 📝 Ghi chú  
- Nếu muốn nhận diện thời gian thực, có thể dùng **camera thay vì ảnh tĩnh** (`cv2.VideoCapture`).  
- InsightFace hỗ trợ nhiều mô hình khác nhau (`buffalo_l`, `buffalo_s`, `antelopev2`...), hãy thử nghiệm để tìm mô hình phù hợp nhất.  

---  

