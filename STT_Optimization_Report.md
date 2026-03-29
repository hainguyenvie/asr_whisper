# Báo Cáo Tối Ưu Hóa Trí Tuệ Nhân Tạo (Speech-to-Text)
**Dự án**: Viettel Internal Meeting STT
**Mục tiêu**: Tối ưu hóa tối đa độ chính xác của văn bản đầu ra, triệt tiêu ảo giác (hallucinations) và nhận diện đúng 100% các từ vựng tiếng Anh chuyên ngành.

---

## 1. Hành Trình Thử Nghiệm Thực Tế (Empirical Experiments)

Trong quá trình nghiên cứu, chúng tôi đã viết ra các kịch bản kiểm thử độc lập để chạy Benchmark trực tiếp trên dữ liệu thật (`audio2.m4a`). Dưới đây là các kết luận rút ra từ từng phép thử nghiệm:

### Thử nghiệm 1: Lựa chọn Mô hình (PhoWhisper vs. Whisper-Large-v3-Turbo)
- **Hành động**: So sánh `vinai/PhoWhisper-large` (finetune riêng cho Tiếng Việt) và mô hình toàn cầu `Whisper-Large-v3-Turbo` của OpenAI.
- **Kết quả**: 
  - `PhoWhisper` tỏ ra chậm hơn và vấp ngã liên tục khi gặp các từ vựng lai (Code-switching) như *Transformer, Ericsson, M8X1*.
  - `Large-v3-Turbo` không chỉ chạy nhanh gấp đôi mà còn xử lý cực kỳ mượt mà song ngữ Anh-Việt.
- **👉 Kết luận**: Dứt khoát chọn **Whisper-large-v3-Turbo**.

### Thử nghiệm 2: Cấu hình Sinh chữ gốc (Hyperparameters Tuning)
- **Hành động**: Test các lời khuyên từ cộng đồng AI quốc tế như dùng `beam_size=1` (Greedy Search) và thêm `repetition_penalty=1.2` để chống lặp từ.
- **Kết quả**:
  - `repetition_penalty` là **thuốc độc** đối với Tiếng Việt. Do Tiếng Việt là từ đơn âm tiết (phải ghép từ), việc cấm lặp từ khiến AI cố tình viết sai chính tả ngớ ngẩn (*báo cáó*, *công nghịch*) để lách luật.
  - `beam_size=1` suy nghĩ quá nông cạn, không tìm ra được token im lặng `<|nospeech|>` gẫn dến bịa chuyện.
- **👉 Kết luận**: Thiết lập `beam_size=5` và thả tự do không dùng Penalty.

### Thử nghiệm 3: Tuyệt chiêu trị sai tên Dự án (Hotwords vs. Prompts)
- **Hành động**: Khảo sát hiện tượng AI thi thoảng nhận diện nhầm *Netmind* thành *NetMai* / *NETMINE x1*.
- **Kết quả**: Chức năng `initial_prompt` chỉ giúp mớm ngữ cảnh nhẹ. Nhưng khi ta đưa **từ khóa thẳng vào `hotwords`**, model bị ép cộng bộ trọng số (Logit Bias) cực lớn. Ngay lập tức, 100% các từ *NetMai, PowerServing* đã được bẻ gập lại thành **Netmind** và **Power Saving** cực kì chuẩn xác.
- **👉 Kết luận**: `hotwords` là "Vũ khí Bí mật" bắt buộc phải có cho các phòng ban kỹ thuật đặc thù.

### Thử nghiệm 4: Thuật toán Cắt Nhiễu Âm Thanh Cấp Cao (AI Denoising)
- **Hành động**: Phân tích việc audio bị dính tiếng quạt gió và máy lạnh. Thử chạy qua bộ lọc cắt nhiễu spectral gating (`noisereduce` - 20% đến 80%).
- **Kết quả**: Whisper vốn đã dung nạp 5 triệu giờ nhiễu môi trường trong lúc huấn luyện. Việc dùng phần mềm ngoài gọt bớt sóng âm sẽ tạo ra âm thanh nghe như "Giọng Robot" chập chờn. Whisper rơi thẳng vào thế **bế tắc** và sinh ra hội chứng nói lắp (vấp đĩa): *"tư tư tư", "hệ thống hệ thống hệ thống"*.
- **👉 Kết luận**: Cắt bỏ hoàn toàn các lớp AI Denoise ngoài vòng vi xử lý. Chỉ dùng **Bandpass EQ** thuần túy (lọc dải tần người nói 100Hz - 8000Hz).

### Thử nghiệm 5: Bắt mạch Vòng Lặp Vô Tận (Temperature Override)
- **Hành động**: Cố tình khóa `temperature` dưới nấc `0.6` để ép AI không được sinh ảo giác.
- **Kết quả**: Khi gặp nấc cụt hoặc tiếng ồn tạp, Whisper bị vướng vào một vòng lặp từ. Thông thường, nó dùng cơ chế cảnh báo *Compression Ratio > 2.4* để nhảy lên nấc Temperature cao hơn (`0.8`, `1.0`), khởi động sự sáng tạo để THOÁT ra khỏi vòng lặp. Nhưng vì chúng ta khóa mất lối thoát, AI đành đi vào ngõ cụt: *"tổng tổng tổng tổng tổng... "*
- **👉 Kết luận**: Phải thả lỏng Temperature chuỗi `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. Khi AI sinh ảo giác (chêm vào các câu quảng cáo *"Hãy subscribe kênh lalaschool"* ở cuối luồng im lặng), hướng đi thông minh nhất là giải quyết bằng Code Python (Regex Cắt đuôi).

---

## 2. Công Thức "Golden Config" Duy Nhất Định Hình Production

Từ tất cả các dữ kiện lâm sàng trên, đây là bộ xương sống bất di bất dịch cho lõi AI (Áp dụng tại `serving/service.py`):

```python
INITIAL_PROMPT = "Cuộc họp nội bộ Viettel Netmind về xây dựng mô hình nền tảng ngôn ngữ tiết kiệm năng lượng, dự án Power Saving, hệ thống Agentic N8N."
HOTWORDS = "Netmind PowerSaving Power Saving OpenAI N8N LLM Ericsson Agentic Transformer"

golden_kwargs = {
    "language": "vi",
    "task": "transcribe",
    "beam_size": 5,                     # Đa luồng tìm kiếm tối ưu nhất
    "no_speech_threshold": 0.45,        # Threshold bỏ qua âm rác hiệu quả
    "condition_on_previous_text": False,# Ngắt lây lan bệnh ảo giác xuyên segment
    "initial_prompt": INITIAL_PROMPT,   # Chống lạc ngữ cảnh
    "hotwords": HOTWORDS,               # Bias vĩnh viễn các tự vựng của Viettel
}
```

## 3. Lời Khuyên Triển Khai Thực Tế

1. **Thay thế Engine cũ**: Các file `compare_models.py` hay Engine cũ sử dụng `Moonshine` nên được đưa vào dạng Archive. `Whisper Large v3 Turbo` đã cân bằng quá xuất sắc giữa tốc độ và độ chính xác phần cứng (VRAM & CPU).
2. **Khâu Hậu Xử Lý (Post-Processing Box)**: Lỗi duy nhất còn sót lại của cỗ máy là thường thêm bớt mấy câu chèo kéo (*"Cảm ơn đã theo dõi", "Subscribe kênh"*). Sử dụng filter chặn String (như hàm `post_process_text()` trong `run_golden_whisper.py`) là phương án dứt điểm, nhanh gọn và nhẹ server nhất.

> **Trạng thái:** Toàn bộ công thức Golden Config này đã được niêm phong vào file executable độc lập: `serving/run_golden_whisper.py`. Có thể dùng nó làm tham chiếu gốc để thay lõi toàn cục cho file Production (`serving/service.py`) bất cứ khi nào sẵn sàng.
