# 📈 HOSE V4 - HỆ THỐNG ĐỊNH LƯỢNG & KIỂM ĐỊNH CHIẾN LƯỢC KẾT HỢP (STREAMLIT WEB APP)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Quantitative Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-orange.svg)](#)

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit**, chuyển thể toàn diện từ nghiên cứu định lượng **HOSE V4 - Generic Nested Walk-Forward FA/TA + Top N + EW / MPT / MinVar / HRP**. 

Hệ thống cho phép nhà đầu tư và chuyên gia tài chính định lượng kiểm định (backtest), tối ưu hóa và đánh giá tính hiệu quả thực chiến của chiến lược đầu tư cổ phiếu trên thị trường chứng khoán Việt Nam (HOSE / VN-INDEX) với tiêu chuẩn khắt khe chống rò rỉ dữ liệu (**Lookahead-Free** & **Anti-Data Leakage**).

---

## 🌟 TÍNH NĂNG NỔI BẬT

### 1. Quy Trình Nested Walk-Forward Chống Overfitting
- Phân chia chuỗi thời gian độc lập: **Core Train** (Tối ưu ngưỡng tín hiệu) $\rightarrow$ **Validation** (Khảo sát các mô hình phân bổ danh mục và khóa Champion) $\rightarrow$ **Final Test** (Kiểm định thực chiến ngoài mẫu hoàn toàn độc lập).
- Tái lựa chọn Top N cổ phiếu theo chu kỳ (Hàng Quý hoặc Hàng Tháng), chỉ sử dụng dữ liệu kết thúc tại phiên $t-1$ (As-Of Date).

### 2. Hệ Thống Chấm Điểm Kỹ Thuật TA Score (0 - 10)
Được cấu trúc theo 4 trụ cột định lượng chuẩn xác:
- **Xu hướng (Trend Score: 0 - 5 điểm):** Giá đóng cửa so với EMA50, đường xu hướng trung hạn EMA50 so với EMA200, tín hiệu MACD so với MACD Signal.
- **Động lượng (Momentum Score: 0 - 3 điểm):** RSI Wilder 14 phiên nằm trong vùng sức mạnh [45, 75], ROC 10 phiên dương, vị thế giá tiệm cận đỉnh 52 tuần.
- **Khối lượng (Volume Score: 0 - 1 điểm):** Tỷ lệ thanh khoản so với MA20 khối lượng hoặc dòng tiền tích lũy OBV vượt EMA20 của OBV.
- **Biến động (Volatility Score: 0 - 1 điểm):** Biên độ dao động thực tế ATR% không vượt quá ngưỡng phân vị 90% quá khứ.
- **Cơ chế trạng thái Hysteresis:** Chuyển trạng thái **BUY** khi $TA\_Score \ge BuyThreshold$, giữ lệnh cho đến khi $TA\_Score \le ExitThreshold$ mới kích hoạt **EXIT**.

### 3. Đa Dạng Hóa Thuật Toán Phân Bổ Tỷ Trọng (Portfolio Allocation)
- **EW (Equal Weight):** Tỷ trọng đều $1/N$ làm mốc chuẩn đối sánh.
- **MPT-LW (Modern Portfolio Theory - Max Sharpe):** Tối ưu hóa tỷ suất sinh lời điều chỉnh theo rủi ro kết hợp **Ledoit-Wolf Covariance Shrinkage** và **Mean Return Shrinkage** giúp ổn định ma trận nghịch đảo.
- **MinVar-LW (Minimum Variance):** Tìm danh mục phương sai tối thiểu nhằm kiểm soát triệt để độ biến động.
- **HRP (Hierarchical Risk Parity):** Phân bổ bình đẳng rủi ro theo cấu trúc phân cụm thứ bậc (Machine Learning / Graph Clustering), không cần nghịch đảo ma trận hiệp phương sai.
- **BLEND50:** Kết hợp 50% MPT và 50% EW để cân bằng giữa tối ưu hóa và chống bất định mô hình.

### 4. Lớp Bảo Vệ Rủi Ro Kép (Dual Risk Overlays)
- **Market Regime Filter:** Theo dõi chỉ số thị trường (VN-INDEX) so với đường EMA200 ngày và độ dốc (Slope 20 phiên). Khi thị trường rơi vào xu hướng giảm (Downtrend / Risk-Off), hệ thống tự động đưa **100% tài sản về Tiền mặt (Cash)**.
- **Stock Active Mask:** Quản trị trạng thái từng cổ phiếu. Cổ phiếu nào rơi vào tín hiệu Exit sẽ được giải phóng tỷ trọng về Tiền mặt, **tuyệt đối không tái phân bổ sang các mã khác** để tránh bẫy rủi ro khi thị trường suy yếu.

### 5. Động Cơ Khớp Lệnh Chân Thực (Event-Driven Portfolio Engine)
- Tín hiệu sinh ra từ giá đóng cửa phiên $t$ (Close), lệnh thực thi tại giá mở cửa phiên $t+1$ (Open).
- Tính toán đầy đủ chi phí giao dịch (Commission = 0.15%) và trượt giá thực tế (Slippage = 0.05%).
- Hỗ trợ cổ phiếu lô chẵn theo quy định giao dịch thực tế.

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── app.py                     # Mã nguồn chính của Streamlit Web App
├── requirements.txt           # Danh sách thư viện Python cần thiết
├── README.md                  # Hướng dẫn chi tiết triển khai và sử dụng
├── HOSE - 01.01.2021 - ...    # File dữ liệu mẫu Excel OHLCV (tùy chọn)
└── .gitignore                 # Các tệp bỏ qua khi đẩy lên Git (khuyên dùng)
```

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY TẠI MÁY CÁ NHÂN (LOCAL)

### Yêu cầu hệ thống:
- Đã cài đặt **Python 3.9** trở lên (Khuyên dùng Python 3.10 hoặc 3.11).
- Trình quản lý gói `pip`.

### Các bước thực hiện:

1. **Mở Terminal / PowerShell tại thư mục dự án:**
   ```bash
   cd "duong_dan_den_thu_muc_chua_code"
   ```

2. **(Khuyên dùng) Tạo môi trường ảo:**
   ```bash
   # Trên Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Trên macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy Web App Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Ứng dụng sẽ tự động mở trên trình duyệt mặc định tại địa chỉ: `http://localhost:8501`.

---

## 🌐 HƯỚNG DẪN ĐƯA LÊN GITHUB & DEPLOY MIỄN PHÍ TRÊN STREAMLIT CLOUD

### Bước 1: Khởi tạo Git và tải mã nguồn lên GitHub
1. Truy cập [GitHub.com](https://github.com) và tạo một **New Repository** (ví dụ đặt tên: `hose-v4-quant-app`).
2. Mở Terminal / PowerShell tại thư mục máy bạn và chạy các lệnh:
   ```bash
   git init
   git add app.py requirements.txt README.md
   # Nếu muốn tải kèm file dữ liệu mẫu (dưới 100MB):
   git add *.xlsx
   git commit -m "Initial commit: Deploy HOSE V4 Streamlit Web App"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-github-cua-ban>/hose-v4-quant-app.git
   git push -u origin main
   ```

### Bước 2: Triển khai (Deploy) trên Streamlit Community Cloud
1. Truy cập trang web: [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm vào nút **"New app"** ở góc phải màn hình.
3. Điền các thông tin:
   - **Repository:** Chọn `tai-khoan-github-cua-ban/hose-v4-quant-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Bấm nút **"Deploy!"**.
5. Đợi hệ thống tự động cài đặt các gói trong `requirements.txt` trong khoảng 1 - 2 phút. Sau khi hoàn tất, bạn sẽ nhận được đường link Web App công khai (ví dụ: `https://hose-v4-quant-app.streamlit.app`) có thể chia sẻ cho bất kỳ ai truy cập từ máy tính hoặc điện thoại!

---

## 📊 HƯỚNG DẪN CHUẨN BỊ DỮ LIỆU ĐẦU VÀO

Hệ thống có bộ nạp thông minh tự động nhận diện cả file **Excel (.xlsx, .xls)** lẫn **CSV (.csv)** với tên cột tiếng Việt hoặc tiếng Anh:

| Tên Cột Tiếng Việt | Tên Cột Tiếng Anh | Ý Nghĩa | Bắt Buộc? |
| :--- | :--- | :--- | :---: |
| **Mã** hoặc **Mã CK** | `Ticker`, `Symbol`, `Code` | Mã cổ phiếu (VNM, FPT, HPG, VNINDEX...) | **Có** |
| **Ngày** | `Date`, `Datetime`, `Trading_Date` | Ngày giao dịch (định dạng YYYY-MM-DD hoặc MM/DD/YYYY) | **Có** |
| **Giá mở cửa** | `Open`, `Open_Price` | Giá mở cửa của phiên | **Có** |
| **Giá cao nhất** | `High`, `High_Price` | Giá cao nhất trong phiên | **Có** |
| **Giá thấp nhất** | `Low`, `Low_Price` | Giá thấp nhất trong phiên | **Có** |
| **Giá đóng cửa** | `Close`, `Close_Price` | Giá đóng cửa thực tế | **Có** |
| **Giá điều chỉnh** | `adj_close`, `Adjusted_Close` | Giá sau khi điều chỉnh cổ tức / chia tách | Khuyên dùng |
| **Khối lượng khớp lệnh** | `Volume`, `Vol` | Khối lượng khớp lệnh thực tế trong phiên | **Có** |

*Ghi chú: Nếu file dữ liệu không có chỉ số `VNINDEX`, hệ thống sẽ tự động tổng hợp một Synthetic Market Proxy từ danh mục để làm mốc Benchmark.*

---

## 🛡️ NGUYÊN TẮC KIỂM TOÁN CHỐNG RÒ RỈ THÔNG TIN (LEAKAGE AUDIT)

Để đảm bảo kết quả kiểm định có giá trị thực tiễn cao nhất khi áp dụng vào tài khoản thật, hệ thống tuân thủ chặt chẽ 6 nguyên tắc:
1. **Khóa tham số tín hiệu:** Cặp ngưỡng BUY/EXIT chỉ được tìm kiếm trên giai đoạn **Core Train**.
2. **Khóa mô hình Champion:** Việc so sánh 5 mô hình phân bổ tỷ trọng chỉ diễn ra trên tập **Validation**, mô hình đạt điểm cao nhất (Champion) được khóa cố định.
3. **Không nhìn trước tương lai tại điểm tái cơ cấu:** Mỗi kỳ chọn cổ phiếu Top N chỉ được nhìn giá và thanh khoản kết thúc tại phiên $t-1$.
4. **Không look-ahead trong ma trận trọng số:** Ma trận hiệp phương sai Ledoit-Wolf và tỷ suất kỳ vọng chỉ tính trên cửa sổ quá khứ $\le t-1$.
5. **Độ trễ khớp lệnh 1 phiên:** Tín hiệu đóng nến ngày $t$ chỉ được phép khớp ở giá mở cửa ngày $t+1$.
6. **Tập Final Test thuần túy OOS:** Không bao giờ tinh chỉnh lại tham số dựa trên kết quả của tập Final Test.

---

## 📜 GIẤY PHÉP (LICENSE)
Dự án được phát hành theo giấy phép [MIT License](LICENSE). Tự do sử dụng, chỉnh sửa và ứng dụng vào mục đích cá nhân cũng như thương mại.
