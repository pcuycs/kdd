import os
import sys
import json
import shutil
import subprocess

# 1. Tự động kiểm tra và cài đặt thư viện kaggle (chưa import vì import sẽ tự xác thực)
try:
    import importlib.metadata as md
    md.version("kaggle")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

# 2. Cấu hình API key
# Cách 1 (khuyên dùng): đặt đường dẫn tới file kaggle.json tải từ
#   https://www.kaggle.com/settings -> API -> "Create New Token"
# Cách 2: điền trực tiếp username/key bên dưới (KHÔNG commit lên git)
# Cách 3: đặt biến môi trường KAGGLE_USERNAME và KAGGLE_KEY
path_to_downloaded_json = os.path.join(os.path.expanduser("~"), "Downloads", "kaggle.json")
KAGGLE_USERNAME = ""   # ví dụ: "uyphan"
KAGGLE_KEY = ""        # ví dụ: "xxxxxxxxxxxxxxxxxxxxxxxx"

kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
dest_json_path = os.path.join(kaggle_dir, "kaggle.json")
os.makedirs(kaggle_dir, exist_ok=True)

access_token_path = os.path.join(kaggle_dir, "access_token")

if KAGGLE_KEY.startswith("KGAT_"):
    # Token kiểu mới (KGAT_...) phải lưu vào ~/.kaggle/access_token, không phải kaggle.json
    with open(access_token_path, "w") as f:
        f.write(KAGGLE_KEY)
    if os.path.exists(dest_json_path):
        os.remove(dest_json_path)  # tránh dùng nhầm kaggle.json cũ chứa token sai định dạng
    print(f"-> Đã lưu access token vào {access_token_path}")
elif os.environ.get("KAGGLE_API_TOKEN") or os.path.exists(access_token_path):
    print("-> Dùng access token có sẵn.")
elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
    print("-> Dùng KAGGLE_USERNAME/KAGGLE_KEY từ biến môi trường.")
elif os.path.exists(dest_json_path):
    print(f"-> Dùng token có sẵn tại {dest_json_path}")
elif os.path.exists(path_to_downloaded_json):
    shutil.copy(path_to_downloaded_json, dest_json_path)
    print(f"-> Đã copy kaggle.json từ {path_to_downloaded_json}")
elif KAGGLE_USERNAME and KAGGLE_KEY:
    with open(dest_json_path, "w") as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
    print(f"-> Đã tạo {dest_json_path} từ username/key khai báo trong code.")
else:
    sys.exit(
        "Không tìm thấy Kaggle API key. Hãy tải kaggle.json tại "
        "https://www.kaggle.com/settings (API -> Create New Token), rồi đặt vào "
        f"{path_to_downloaded_json} hoặc {dest_json_path}, hoặc điền KAGGLE_USERNAME/KAGGLE_KEY."
    )

if os.name != "nt" and os.path.exists(dest_json_path):
    os.chmod(dest_json_path, 0o600)

# Import sau khi đã cấu hình xong token
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("-> Xác thực Kaggle thành công!")

# 3. Tải bộ dữ liệu Home Credit Default Risk
competition_name = "home-credit-default-risk"
output_folder = "./home_credit_data"

print(f"Đang tiến hành tải dữ liệu từ cuộc thi '{competition_name}'...")

# Sử dụng hàm competition_download_files cho các bộ dữ liệu cuộc thi
api.competition_download_files(competition_name, path=output_folder, quiet=False)

print("Tải hoàn tất! Đang tiến hành giải nén...")

# Tiến hành giải nén file zip vừa tải về
import zipfile
zip_file_path = os.path.join(output_folder, f"{competition_name}.zip")

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    os.remove(zip_file_path)  # Xóa file zip gốc sau khi giải nén để tiết kiệm bộ nhớ
    print(f"-> Dữ liệu đã được giải nén thành công tại thư mục: {output_folder}")
else:
  print("Không tìm thấy file zip dữ liệu. Vui lòng kiểm tra lại quyền truy cập cuộc thi.")