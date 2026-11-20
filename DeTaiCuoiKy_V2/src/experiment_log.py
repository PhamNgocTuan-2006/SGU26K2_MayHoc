# src/logger.py
import getpass
import subprocess
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from .config import LOG_DIR, RANDOM_STATE
from .evaluation import cv_evaluate

COLS = [
    "Exp ID", "Batch", "Thoi gian ghi nhan", "Nguoi chay", "Random Seed", "Ten mo hinh",
    "Lop thuat toan", "Tham so (Hyperparameters)", "Chien luoc xu ly mat can bang",
    "Macro-F1 (TB)", "Macro-F1 (std)", "Balanced Acc (TB)", "MCC (TB)", "Accuracy (TB)",
    "Recall c4 Metasploit_SSH", "Recall c5 NMAP_FIN_SCAN", "Recall c11 Wipro_bulb",
    "Fit time (s)", "Predict time (s)", "So fold", "Git commit", "Ghi chu"
]

def _git_hash():
    try:
        git_process = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, 
            text=True
        )
        return git_process.strip()
    except Exception:
        return "n/a"

def _final_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model

NOISE = {"n_jobs", "verbose", "random_state", "warm_start"}
KEY = {
    "n_neighbors", "n_estimators", "max_depth", "min_samples_leaf",
    "max_features", "class_weight", "C", "weights"
}

def _params_str(model):
    # Lấy estimator cuối cùng (nếu là Pipeline thì lấy bước model chính)
    est = _final_estimator(model)
    cur = est.get_params(deep=False)

    # Lấy tham số mặc định của lớp mô hình để so sánh
    try:
        default = type(est)().get_params(deep=False)
    except Exception:
        default = {}

    # Lọc các tham số quan trọng hoặc khác giá trị mặc định, bỏ qua tham số nhiễu
    selected_params = {}
    for k, v in cur.items():
        if k in NOISE:
            continue
        is_key = k in KEY
        is_new_param = k not in default
        is_changed = (k in default) and (v != default[k])
        if is_key or is_new_param or is_changed:
            selected_params[k] = v

    # Ghép thành chuỗi hiển thị
    parts = []
    for k, v in selected_params.items():
        parts.append(f"{k}={v!r}")

    if len(parts) == 0:
        return "(mac dinh)"
    return ", ".join(parts)

class ExperimentLogger:
    def __init__(self, author=None):
        if author is None:
            self.author = getpass.getuser()
        else:
            self.author = author
            
        self.path = LOG_DIR / f"log_{self.author}.csv"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.path.exists():
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=COLS)
            
        # Mỗi lần khởi tạo Logger mới = 1 batch thí nghiệm mới
        if len(self.df) > 0:
            self.batch = int(self.df["Batch"].max()) + 1
        else:
            self.batch = 1

    def log(self, model, cv_df, name, strategy="Khong xu ly", notes=""):
        n = len(self.df) + 1
        mean = cv_df.mean(numeric_only=True)
        
        row = {
            "Exp ID": f"{self.author[:1].upper()}_{n:03d}", 
            "Batch": self.batch,
            "Thoi gian ghi nhan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nguoi chay": self.author, 
            "Random Seed": RANDOM_STATE, 
            "Ten mo hinh": name,
            "Lop thuat toan": type(_final_estimator(model)).__name__,
            "Tham so (Hyperparameters)": _params_str(model), 
            "Chien luoc xu ly mat can bang": strategy,
            "Macro-F1 (TB)": mean["macro_f1"], 
            "Macro-F1 (std)": cv_df["macro_f1"].std(),
            "Balanced Acc (TB)": mean["balanced_acc"], 
            "MCC (TB)": mean["mcc"], 
            "Accuracy (TB)": mean["accuracy"],
            "Recall c4 Metasploit_SSH": mean["recall_c4"], 
            "Recall c5 NMAP_FIN_SCAN": mean["recall_c5"],
            "Recall c11 Wipro_bulb": mean["recall_c11"],
            "Fit time (s)": mean["fit_time"], 
            "Predict time (s)": mean["pred_time"],
            "So fold": len(cv_df), 
            "Git commit": _git_hash(), 
            "Ghi chu": notes
        }
        
        # Thêm dòng mới vào DataFrame và lưu lại ngay lập tức
        new_row_df = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return row

    def run(self, model, X, y, name, strategy="Khong xu ly", notes=""):
        """Chạy Cross-Validation và ghi nhật ký trong 1 lệnh."""
        cv_df = cv_evaluate(model, X, y, name)
        self.log(model, cv_df, name, strategy, notes)
        return cv_df

def export_excel(out=None):
    """Gộp các file log_*.csv của cả nhóm thành 1 file Excel định dạng đẹp."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter as L
    
    if out is None:
        out = LOG_DIR / "experiment_log.xlsx"
        
    files = sorted(LOG_DIR.glob("log_*.csv"))
    if not files:
        raise FileNotFoundError("Chưa có file log_*.csv nào để xuất Excel.")
        
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # them ten nguoi chay vao ma thi nghiem de khong trung giua cac thanh vien
    df["Exp ID"] = df["Nguoi chay"] + "-" + df["Exp ID"]
    # bo dong trung do chay lai cung cau hinh
    khoa = ["Nguoi chay", "Ten mo hinh", "Tham so (Hyperparameters)",
            "Chien luoc xu ly mat can bang", "Random Seed"]
    df = df.drop_duplicates(subset=khoa, keep="last")
    df = df.sort_values(["Thoi gian ghi nhan", "Exp ID"]).reset_index(drop=True)
    
    df.to_excel(out, sheet_name="Experiment Log", index=False)
    
    wb = load_workbook(out)
    ws = wb["Experiment Log"]
    
    header_fill = PatternFill("solid", fgColor="1F3864")
    tints = ["DCE6F1", "FDE9D9", "EBF1DE", "E4DFEC", "FFF2CC"]
    batch_col_idx = COLS.index("Batch") + 1
    
    # Định dạng dòng tiêu đề (Header)
    for cell in ws[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
    # Định dạng các dòng dữ liệu
    for r in range(2, ws.max_row + 1):
        batch_val = ws.cell(r, batch_col_idx).value
        batch_num = int(batch_val) if batch_val is not None else 0
        row_fill = PatternFill("solid", fgColor=tints[batch_num % len(tints)])
        
        for j in range(1, len(COLS) + 1):
            cell = ws.cell(r, j)
            cell.fill = row_fill
            cell.font = Font(name="Arial", size=9)
            
            # Căn lề trái cho cột tham số và ghi chú, còn lại căn giữa
            if j in (8, 22):
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                
            # Định dạng số thập phân cho các cột metric và thời gian
            if 10 <= j <= 17:
                cell.number_format = "0.0000"
            if j in (18, 19):
                cell.number_format = "0.00"
                
    # Điều chỉnh độ rộng cột
    for j, c_name in enumerate(COLS, 1):
        col_letter = L(j)
        if j == 8:
            ws.column_dimensions[col_letter].width = 60
        elif j == 22:
            ws.column_dimensions[col_letter].width = 28
        else:
            ws.column_dimensions[col_letter].width = 16
            
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = ws.dimensions
    
    wb.save(out)
    return out