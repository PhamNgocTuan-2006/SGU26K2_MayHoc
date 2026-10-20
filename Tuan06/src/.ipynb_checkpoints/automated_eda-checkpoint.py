import pandas as pd

# Bẫy lỗi cho sweetviz
try:
    import sweetviz as sv
except ImportError:
    sv = None

def run_ydata_profiling(data_path="data/marketing_campaign.csv", output_path="reports/profile_output.html"):
    print("Đã bỏ qua ydata-profiling.")

def run_sweetviz(data_path="data/marketing_campaign.csv", output_path="reports/sweetviz_report.html"):
    if sv is None:
        print("Thư viện sweetviz chưa được cài đặt.")
        return
    df = pd.read_csv(data_path, sep=';') 
    report = sv.analyze(df)
    report.show_html(output_path)
    print(f"SweetViz report saved to {output_path}")

def run_autoviz(data_path="data/marketing_campaign.csv", output_dir="reports/autoviz_report"):
    print("AutoViz đã được vô hiệu hóa do lỗi tương thích phiên bản xgboost trên máy.")