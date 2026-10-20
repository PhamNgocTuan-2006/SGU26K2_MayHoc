import numpy as np
import pandas as pd
from scipy import stats

def analyze_covid_data(filepath="data/covid-data.csv"):
    """Thực hiện nhiệm vụ 1 - Bài toán 1 (Nhiệm vụ COVID)"""
    covid_data = pd.read_csv(filepath)
    covid_data = covid_data[['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases']]
    
    # Xử lý missing value cơ bản trước khi tính toán
    new_cases_clean = covid_data["new_cases"].dropna()
    
    results = {
        "mean": np.mean(new_cases_clean),
        "median": np.median(new_cases_clean),
        "mode": stats.mode(new_cases_clean, keepdims=True).mode[0],
        "variance": np.var(new_cases_clean),
        "std_dev": np.std(new_cases_clean),
        "max": np.max(new_cases_clean),
        "min": np.min(new_cases_clean),
        "percentile_60": np.percentile(new_cases_clean, 60),
        "quartile_75": np.quantile(new_cases_clean, 0.75),
        "iqr": stats.iqr(new_cases_clean)
    }
    return results

def process_marketing_data(filepath="data/marketing_campaign.csv"):
    """Thực hiện nhiệm vụ 2 - Xử lý dữ liệu Marketing Campaign"""
    marketing_data = pd.read_csv(filepath)
    marketing_data = marketing_data[['ID', 'Year_Birth', 'Education', 'Marital_Status', 
                                     'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 
                                     'Recency', 'NumStorePurchases', 'NumWebVisitsMonth']]
    
    # 1. Loại bỏ trùng lặp
    marketing_data = marketing_data.drop_duplicates()
    
    # 2. Thay thế dữ liệu
    marketing_data['Teenhome_replaced'] = marketing_data['Teenhome'].replace([0, 1, 2], ['has no teen', 'has teen', 'has teen'])
    
    # 3. Điền giá trị thiếu và đổi kiểu dữ liệu
    marketing_data['Income'] = marketing_data['Income'].fillna(0)
    marketing_data['Income_changed'] = marketing_data['Income'].astype(int)
    
    return marketing_data

def analyze_dataset_summary(df):
    """Tính toán nhanh thống kê mô tả cho dữ liệu bất kỳ (Rượu đỏ / Tiểu đường)"""
    return df.describe().T