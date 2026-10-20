import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_housing_prices(filepath="data/HousingPricesData.csv"):
    """Trực quan hóa dữ liệu giá nhà"""
    houseprices_data = pd.read_csv(filepath)
    
    houseprices_data = houseprices_data.rename(columns={
        'Price': 'Price',
        'SqFt': 'Area',
        'Neighborhood': 'Zip'
    })
    
    houseprices_data['PriceperSqm'] = houseprices_data['Price'] / houseprices_data['Area']
    houseprices_sorted = houseprices_data.sort_values('Price', ascending=False).head(10)
    
    # 1. Tự động tạo thư mục 'reports' nếu chưa có để tránh lỗi FileNotFoundError
    os.makedirs('reports', exist_ok=True)
    
    # Vẽ biểu đồ Seaborn
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.barplot(data=houseprices_sorted, x='Zip', y='Price', ax=ax[0], palette='viridis', hue='Zip', legend=False)
    ax[0].set_title('Top 10 Areas with Highest House Prices')
    ax[0].set_xlabel('Neighborhood')
    ax[0].set_ylabel('Price')
    
    sns.barplot(data=houseprices_sorted, x='Zip', y='PriceperSqm', ax=ax[1], palette='magma', hue='Zip', legend=False)
    ax[1].set_title('Top 10 Areas with Highest Price per SqFt')
    ax[1].set_xlabel('Neighborhood')
    ax[1].set_ylabel('Price per SqFt')
    
    plt.tight_layout()
    
    # Lưu file ảnh
    plt.savefig('../reports/housing_prices_analysis.png')
    plt.close()
    print("Đã vẽ và lưu biểu đồ giá nhà thành công vào thư mục reports/")

def plot_univariate_penguin(filepath="data/penguins_size.csv"):
    """Phân tích đơn biến trên tập dữ liệu chim cánh cụt"""
    penguins_data = pd.read_csv(filepath)
    
    # 1. Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(data=penguins_data, x="culmen_length_mm", kde=True)
    plt.title('Univariate Analysis of Culmen Length')
    plt.savefig('reports/penguin_culmen_hist.png')
    plt.close()
    
    # 2. Countplot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=penguins_data, x='species')
    plt.title('Univariate Analysis of Penguin Species')
    plt.savefig('reports/penguin_species_count.png')
    plt.close()

def plot_bivariate_penguin(filepath="data/penguins_size.csv"):
    """Phân tích hai biến trên dữ liệu chim cánh cụt"""
    penguins_data = pd.read_csv(filepath)
    
    # 1. Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=penguins_data, x="culmen_length_mm", y="body_mass_g", hue="species")
    plt.title('Bivariate Analysis: Culmen Length vs Body Mass')
    plt.savefig('reports/penguin_bivariate_scatter.png')
    plt.close()
    
    # 2. Pairplot
    pair_fig = sns.pairplot(data=penguins_data[['species', 'culmen_length_mm', 'body_mass_g']], hue='species')
    pair_fig.savefig('reports/penguin_pairplot.png')
    plt.close()