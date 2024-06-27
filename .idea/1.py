import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 创建数据集
data = {
    '姓名': ['A001', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008', 'A009', 'A010', 'A013', 'A014', 'A015', 'A016'],
    '神经质（N）': [33, 28, 18, 24, 26, 23, 31, 25, 30, 29, 21, 30, 25, 12],
    '外向性（E）': [45, 30, 47, 58, 46, 41, 39, 44, 41, 27, 42, 41, 29, 58],
    '开放性（O）': [42, 34, 27, 45, 40, 34, 37, 41, 39, 32, 44, 36, 35, 49],
    '宜人性（A）': [46, 44, 47, 60, 38, 42, 43, 44, 44, 44, 51, 34, 47, 55],
    '严谨性（C）': [47, 29, 48, 58, 47, 50, 44, 48, 46, 29, 56, 38, 49, 55]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 去除缺失值
df.dropna(inplace=True)

# 去除姓名列
X = df.drop('姓名', axis=1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K均值聚类分析
k = 3  # 假设选择聚类数为3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 显示聚类结果
print(df[['姓名', 'Cluster']])