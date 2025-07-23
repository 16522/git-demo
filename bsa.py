import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 读取Excel数据（先处理缺失值）
df = pd.read_excel(r'D:\毕设\sj1.xlsx', parse_dates=['Date'], index_col='Date').dropna()

# 计算一阶差分（自动对齐索引）
df['First_Diff'] = df['price'].diff()

#ADF检验函数
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# 对原始序列和差分序列进行ADF检验
print("原始序列ADF检验结果：")
adf_test(df['price'])

print("\n一阶差分序列ADF检验结果：")
adf_test(df['First_Diff'].dropna())  # 差分后第一个值为NaN，需要删除

# 可视化（修正索引对齐方式）
plt.figure(figsize=(12,6))

# 原始价格序列
plt.subplot(2,1,1)
plt.plot(df.index, df['price'], label='Original Price')
plt.title('Original Price Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# 一阶差分序列（使用修正后的索引）
plt.subplot(2,1,2)
plt.plot(df.index[1:], df['First_Diff'].dropna(),  # 使用修正后的索引和差分序列
         color='orange', label='First Difference')
plt.title('First Difference of Price')
plt.xlabel('Date')
plt.ylabel('Difference')
plt.legend()

plt.tight_layout()
plt.show()
