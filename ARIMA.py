import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from scipy.stats import chi2
import warnings
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
output_dir = "D:\\毕设\\output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取训练数据和真实数据
train_path = r"D:\毕设\sj1.xlsx"
test_path = r"D:\毕设\sj2.xlsx"
df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)


# 数据预处理函数
def preprocess_data(df, is_train=True):
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'price'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


# 预处理数据
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test, is_train=False)


# ADF检验函数
def adf_test(series):
    result = adfuller(series.dropna())
    print('ADF检验统计量: %f' % result[0])
    print('p-值: %f' % result[1])
    print('临界值:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print("结论：序列是平稳的(拒绝原假设)")
        return True
    else:
        print("结论：序列是非平稳的(接受原假设)")
        return False


# 计算准确率指标函数
def calculate_accuracy_metrics(y_true, y_pred):
    # 计算基本误差指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 计算平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 计算方向准确率 (DA)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    direction_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100

    # 计算预测准确率（在不同误差容限下）
    def accuracy_within_threshold(y_true, y_pred, threshold):
        return np.mean(np.abs((y_true - y_pred) / y_true) <= threshold) * 100

    accuracy_1p = accuracy_within_threshold(y_true, y_pred, 0.01)  # 1%误差内
    accuracy_3p = accuracy_within_threshold(y_true, y_pred, 0.03)  # 3%误差内
    accuracy_5p = accuracy_within_threshold(y_true, y_pred, 0.05)  # 5%误差内

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE(%)': mape,
        'Direction_Accuracy(%)': direction_accuracy,
        'Accuracy_within_1%(%)': accuracy_1p,
        'Accuracy_within_3%(%)': accuracy_3p,
        'Accuracy_within_5%(%)': accuracy_5p
    }


# 检查平稳性并进行差分
print("原始序列的ADF检验结果：")
is_stationary = adf_test(df_train['price'])

diff_order = 0
diff_series = df_train['price'].copy()
while not is_stationary and diff_order < 2:
    diff_order += 1
    diff_series = diff_series.diff().dropna()
    print(f"\n{diff_order}阶差分后的ADF检验结果：")
    is_stationary = adf_test(diff_series)

print(f"\n序列在{diff_order}阶差分后是平稳的")

# 尝试不同的SARIMA模型组合
print("开始尝试不同的SARIMA模型组合...")

best_aic = float('inf')
best_order = None
best_seasonal_order = None
best_model = None

# 定义参数范围
p_range = range(0, 1)
d_range = [diff_order]
q_range = range(0, 1)
P_range = range(0, 2)
D_range = [0, 1]
Q_range = range(0, 2)
s = 126  # 季节周期设为6个月（约126个交易日）

for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        try:
                            seasonal_order = (P, D, Q, s)
                            print(f"尝试SARIMA({p},{d},{q})x{seasonal_order}模型")

                            model = SARIMAX(df_train['price'],
                                            order=(p, d, q),
                                            seasonal_order=seasonal_order,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                            results = model.fit(disp=False)

                            # 测试模型是否能预测
                            test_forecast = results.get_forecast(steps=5)
                            test_mean = test_forecast.predicted_mean

                            if not np.isnan(test_mean.values).any():
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = seasonal_order
                                    best_model = results
                                    print(f"  当前最佳模型更新为: SARIMA{best_order}x{best_seasonal_order}")
                                    print(f"  AIC: {best_aic}")
                        except:
                            continue

if best_model is None:
    raise Exception("无法拟合任何模型，请检查数据")

print(f"\n最佳SARIMA模型参数: {best_order}x{best_seasonal_order}")
print(f"AIC值: {best_aic}")

# 进行预测
forecast_steps = len(df_test)  # 使用实际测试集的长度
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# 创建预测日期索引
forecast_dates = df_test.index

# 计算评估指标
metrics = calculate_accuracy_metrics(df_test['price'].values, forecast_mean.values)

# 打印评估结果
print("\n模型评估指标:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# 在计算评估指标后，单独提取并显示MAE和MSE
mae = metrics['MAE']
mse = metrics['MSE']

print("\n关键误差指标:")
print("-" * 50)
print(f"平均绝对误差(MAE): {mae:.2f}")
print(f"均方误差(MSE): {mse:.2f}")
print("-" * 50)

# 绘制原始数据、预测结果和真实值的对比图
plt.figure(figsize=(16, 8))
plt.plot(df_train['price'], label='历史数据', color='blue')
plt.plot(forecast_dates, forecast_mean, color='red', label='预测值')
plt.plot(df_test['price'], color='green', label='真实值')
plt.fill_between(forecast_dates,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.3, label='95%置信区间')
plt.title('茅台股价SARIMA预测结果对比 (2024)')
plt.xlabel('日期')
plt.ylabel('股价')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_预测结果对比.png"), dpi=300, bbox_inches='tight')
plt.show()

# 创建对比数据框
comparison_df = pd.DataFrame({
    '预测值': forecast_mean.values,
    '真实值': df_test['price'].values,
    '绝对误差': abs(forecast_mean.values - df_test['price'].values),
    '相对误差(%)': abs(forecast_mean.values - df_test['price'].values) / df_test['price'].values * 100,
    '95%置信区间下限': forecast_ci.iloc[:, 0].values,
    '95%置信区间上限': forecast_ci.iloc[:, 1].values
}, index=forecast_dates)

# 添加月份列
comparison_df['月份'] = comparison_df.index.strftime('%Y-%m')

# 按月显示对比结果的统计信息
monthly_comparison = comparison_df.groupby('月份').agg({
    '预测值': ['mean', 'min', 'max'],
    '真实值': ['mean', 'min', 'max'],
    '绝对误差': 'mean',
    '相对误差(%)': 'mean'
}).round(2)

print("\n按月份汇总的对比结果:")
print(monthly_comparison)

# 将对比结果保存到Excel文件
with pd.ExcelWriter(os.path.join(output_dir, "预测结果对比.xlsx")) as writer:
    comparison_df.to_excel(writer, sheet_name='每日对比')
    monthly_comparison.to_excel(writer, sheet_name='月度汇总')

    # 添加详细的评估指标sheet
    metrics_df = pd.DataFrame({
        '指标': list(metrics.keys()),
        '值': list(metrics.values())
    })
    metrics_df.to_excel(writer, sheet_name='评估指标')

# 绘制预测误差分布图
plt.figure(figsize=(10, 6))
plt.hist(comparison_df['绝对误差'], bins=30, edgecolor='black')
plt.title('预测误差分布图')
plt.xlabel('绝对误差')
plt.ylabel('频数')
plt.savefig(os.path.join(output_dir, "06_预测误差分布.png"), dpi=300, bbox_inches='tight')
plt.show()

# 绘制预测值vs真实值散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_test['price'], forecast_mean, alpha=0.5)
plt.plot([df_test['price'].min(), df_test['price'].max()],
         [df_test['price'].min(), df_test['price'].max()],
         'r--', label='理想预测线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测值vs真实值对比散点图')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "07_预测值vs真实值散点图.png"), dpi=300, bbox_inches='tight')
plt.show()

# 绘制准确率指标柱状图
plt.figure(figsize=(12, 6))
accuracy_metrics = {
    '1%误差内': metrics['Accuracy_within_1%(%)'],
    '3%误差内': metrics['Accuracy_within_3%(%)'],
    '5%误差内': metrics['Accuracy_within_5%(%)']
}

plt.bar(accuracy_metrics.keys(), accuracy_metrics.values(), color=['lightblue', 'lightgreen', 'lightpink'])
plt.title('SARIMA模型的预测准确率')
plt.ylabel('准确率 (%)')
plt.grid(True, axis='y')
for i, v in enumerate(accuracy_metrics.values()):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.savefig(os.path.join(output_dir, "08_预测准确率指标.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n所有图片和对比结果已保存至: {output_dir}")

# 残差分析
residuals = best_model.resid

# 绘制残差序列图
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('残差序列图')
plt.xlabel('时间')
plt.ylabel('残差值')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "09_残差序列图.png"), dpi=300, bbox_inches='tight')
plt.show()

# 绘制残差的ACF和PACF图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(residuals, ax=ax1, lags=40)
plot_pacf(residuals, ax=ax2, lags=40)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "10_残差ACF_PACF图.png"), dpi=300, bbox_inches='tight')
plt.show()


# 进行Ljung-Box检验
def ljung_box_test(residuals, lags=None):
    if lags is None:
        lags = min(len(residuals) // 5, 40)

    acf_values = acf(residuals, nlags=lags, fft=False)
    n = len(residuals)
    Q = n * (n + 2) * np.sum([(acf_values[k] ** 2) / (n - k) for k in range(1, lags + 1)])
    p_value = 1 - chi2.cdf(Q, lags)
    return Q, p_value


Q_stat, p_value = ljung_box_test(residuals)
print(f"\nLjung-Box检验结果:")
print(f"Q统计量: {Q_stat:.4f}")
print(f"P值: {p_value:.4f}")
if p_value > 0.05:
    print("结论：残差序列是白噪声（接受原假设）")
else:
    print("结论：残差序列不是白噪声（拒绝原假设）")

# 绘制残差直方图和Q-Q图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(residuals, bins=30, density=True, edgecolor='black')
ax1.set_title('残差直方图')
ax1.set_xlabel('残差值')
ax1.set_ylabel('密度')

sm.qqplot(residuals, line='s', ax=ax2)
ax2.set_title('残差Q-Q图')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "11_残差分布检验图.png"), dpi=300, bbox_inches='tight')
plt.show()
