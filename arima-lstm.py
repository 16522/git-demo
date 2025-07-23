import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
output_dir = "D:\\毕设\\output_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 数据预处理函数
def preprocess_data(df):
    if 'Date' not in df.columns and 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'price'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


# 准确率计算函数
def calculate_accuracy(y_true, y_pred):
    relative_errors = np.abs(y_true - y_pred) / y_true
    accuracy_1p = np.mean(relative_errors <= 0.01) * 100
    accuracy_3p = np.mean(relative_errors <= 0.03) * 100
    accuracy_5p = np.mean(relative_errors <= 0.05) * 100
    return accuracy_1p, accuracy_3p, accuracy_5p


# 创建LSTM数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)


# 读取数据
train_path = r"D:\毕设\sj1.xlsx"
test_path = r"D:\毕设\sj2.xlsx"
df_train = preprocess_data(pd.read_excel(train_path))
df_test = preprocess_data(pd.read_excel(test_path))

# 1. ARIMA部分
# 寻找最佳SARIMA模型
print("训练SARIMA模型...")
best_model = SARIMAX(df_train['price'],
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 126),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)

# 获取ARIMA预测和残差
arima_train_pred = best_model.get_prediction(start=0)
arima_train_mean = arima_train_pred.predicted_mean
arima_residuals = df_train['price'] - arima_train_mean

# 对测试集进行ARIMA预测
arima_test_pred = best_model.get_forecast(steps=len(df_test))
arima_test_mean = arima_test_pred.predicted_mean

# 2. LSTM部分
# 准备LSTM的训练数据（使用ARIMA残差）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_residuals = scaler.fit_transform(arima_residuals.values.reshape(-1, 1))

# 创建LSTM数据集
time_step = 20
X_lstm, y_lstm = create_dataset(scaled_residuals, time_step)


# 构建LSTM模型
lstm_model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练LSTM模型
print("训练LSTM模型...")
lstm_model.fit(X_lstm, y_lstm,
               epochs=100,
               batch_size=32,
               validation_split=0.1,
               callbacks=[early_stopping],
               verbose=1)

# 3. 混合预测
# 准备测试集的LSTM输入
last_sequence = scaled_residuals[-time_step:]
lstm_test_pred = []

for _ in range(len(df_test)):
    # 预测下一个残差值
    next_pred = lstm_model.predict(last_sequence.reshape(1, time_step, 1))
    lstm_test_pred.append(next_pred[0, 0])
    # 更新序列
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# 将LSTM预测的残差转换回原始比例
lstm_test_pred = scaler.inverse_transform(np.array(lstm_test_pred).reshape(-1, 1))

# 组合ARIMA和LSTM的预测结果
final_predictions = arima_test_mean + lstm_test_pred.flatten()

# 计算准确率
acc_1, acc_3, acc_5 = calculate_accuracy(df_test['price'].values, final_predictions)

print("\n混合模型预测准确率：")
print(f"1%误差内准确率: {acc_1:.2f}%")
print(f"3%误差内准确率: {acc_3:.2f}%")
print(f"5%误差内准确率: {acc_5:.2f}%")

# 可视化结果
plt.figure(figsize=(15, 7))
plt.plot(df_train.index, df_train['price'], label='训练数据', color='blue')
plt.plot(df_test.index, df_test['price'], label='真实值', color='green')
plt.plot(df_test.index, final_predictions, label='混合模型预测', color='red')
plt.plot(df_test.index, arima_test_mean, label='ARIMA预测', color='orange', linestyle='--')
plt.title('SARIMA-LSTM混合模型预测结果')
plt.xlabel('日期')
plt.ylabel('股价')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "hybrid_model_predictions.png"))
plt.show()

# 绘制准确率柱状图
plt.figure(figsize=(10, 6))
accuracies = [acc_1, acc_3, acc_5]
labels = ['1%误差内', '3%误差内', '5%误差内']
plt.bar(labels, accuracies, color=['lightblue', 'lightgreen', 'lightpink'])
plt.title('混合模型预测准确率')
plt.ylabel('准确率 (%)')
plt.grid(True, axis='y')
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.savefig(os.path.join(output_dir, "hybrid_model_accuracy.png"))
plt.show()

# 保存预测结果
results_df = pd.DataFrame({
    '日期': df_test.index,
    '真实值': df_test['price'].values,
    'ARIMA预测': arima_test_mean.values,
    'LSTM残差预测': lstm_test_pred.flatten(),
    '混合模型预测': final_predictions,
    '绝对误差': np.abs(df_test['price'].values - final_predictions),
    '相对误差(%)': np.abs(df_test['price'].values - final_predictions) / df_test['price'].values * 100
})

results_df.to_excel(os.path.join(output_dir, "hybrid_model_results.xlsx"))

print(f"\n所有结果已保存至: {output_dir}")
