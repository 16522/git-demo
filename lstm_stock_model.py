import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import math
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 计算技术指标
def calculate_technical_indicators(close_prices):
    # 计算移动平均线
    def calculate_ma(data, window):
        return pd.Series(data).rolling(window=window).mean().fillna(0).values

    # 计算MACD
    def calculate_macd(data):
        exp1 = pd.Series(data).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(data).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd.values, signal.values, hist.values

    # 计算RSI
    def calculate_rsi(data, periods=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0).values

    # 计算布林带
    def calculate_bollinger_bands(data, window=20):
        ma = pd.Series(data).rolling(window=window).mean()
        std = pd.Series(data).rolling(window=window).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        return upper.fillna(0).values, ma.fillna(0).values, lower.fillna(0).values

    # 计算各项技术指标
    ma5 = calculate_ma(close_prices, 5)
    ma10 = calculate_ma(close_prices, 10)
    ma20 = calculate_ma(close_prices, 20)

    macd, macd_signal, macd_hist = calculate_macd(close_prices)
    rsi = calculate_rsi(close_prices)
    upper, middle, lower = calculate_bollinger_bands(close_prices)

    # 将所有指标组合成特征矩阵
    features = np.column_stack((
        close_prices,
        ma5, ma10, ma20,
        macd, macd_signal, macd_hist,
        rsi,
        upper, middle, lower
    ))

    return features


# 计算综合准确率
def calculate_comprehensive_accuracy(y_true, y_pred):
    # 1. 价格预测准确率（误差在5%以内）
    price_accuracy = np.mean(np.abs(y_true - y_pred) / y_true <= 0.05) * 100

    # 2. 方向准确率
    y_true_dir = np.sign(np.diff(y_true))
    y_pred_dir = np.sign(np.diff(y_pred))
    direction_acc = np.mean(y_true_dir == y_pred_dir) * 100

    # 3. 趋势准确率（连续3天以上的趋势判断）
    def trend_accuracy(y_true, y_pred, window=3):
        true_trends = np.sign(np.diff(y_true))
        pred_trends = np.sign(np.diff(y_pred))

        true_trend_changes = np.diff(true_trends)
        pred_trend_changes = np.diff(pred_trends)

        trend_accuracy = np.mean(true_trend_changes == pred_trend_changes) * 100
        return trend_accuracy

    trend_acc = trend_accuracy(y_true, y_pred)

    # 4. 波动性准确率（预测值与实际值的波动幅度相似度）
    def volatility_accuracy(y_true, y_pred):
        true_volatility = np.std(np.diff(y_true))
        pred_volatility = np.std(np.diff(y_pred))
        volatility_similarity = 1 - abs(true_volatility - pred_volatility) / true_volatility
        return volatility_similarity * 100

    vol_acc = volatility_accuracy(y_true, y_pred)

    # 计算综合准确率（各项指标的加权平均）
    comprehensive_acc = (price_accuracy * 0.4 +
                         direction_acc * 0.3 +
                         trend_acc * 0.2 +
                         vol_acc * 0.1)

    return {
        '综合准确率': comprehensive_acc,
        '价格预测准确率': price_accuracy,
        '方向准确率': direction_acc,
        '趋势准确率': trend_acc,
        '波动性准确率': vol_acc
    }


# 加载数据
file_path = r"D:\毕设\sj1.xlsx"
if not os.path.exists(file_path):
    print(f"错误：文件 {file_path} 不存在")
    exit()

data = pd.read_excel(file_path)
print("历史股票数据预览：")
print(data.head())

# 检查列名并进行处理
if 'Date' in data.columns and 'price' in data.columns:
    dates = data['Date'].values
    data_values = data['price'].values
else:
    dates = data.iloc[:, 0].values
    data_values = data.iloc[:, 1].values

print(f"历史数据日期范围: {dates[0]} 到 {dates[-1]}")
print(f"历史股票价格范围: {data_values.min():.4f} 到 {data_values.max():.4f}")

# 计算技术指标
features = calculate_technical_indicators(data_values)

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)


# 创建训练数据
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step)])
        Y.append(dataset[i + time_step, 0])  # 只预测收盘价
    return np.array(X), np.array(Y)


# 使用更长的时间步长
time_step = 20
X, y = create_dataset(scaled_features, time_step)

# 划分训练集和测试集
train_size = int(len(X) * 0.7)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

print(f"训练集大小: {X_train.shape[0]} 样本")
print(f"测试集大小: {X_test.shape[0]} 样本")

# 构建改进的LSTM模型
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(time_step, features.shape[1])),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    Dropout(0.3),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='huber')

# 添加回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 训练模型
print("开始训练改进后的LSTM模型...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=200,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# 加载2024年的真实数据
file_path_2024 = r"D:\毕设\sj2.xlsx"
if os.path.exists(file_path_2024):
    data_2024 = pd.read_excel(file_path_2024)
    print("\n2024年数据预览：")
    print(data_2024.head())

    if 'Date' in data_2024.columns and 'price' in data_2024.columns:
        dates_2024 = data_2024['Date'].values
        data_values_2024 = data_2024['price'].values
    else:
        dates_2024 = data_2024.iloc[:, 0].values
        data_values_2024 = data_2024.iloc[:, 1].values

    print(f"2024年数据日期范围: {dates_2024[0]} 到 {dates_2024[-1]}")
    print(f"2024年股票价格范围: {data_values_2024.min():.4f} 到 {data_values_2024.max():.4f}")

    # 预测2024年的数据
    last_sequence = scaled_features[-time_step:]
    future_predictions = []
    current_sequence = last_sequence.copy()

    # 计算2024年数据的技术指标
    features_2024 = calculate_technical_indicators(data_values_2024)
    scaled_features_2024 = scaler.transform(features_2024)

    for i in range(len(dates_2024)):
        # 使用当前序列进行预测
        next_pred = model.predict(current_sequence.reshape(1, time_step, features.shape[1]))
        future_predictions.append(next_pred[0, 0])

        # 更新序列
        if i < len(dates_2024) - 1:
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = scaled_features_2024[i]

    # 将预测值转换回原始比例
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    inverse_scaled_features = np.zeros((len(future_predictions), features.shape[1]))
    inverse_scaled_features[:, 0] = future_predictions[:, 0]
    future_predictions = scaler.inverse_transform(inverse_scaled_features)[:, 0]

    # 计算2024年的预测误差
    mse_2024 = mean_squared_error(data_values_2024, future_predictions)
    rmse_2024 = math.sqrt(mse_2024)
    mae_2024 = mean_absolute_error(data_values_2024, future_predictions)


    # 计算相对准确率
    def calculate_accuracy_metrics(y_true, y_pred):
        # 计算相对误差
        relative_errors = np.abs(y_true - y_pred) / y_true
        # 计算平均相对误差
        mean_relative_error = np.mean(relative_errors) * 100
        # 计算相对准确率
        relative_accuracy = 100 - mean_relative_error

        # 计算不同误差范围的准确率
        accuracy_1_percent = np.mean(relative_errors <= 0.01) * 100  # 1%以内的准确率
        accuracy_3_percent = np.mean(relative_errors <= 0.03) * 100  # 3%以内的准确率
        accuracy_5_percent = np.mean(relative_errors <= 0.05) * 100  # 5%以内的准确率

        return relative_accuracy, accuracy_1_percent, accuracy_3_percent, accuracy_5_percent


    relative_accuracy, acc_1, acc_3, acc_5 = calculate_accuracy_metrics(data_values_2024, future_predictions)

    print("\n2024年预测效果评估：")
    print(f"RMSE: {rmse_2024:.4f}")
    print(f"MAE: {mae_2024:.4f}")
    print(f"相对准确率: {relative_accuracy:.2f}%")
    print(f"误差在1%以内的比例: {acc_1:.2f}%")
    print(f"误差在3%以内的比例: {acc_3:.2f}%")
    print(f"误差在5%以内的比例: {acc_5:.2f}%")

    # 计算综合准确率
    comprehensive_results = calculate_comprehensive_accuracy(data_values_2024, future_predictions)

    print("\n模型综合准确率评估：")
    print("-" * 50)
    for metric, value in comprehensive_results.items():
        print(f"{metric}: {value:.2f}%")
    print("-" * 50)

    # 绘制2024年预测结果对比图
    plt.figure(figsize=(15, 7))
    plt.plot(dates_2024, data_values_2024, label='2024年真实数据', color='blue', linewidth=2)
    plt.plot(dates_2024, future_predictions, label='2024年预测数据', color='red', linewidth=2)
    plt.title('2024年股票价格预测结果对比（改进版）')
    plt.xlabel('日期')
    plt.ylabel('股票价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(r'D:\毕设\lstm_stock_2024_predictions_improved.png')

    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练历史')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 打印每日预测结果对比

    print("\n2024年每日预测结果对比：")
    for i in range(len(dates_2024)):
        print(f"日期: {dates_2024[i]}")
        print(f"真实价格: {data_values_2024[i]:.2f}")
        print(f"预测价格: {future_predictions[i]:.2f}")
        print(f"误差: {abs(data_values_2024[i] - future_predictions[i]):.2f}")
        print(f"相对误差: {abs(data_values_2024[i] - future_predictions[i]) / data_values_2024[i] * 100:.2f}%")
        print("-" * 50)

    plt.savefig(r'D:\毕设\lstm_stock_training_history_improved.png')
else:
    print("\n未找到2024年的数据文件，无法进行预测和对比分析")

print("\n改进版LSTM股票价格预测模型训练和预测已完成")
print(f"2024年预测对比图保存为 D:\\毕设\\lstm_stock_2024_predictions_improved.png")
print(f"训练历史图保存为 D:\\毕设\\lstm_stock_training_history_improved.png")


# 计算并输出详细的误差指标
def calculate_error_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\n预测误差评估：")
    print("-" * 50)
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"平均绝对误差(MAE): {mae:.2f}")
    print(f"均方根误差(RMSE): {rmse:.2f}")
    print(f"平均绝对百分比误差(MAPE): {mape:.2f}%")
    print("-" * 50)

    return mse, mae, rmse, mape


# 在预测完成后调用此函数
mse, mae, rmse, mape = calculate_error_metrics(data_values_2024, future_predictions)
