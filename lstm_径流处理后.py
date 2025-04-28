import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 使用StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
train_data = pd.read_csv(r'.\data\raindistrain.csv')
val_data = pd.read_csv(r'.\data\raindisval.csv')
test_data = pd.read_csv(r'.\data\raindistest.csv')

# 提取特征和标签
def prepare_data(df):
    # 计算新的标签列 (dis - baseflow) / area
    df['new_dis'] = (df['dis'] - df['baseflow']) * 3600 / df['area']
    
    # 如果新标签小于0，则设置为0
    df['new_dis'] = np.maximum(df['new_dis'], 0)
    
    # 选择特征列
    X = df[['rainmean']].values
    
    # 使用新的标签列替换旧标签
    y = df['dis'].values
    return X, y

# 数据标准化（使用 StandardScaler）
# 使用MinMaxScaler进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))  # 默认将数据缩放到[0, 1]

# 训练数据标准化
X_train, y_train = prepare_data(train_data)
X_train = scaler.fit_transform(X_train)  # 计算训练数据的最大最小值，并进行转换

# 验证数据和测试数据使用相同的缩放器
X_val, y_val = prepare_data(val_data)
X_val = scaler.transform(X_val)  # 使用训练数据的最小最大值进行转换

X_test, y_test = prepare_data(test_data)
X_test = scaler.transform(X_test)  # 使用训练数据的最小最大值进行转换

# 创建时间序列数据
def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])  # 获取长度为seq_length的序列
        labels.append(y[i+seq_length])  # 对应标签
    return np.array(sequences), np.array(labels)

# 设置时间步长
seq_length = 10  # 设置时间步长

# 为训练集、验证集和测试集创建时间序列
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# 数据标准化（使用MinMaxScaler）
X_train_seq = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
X_val_seq = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
X_test_seq = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

# 转换为PyTorch张量
X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32)
X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_seq = torch.tensor(y_val_seq, dtype=torch.float32)
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32)

# 创建训练、验证和测试数据集
train_dataset = TensorDataset(X_train_seq, y_train_seq)
val_dataset = TensorDataset(X_val_seq, y_val_seq)
test_dataset = TensorDataset(X_test_seq, y_test_seq)

# 使用DataLoader来处理批量数据
batch_size = 128  # 设置批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        # LSTM层输入：input_size 是特征的维度，hidden_layer_size 是每层的神经元数量
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        # x 形状为 (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # 只使用最后一个时间步的输出
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    # 学习率 (lr), 训练轮数 (epochs) 是超参数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)  # 学习率和L2正则化（weight_decay）是超参数
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # 在验证集上评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}')

# 创建模型实例
hidden_layer_size = 128  # 设置隐藏层神经元数量
num_layers = 3  # 设置LSTM层数
model = LSTMModel(hidden_layer_size=hidden_layer_size, num_layers=num_layers)

# 训练模型
epochs = 20  # 设置训练轮数
learning_rate = 0.001  # 设置学习率
train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)

# 在测试集上评估
model.eval()
test_predictions = []
test_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        test_predictions.append(y_pred.numpy())
        test_labels.append(y_batch.numpy())

test_predictions = np.concatenate(test_predictions)
test_labels = np.concatenate(test_labels)

# 计算MSE
mse = mean_squared_error(test_labels, test_predictions)
print(f'Mean Squared Error on Test Set: {mse}')
# 逆变换（还原数据）
test_predictions_rescaled = scaler.inverse_transform(test_predictions)
test_labels_rescaled = scaler.inverse_transform(test_labels.reshape(-1, 1))

# 可视化对比
plt.figure(figsize=(10, 6))
plt.plot(test_labels_rescaled, label='True Values', color='blue')
plt.plot(test_predictions_rescaled, label='Predicted Values', color='red', linestyle='--')
plt.title('True vs Predicted Values on Test Set')
plt.xlabel('Time Steps')
plt.ylabel('Rain Distribution (Discharge)')
plt.legend()
plt.show()