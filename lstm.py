import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
train_data = pd.read_csv(r'.\data\raindistrain.csv')
val_data = pd.read_csv(r'.\data\raindisval.csv')
test_data = pd.read_csv(r'.\data\raindistest.csv')

# 提取特征和标签
def prepare_data(df):
    df['new_dis'] = (df['dis'] - df['baseflow']) * 3600 / df['area']
    df['new_dis'] = np.maximum(df['new_dis'], 0)
    X = df[['rainmean']].values
    y = df['dis'].values
    return X, y

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
X_train, y_train = prepare_data(train_data)
X_train = scaler.fit_transform(X_train)

X_val, y_val = prepare_data(val_data)
X_val = scaler.transform(X_val)

X_test, y_test = prepare_data(test_data)
X_test = scaler.transform(X_test)

# 创建时间序列数据
def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        labels.append(y[i+seq_length])
    return np.array(sequences), np.array(labels)

# 设置时间步长
seq_length = 10

# 为训练集、验证集和测试集创建时间序列
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# 数据标准化（使用MinMaxScaler）
X_train_seq = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
X_val_seq = scaler.transform(X_val_seq.reshape(-1, X_val_seq.shape[-1])).reshape(X_val_seq.shape)
X_test_seq = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)

# 转换为PyTorch张量，并移到GPU
X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
y_val_seq = torch.tensor(y_val_seq, dtype=torch.float32).to(device)
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

# 创建训练、验证和测试数据集
train_dataset = TensorDataset(X_train_seq, y_train_seq)
val_dataset = TensorDataset(X_val_seq, y_val_seq)
test_dataset = TensorDataset(X_test_seq, y_test_seq)

# 使用DataLoader来处理批量数据
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        return predictions

# 创建模型并将其移到GPU
model = LSTMModel(input_size=1, hidden_layer_size=128, output_size=1, num_layers=2)
model = model.to(device)

# 训练模型函数
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            # 将数据移到GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

# 训练模型
train_model(model, train_loader, val_loader, epochs=20, lr=0.001)

# 测试模型并可视化
model.eval()
with torch.no_grad():
    X_test_seq = X_test_seq.to(device)
    predictions = model(X_test_seq)
    
    # 逆变换（还原数据）
    test_predictions_rescaled = scaler.inverse_transform(predictions.cpu().numpy())
    test_labels_rescaled = scaler.inverse_transform(y_test_seq.cpu().numpy().reshape(-1, 1))

# 可视化对比
plt.figure(figsize=(10, 6))
plt.plot(test_labels_rescaled, label='True Values', color='blue')
plt.plot(test_predictions_rescaled, label='Predicted Values', color='red', linestyle='--')
plt.title('True vs Predicted Values on Test Set')
plt.xlabel('Time Steps')
plt.ylabel('Rain Distribution (Discharge)')
plt.legend()
plt.show()
