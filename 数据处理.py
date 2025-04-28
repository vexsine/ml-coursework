import pandas as pd
import numpy as np

# 读取数据
train_data = pd.read_csv(r'.\data\raindistrain.csv')
val_data = pd.read_csv(r'.\data\raindisval.csv')
test_data = pd.read_csv(r'.\data\raindistest.csv')

# 提取特征和标签
def prepare_data(df):
    # 计算新的标签列 (dis - baseflow) / area
    df['new_dis'] = (df['dis'] - df['baseflow']) * 3600 / df['area']
    
    # 如果新标签小于0，则设置为0
    df['new_dis'] = np.maximum(df['new_dis'], 0)
    
    # 返回处理后的DataFrame（包括原数据和新标签）
    return df

# 处理数据
train_data_processed = prepare_data(train_data)
val_data_processed = prepare_data(val_data)
test_data_processed = prepare_data(test_data)

# 将处理后的数据写入新的CSV文件
train_data_processed.to_csv(r'.\data\raindistrain_processed.csv', index=False)
val_data_processed.to_csv(r'.\data\raindisval_processed.csv', index=False)
test_data_processed.to_csv(r'.\data\raindistest_processed.csv', index=False)

print("处理后的数据已成功保存到新的CSV文件。")
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