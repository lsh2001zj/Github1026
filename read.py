import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import numpy as np

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
scale = 50

# 读取 Parquet 文件
df_train = pd.read_parquet('train.parquet')
df_test= pd.read_parquet('test.parquet')
# 训练集与测试集的划分,df_train_1是训练集
feature_columns = [f'X{i}' for i in range(559)]  # X0 到 X558
# 创建一个空字典用于存储每一列的最小值
min_values = {}
for col in feature_columns:
    min_value = min(df_train[col].min(),df_test[col].min()) 
    min_values[col] = min_value
# 提取训练集的第2——第560列作为输入特征
min_values_test = {}
for col in feature_columns:
    #min_value = min(df_train[col].min(),df_test[col].min()) 
    min_value = df_test[col].min()
    min_values_test[col] = min_value
df_train_x = df_train.iloc[:, 1:560]
df_test_x = df_test.iloc[:, 1:560]
df_train_y= df_train.iloc[:, -8:]
df_train_y = df_train_y * scale

def add_log_transformed_columns(df, min_values):
    for col in feature_columns:
        new_col_name = f'Xlog_{col[1:]}'  
        min_value = min_values[col]
        df[new_col_name] = np.log(df[col] - min_value + 0.000001)
    return df

# 对 df1 和 df2 进行处理
df_train_x = add_log_transformed_columns(df_train_x, min_values)
df_test_x = add_log_transformed_columns(df_test_x, min_values)

# 训练集中的训练集
#df_train_x1 = df_train_x.iloc[:math.floor(0.7*len(df_train_x)),:]
#df_train_y1 = df_train_y.iloc[:math.floor(0.7*len(df_train_y)),:]
# 训练集中的测试集
#df_train_x2 = df_train_x.iloc[math.floor(0.7*len(df_train_x)):,:]
#df_train_y2 = df_train_y.iloc[math.floor(0.7*len(df_train_y)):,:]
#x_test = torch.tensor(df_train_x2.values, dtype=torch.float32)
#y_test = torch.tensor(df_train_y2.values, dtype=torch.float32)
x_test = df_test_x

# 计算均方根误差（RMSE）
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 计算所有目标的列平均RMSE
def calculate_columnwise_rmse(y_true, y_pred, num_targets):
    columnwise_rmse = []
    for i in range(num_targets):
        rmse = calculate_rmse(y_true[:, i], y_pred[i].squeeze())  # 对每一列计算RMSE
        columnwise_rmse.append(rmse)
    # 计算列平均RMSE
    return np.mean(columnwise_rmse)

# 定义主 MLP 模型
class MainMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(MainMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 第一层 1118 -> 256
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第二层 256 -> 128
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        return x

# 定义每个目标的 MLP 模型
class TargetMLP(nn.Module):
    def __init__(self):
        super(TargetMLP, self).__init__()
        self.fc1 = nn.Linear(128, 32)  # 输入128 -> 隐层64
        self.fc2 = nn.Linear(32, 8)   # 隐层64 -> 隐层16
        self.fc3 = nn.Linear(8, 1)    # 隐层16 -> 输出1（预测一个目标值）
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        x = self.fc3(x)  # 输出层
        return x

# 定义完整的模型
class DNNModel(nn.Module):
    def __init__(self, input_dim, main_hidden_dim1, main_hidden_dim2,num_targets):
        super(DNNModel, self).__init__()
        self.main_mlp = MainMLP(input_dim, main_hidden_dim1, main_hidden_dim2)
        self.target_mlps = nn.ModuleList([TargetMLP() for _ in range(num_targets)])
    def forward(self, x):
        shared_rep = self.main_mlp(x)
        predictions = [target_mlp(shared_rep) for target_mlp in self.target_mlps] 
        return predictions

# 模型参数
input_dim = 1118        # 输入特征的维度
main_hidden_dim1 = 256  # 第一层隐藏维度
main_hidden_dim2 = 128  # 第二层隐藏维度
num_targets = 8  # 预测目标数量
model = DNNModel(input_dim, main_hidden_dim1, main_hidden_dim2,num_targets)
criterion = nn.MSELoss()  # 假设预测目标是连续值，使用MSE损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
x_data = torch.tensor(df_train_x.values, dtype=torch.float32)
y_data = torch.tensor(df_train_y.values, dtype=torch.float32)
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = 0
        for i in range(num_targets):
            loss += criterion(predictions[i].squeeze(), y_batch[:, i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    #model.eval()  # 切换到评估模式
    #with torch.no_grad():
    #    test_predictions = model(x_test)  # 假设 x_test 是测试数据
    #    columnwise_rmse = calculate_columnwise_rmse(y_test, test_predictions, num_targets)
    #    print(f"Epoch [{epoch+1}/{num_epochs}] - Columnwise Mean RMSE: {columnwise_rmse:.4f}")
    #model.train()  # 切换回训练模式

# 下面对测试集进行预测
result=pd.DataFrame(columns=['ID', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7'])
for i in range(len(df_test)):
    test= torch.tensor(x_test.iloc[i].values,dtype=torch.float32)
    test_predictions = model(test)
    new_row = list([df_test.iloc[i,0]] + [t.item()/scale for t in test_predictions])
    result.loc[len(result)] = new_row
result['ID'] = result['ID'].astype(int)
result.to_csv('predictions_result.csv', index=False)