# ========================== 1. 导入所有依赖库 ==========================
import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ========================== 新增：解决matplotlib中文显示问题（关键修改） ==========================
# 全局配置中文字体，一次性解决所有图片的中文乱码/方框问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows优先：微软雅黑（无需额外安装，系统自带）
# 若在Linux系统运行，替换为：plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
# 若在Mac OS运行，替换为：plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为空心方框的问题
# ===================================================================================

# 设置设备（自动检测GPU/CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# ========================== 2. 文件夹路径配置与自动创建 ==========================
# 总结果根目录
ROOT_RESULT_DIR = "./stock_training_results"
# 子文件夹路径（按文件类型分类）
LOG_DIR = os.path.join(ROOT_RESULT_DIR, "logs")  # 日志文件（全局日志+训练单条日志）
MODEL_DIR = os.path.join(ROOT_RESULT_DIR, "models")  # 保存模型权重
LOSS_CURVES_DIR = os.path.join(ROOT_RESULT_DIR, "loss_curves")  # 损失曲线图片
PREDICTION_DIR = os.path.join(ROOT_RESULT_DIR, "prediction_results")  # 预测结果图片
EVAL_RESULT_DIR = os.path.join(ROOT_RESULT_DIR, "eval_results")  # 评估结果CSV

# 自动创建所有文件夹（不存在则创建，存在则跳过）
os.makedirs(ROOT_RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOSS_CURVES_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(EVAL_RESULT_DIR, exist_ok=True)

print(f"所有结果将保存至根目录：{ROOT_RESULT_DIR}")
print(f"日志路径：{LOG_DIR}")
print(f"模型路径：{MODEL_DIR}")
print(f"损失曲线路径：{LOSS_CURVES_DIR}")
print(f"预测图片路径：{PREDICTION_DIR}")
print(f"评估结果路径：{EVAL_RESULT_DIR}")

# 核心参数配置
FOLDER_PATH = "./Data/Stocks"
TARGET_COL = "Close"
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
STOCK_COUNT_GRADIENT = [1, 10, 50, 100, 200, 500, 1000]
RESERVED_TEST_STOCKS = 25  # 目标预留测试股票数量

# ========================== 3. 日志初始化（修改为日志文件夹路径） ==========================
# 全局日志文件路径（放到LOG_DIR下）
GLOBAL_LOG_FILE = os.path.join(LOG_DIR, "stock_gradient_training_pytorch.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(GLOBAL_LOG_FILE, encoding='utf-8'),  # 保存到日志文件夹
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"所有文件夹创建完成，全局日志保存至：{GLOBAL_LOG_FILE}")
logger.info("matplotlib中文字体配置生效，后续所有图片中文将正常显示")  # 新增日志提示

# 全局结果字典
global_eval_results = {}

# ========================== 4. 自定义工具类（早停+模型） ==========================
class EarlyStopping:
    """自定义早停类（等效TensorFlow EarlyStopping）"""
    def __init__(self, patience=5, verbose=True, restore_best_weights=True):
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = np.Inf
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # 保存最佳权重
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)  # 恢复最佳权重
                    if self.verbose:
                        logger.info("Restored best model weights from early stopping")
                return True  # 触发早停
        return False  # 未触发早停

class SelfAttention(nn.Module):
    """自定义自注意力层（等效TensorFlow Attention）"""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算注意力权重 (batch_size, seq_len, seq_len)
        attention_weights = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        attention_weights = self.softmax(attention_weights)

        # 加权求和 (batch_size, seq_len, hidden_dim)
        attention_output = torch.bmm(attention_weights, v)
        return attention_output

class LSTMModel(nn.Module):
    """PyTorch LSTM模型（修正：提取最后一个时间步，匹配标签形状）"""
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)  # out: (batch, 60, 64)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # out: (batch, 60, 32)
        out = self.dropout2(out)
        # 核心修正1：提取最后一个时间步特征 (batch, 32)
        out = out[:, -1, :]  # 丢弃前59个时间步，仅保留最后一个
        out = self.fc1(out)   # (batch, 16)
        out = self.relu(out)
        out = self.fc2(out)   # (batch, 1) 与标签形状对齐
        return out

class GRUModel(nn.Module):
    """PyTorch GRU模型（修正：提取最后一个时间步，匹配标签形状）"""
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.gru1(x)  # out: (batch, 60, 64)
        out = self.dropout1(out)
        out, _ = self.gru2(out)  # out: (batch, 60, 32)
        out = self.dropout2(out)
        # 核心修正2：提取最后一个时间步特征 (batch, 32)
        out = out[:, -1, :]
        out = self.fc1(out)   # (batch, 16)
        out = self.relu(out)
        out = self.fc2(out)   # (batch, 1) 与标签形状对齐
        return out

class CNNLSTMModel(nn.Module):
    """PyTorch CNN-LSTM模型（修正：提取最后一个时间步，匹配标签形状）"""
    def __init__(self, input_dim, conv_filters=32, kernel_size=3, pool_size=2, lstm_hidden=64, dropout_rate=0.2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=kernel_size, padding='same')
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_size, padding=0)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch, 60, input_dim) → 转置为 (batch, input_dim, 60) 适配Conv1d
        x = x.transpose(1, 2)
        out = self.conv1d(x)
        out = self.relu(out)
        out = self.maxpool1d(out)  # (batch, 32, 30)（池化后seq_len从60变为30）
        # 转置回 (batch, 30, 32) 适配LSTM
        out = out.transpose(1, 2)
        out, _ = self.lstm(out)    # (batch, 30, 64)
        out = self.dropout(out)
        # 核心修正3：提取最后一个时间步特征 (batch, 64)
        out = out[:, -1, :]
        out = self.fc1(out)        # (batch, 16)
        out = self.relu(out)
        out = self.fc2(out)        # (batch, 1) 与标签形状对齐
        return out

class AttentionLSTMModel(nn.Module):
    """PyTorch Attention-LSTM模型（修正：提取最后一个时间步，匹配标签形状）"""
    def __init__(self, input_dim, lstm_hidden1=64, lstm_hidden2=32, dropout_rate=0.2):
        super(AttentionLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, lstm_hidden1, batch_first=True)
        self.attention = SelfAttention(lstm_hidden1)
        self.lstm2 = nn.LSTM(lstm_hidden1, lstm_hidden2, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)     # (batch, 60, 64)
        out = self.attention(out)  # (batch, 60, 64)
        out, _ = self.lstm2(out)   # (batch, 60, 32)
        out = self.dropout(out)
        # 核心修正4：提取最后一个时间步特征 (batch, 32)
        out = out[:, -1, :]
        out = self.fc1(out)        # (batch, 16)
        out = self.relu(out)
        out = self.fc2(out)        # (batch, 1) 与标签形状对齐
        return out

# ========================== 5. 核心工具函数（关键修改：确保预留测试股票凑够数量） ==========================
def batch_load_stock_data(folder_path, max_stocks, reserved_stocks=0):
    """
    批量加载股票数据
    关键修改：预留测试股票会遍历所有文件，直到凑够reserved_stocks只有效股票或遍历结束
    """
    # 1. 获取所有股票文件并打乱（保证随机性）
    stock_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    np.random.shuffle(stock_files)
    total_file_count = len(stock_files)
    logger.info(f"总股票文件数：{total_file_count}")

    # 2. 优先收集足够数量的预留测试股票（核心修改：遍历所有文件凑够数量）
    test_stock_dict = {}
    reserved_file_names = []  # 记录已选中的预留测试股票文件名，用于后续排除
    if reserved_stocks > 0:
        logger.info(f"开始收集{reserved_stocks}只有效预留测试股票...")
        for idx, file in enumerate(stock_files):
            # 若已凑够预留测试股票数量，停止遍历
            if len(test_stock_dict) >= reserved_stocks:
                break

            file_path = os.path.join(folder_path, file)
            stock_name = file[:-4]
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                df = df.drop('OpenInt', axis=1)
                df = df.interpolate(method='time')
                df = df.dropna()
                # 严格校验数据量：确保有足够的序列用于测试
                min_test_data_require = SEQUENCE_LENGTH + 50
                if len(df) >= min_test_data_require:
                    test_stock_dict[stock_name] = df
                    reserved_file_names.append(file)  # 记录文件名
                    # 打印收集进度
                    if len(test_stock_dict) % 5 == 0 and len(test_stock_dict) > 0:
                        logger.info(f"预留测试股票收集进度：{len(test_stock_dict)}/{reserved_stocks}")
                else:
                    logger.debug(f"预留测试股票{stock_name}数据量不足（仅{len(df)}条），跳过（需至少{min_test_data_require}条）")
            except Exception as e:
                logger.error(f"加载预留测试股票{stock_name}失败：{str(e)}，继续遍历下一支")
                continue

        # 打印预留测试股票收集结果
        collected_reserved_count = len(test_stock_dict)
        logger.info(f"预留测试股票收集完成：共{collected_reserved_count}只有效股票（目标{reserved_stocks}只）")
        if collected_reserved_count < reserved_stocks:
            logger.warning(f"警告：遍历完所有{total_file_count}只股票后，仍未凑够目标预留测试股票数量（缺{reserved_stocks - collected_reserved_count}只）")
        else:
            logger.info(f"预留测试股票数量达标！")

    # 3. 加载训练股票（排除已选中的预留测试股票，然后凑够max_stocks只）
    train_stock_dict = {}
    if max_stocks > 0:
        # 排除预留测试股票，得到训练候选文件
        train_candidate_files = [f for f in stock_files if f not in reserved_file_names]
        candidate_count = len(train_candidate_files)
        logger.info(f"可用于训练的股票文件数：{candidate_count}，开始遍历收集{max_stocks}只有效训练股票...")

        # 逐个遍历训练候选文件，直到凑够max_stocks
        for idx, file in enumerate(train_candidate_files):
            if len(train_stock_dict) >= max_stocks:
                break

            file_path = os.path.join(folder_path, file)
            stock_name = file[:-4]
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                df = df.drop('OpenInt', axis=1)
                df = df.interpolate(method='time')
                df = df.dropna()
                # 检查训练数据量是否充足
                min_train_data_require = SEQUENCE_LENGTH + 10
                if len(df) < min_train_data_require:
                    logger.debug(f"训练股票{stock_name}数据量不足（仅{len(df)}条），跳过（需至少{min_train_data_require}条）")
                    continue
                # 数据充足，加入训练集
                train_stock_dict[stock_name] = df
                # 打印训练股票收集进度
                if len(train_stock_dict) % 10 == 0 and len(train_stock_dict) > 0:
                    logger.info(f"训练股票收集进度：{len(train_stock_dict)}/{max_stocks}")
            except Exception as e:
                logger.error(f"加载训练股票{stock_name}失败：{str(e)}，继续遍历下一支")
                continue

        # 打印训练股票收集结果
        collected_train_count = len(train_stock_dict)
        if collected_train_count < max_stocks:
            logger.warning(f"遍历完所有{candidate_count}只训练候选股票后，仅收集到{collected_train_count}只有效训练股票（目标{max_stocks}只）")
        else:
            logger.info(f"训练股票收集完成：成功获取{collected_train_count}只有效训练股票（目标{max_stocks}只）")

    return train_stock_dict, test_stock_dict

def create_stock_sequences(stock_data_dict, sequence_length, target_col):
    """构造序列数据（逻辑不变）"""
    all_X = []
    all_y = []
    scaler_dict = {}

    for stock_name, df in stock_data_dict.items():
        df_feat = df.copy()
        df_feat['MA5'] = df_feat[target_col].rolling(window=5).mean()
        df_feat['MA10'] = df_feat[target_col].rolling(window=10).mean()
        df_feat['Price_Change'] = df_feat['Close'] - df_feat['Open']
        df_feat['High_Low_Spread'] = df_feat['High'] - df_feat['Low']
        df_feat['Close_Lag1'] = df_feat['Close'].shift(1)
        df_feat = df_feat.dropna()
        if len(df_feat) < sequence_length + 1:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_feat)
        scaled_df = pd.DataFrame(scaled_data, columns=df_feat.columns, index=df_feat.index)
        scaler_dict[stock_name] = scaler

        feature_cols = scaled_df.columns.drop(target_col).tolist()
        X, y = [], []
        for i in range(sequence_length, len(scaled_df)):
            X.append(scaled_df[feature_cols].iloc[i-sequence_length:i].values)
            y.append(scaled_df[target_col].iloc[i])
        all_X.append(np.array(X))
        all_y.append(np.array(y))

    if not all_X or not all_y:
        return np.array([]), np.array([]), scaler_dict
    X_concat = np.concatenate(all_X, axis=0)
    y_concat = np.concatenate(all_y, axis=0)
    logger.info(f"序列构造完成：总样本数{X_concat.shape[0]}，序列长度{X_concat.shape[1]}，特征数{X_concat.shape[2]}")
    return X_concat, y_concat, scaler_dict

def build_models_pytorch(input_dim):
    """构建PyTorch 4种模型（逻辑不变，已修正输出形状）"""
    models = {
        "LSTM": LSTMModel(input_dim).to(device),
        "GRU": GRUModel(input_dim).to(device),
        "CNN-LSTM": CNNLSTMModel(input_dim).to(device),
        "Attention-LSTM": AttentionLSTMModel(input_dim).to(device)
    }
    return models

def train_single_model_pytorch(model, model_name, stock_count, X_train, y_train, X_val, y_val):
    """PyTorch模型训练（修改日志、模型、损失曲线保存路径）"""
    # 转换为Tensor并构建DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 时间序列不shuffle
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True)
    train_loss_history = []
    val_loss_history = []

    # 1. 训练日志CSV路径（放到LOG_DIR文件夹）
    train_log_file = os.path.join(LOG_DIR, f"training_log_stockCount_{stock_count}_{model_name}_pytorch.csv")
    with open(train_log_file, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss\n")
    logger.info(f"当前模型训练日志将保存至：{train_log_file}")

    logger.info(f"开始训练{model_name}模型（股票数量：{stock_count}，设备：{device}）")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss_epoch = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)  # 输出形状 (batch_size, 1)
            loss = criterion(outputs, batch_y)  # 形状完全匹配，可正常计算
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * batch_X.size(0)
        train_loss_epoch /= len(train_loader.dataset)
        train_loss_history.append(train_loss_epoch)

        # 验证阶段
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss_epoch += loss.item() * batch_X.size(0)
        val_loss_epoch /= len(val_loader.dataset)
        val_loss_history.append(val_loss_epoch)

        # 写入日志
        with open(train_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{train_loss_epoch:.6f},{val_loss_epoch:.6f}\n")

        # 打印日志
        if (epoch+1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}")

        # 早停判断
        if early_stopping(val_loss_epoch, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 2. 保存模型权重（放到MODEL_DIR文件夹）
    model_save_path = os.path.join(MODEL_DIR, f"model_stockCount_{stock_count}_{model_name}_pytorch.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型权重保存至：{model_save_path}")

    # 3. 可视化训练损失并保存（放到LOSS_CURVES_DIR文件夹）
    loss_curve_path = os.path.join(LOSS_CURVES_DIR, f"loss_curve_stockCount_{stock_count}_{model_name}_pytorch.png")
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_history, label='训练损失')
    plt.plot(val_loss_history, label='验证损失')
    plt.title(f"股票数量={stock_count} - {model_name} 训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(loss_curve_path)
    plt.close()
    logger.info(f"损失曲线保存至：{loss_curve_path}")

    return model

def evaluate_model_generalization_pytorch(model, model_name, stock_count, test_stock_dict, sequence_length, target_col):
    """PyTorch模型评估（修改预测图片保存路径）"""
    all_mae = []
    all_mse = []
    all_rmse = []
    all_r2 = []

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for stock_name, df in test_stock_dict.items():
            X_test, y_test, scaler_dict = create_stock_sequences({stock_name: df}, sequence_length, target_col)
            if len(X_test) == 0:
                logger.warning(f"测试股票{stock_name}无有效序列，跳过")
                continue

            # 转换为Tensor
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred_tensor = model(X_test_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()  # 转回CPU和numpy

            # 反归一化（修复形状不匹配问题）
            scaler = scaler_dict[stock_name]
            n_features = scaler.n_features_in_
            # 构造特征工程后的df_feat
            df_feat = df.copy()
            df_feat['MA5'] = df_feat[target_col].rolling(window=5).mean()
            df_feat['MA10'] = df_feat[target_col].rolling(window=10).mean()
            df_feat['Price_Change'] = df_feat['Close'] - df_feat['Open']
            df_feat['High_Low_Spread'] = df_feat['High'] - df_feat['Low']
            df_feat['Close_Lag1'] = df_feat['Close'].shift(1)
            target_idx_feat = df_feat.columns.get_loc(target_col)

            # 构造dummy数组
            dummy_test = np.zeros((len(y_test), n_features))
            dummy_test[:, target_idx_feat] = y_test.flatten()
            y_test_real = scaler.inverse_transform(dummy_test)[:, target_idx_feat]

            dummy_pred = np.zeros((len(y_pred), n_features))
            dummy_pred[:, target_idx_feat] = y_pred.flatten()
            y_pred_real = scaler.inverse_transform(dummy_pred)[:, target_idx_feat]

            # 计算指标
            mae = mean_absolute_error(y_test_real, y_pred_real)
            mse = mean_squared_error(y_test_real, y_pred_real)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_real, y_pred_real)

            all_mae.append(mae)
            all_mse.append(mse)
            all_rmse.append(rmse)
            all_r2.append(r2)

            # 预测图片保存路径（放到PREDICTION_DIR文件夹）
            pred_img_path = os.path.join(PREDICTION_DIR, f"pred_result_stockCount_{stock_count}_{model_name}_{stock_name}_pytorch.png")
            plt.figure(figsize=(12, 4))
            plt.plot(y_test_real, label='真实收盘价')
            plt.plot(y_pred_real, label='预测收盘价')
            plt.title(f"股票数量={stock_count} - {model_name} - 测试股票{stock_name}预测结果")
            plt.xlabel("样本序号")
            plt.ylabel("价格")
            plt.legend()
            plt.savefig(pred_img_path)
            plt.close()
            logger.debug(f"预测结果图片保存至：{pred_img_path}")

    # 计算平均指标
    avg_mae = np.mean(all_mae)
    avg_mse = np.mean(all_mse)
    avg_rmse = np.mean(all_rmse)
    avg_r2 = np.mean(all_r2)

    logger.info(f"股票数量={stock_count} - {model_name} 泛化能力评估（平均）：")
    logger.info(f"MAE={avg_mae:.2f}, MSE={avg_mse:.2f}, RMSE={avg_rmse:.2f}, R2={avg_r2:.4f}")

    return {
        "MAE": avg_mae,
        "MSE": avg_mse,
        "RMSE": avg_rmse,
        "R2": avg_r2,
        "per_stock_metrics": {"MAE": all_mae, "MSE": all_mse, "RMSE": all_rmse, "R2": all_r2}
    }

# ========================== 6. 主流程（关键修改：强化预留测试股票数量校验） ==========================
if __name__ == "__main__":
    # 加载预留独立测试股票（优先确保数量达标）
    _, fixed_test_stock_dict = batch_load_stock_data(
        folder_path=FOLDER_PATH,
        max_stocks=0,
        reserved_stocks=RESERVED_TEST_STOCKS
    )

    # 强化校验：确保预留测试股票数量达标，否则终止程序
    actual_reserved_count = len(fixed_test_stock_dict)
    if actual_reserved_count == 0:
        logger.error("未加载到任何独立测试股票，程序终止")
        exit(1)
    if actual_reserved_count < RESERVED_TEST_STOCKS:
        logger.error(f"预留测试股票数量不足（仅{actual_reserved_count}只，目标{RESERVED_TEST_STOCKS}只），程序终止")
        exit(1)
    logger.info(f"预留测试股票数量校验通过：{actual_reserved_count}只（达标{RESERVED_TEST_STOCKS}只要求）")

    # 循环遍历股票数量梯度
    for stock_count in STOCK_COUNT_GRADIENT:
        logger.info(f"\n=====================================")
        logger.info(f"开始处理股票数量：{stock_count}")
        logger.info(f"=====================================\n")

        # 1. 加载训练股票（已修改为自动凑够数量，且排除预留测试股票）
        train_stock_dict, _ = batch_load_stock_data(
            folder_path=FOLDER_PATH,
            max_stocks=stock_count,
            reserved_stocks=RESERVED_TEST_STOCKS
        )
        # 仅当遍历完所有股票仍未收集到任何有效股票时，才跳过
        if len(train_stock_dict) == 0:
            logger.error(f"未加载到任何有效训练股票，跳过当前股票数量梯度")
            continue

        # 2. 构造序列
        X_multi, y_multi, _ = create_stock_sequences(
            train_stock_dict,
            sequence_length=SEQUENCE_LENGTH,
            target_col=TARGET_COL
        )
        if len(X_multi) == 0:
            logger.error(f"股票数量{stock_count}：无有效序列，跳过")
            continue

        # 3. 划分训练集/验证集
        train_size = int(0.8 * len(X_multi))
        val_size = int(0.1 * len(X_multi))
        X_train = X_multi[:train_size]
        y_train = y_multi[:train_size]
        X_val = X_multi[train_size:train_size+val_size]
        y_val = y_multi[train_size:train_size+val_size]
        X_inner_test = X_multi[train_size+val_size:]
        y_inner_test = y_multi[train_size+val_size:]

        logger.info(f"数据集划分：训练集{X_train.shape}，验证集{X_val.shape}，内部测试集{X_inner_test.shape}")

        # 4. 构建PyTorch模型（已修正输出形状）
        input_dim = X_train.shape[2]  # 特征数
        models = build_models_pytorch(input_dim)

        # 5. 训练所有模型
        trained_models = {}
        for model_name, model in models.items():
            trained_model = train_single_model_pytorch(
                model=model,
                model_name=model_name,
                stock_count=stock_count,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
            trained_models[model_name] = trained_model

        # 6. 评估所有模型
        stock_count_eval_results = {}
        for model_name, model in trained_models.items():
            eval_res = evaluate_model_generalization_pytorch(
                model=model,
                model_name=model_name,
                stock_count=stock_count,
                test_stock_dict=fixed_test_stock_dict,
                sequence_length=SEQUENCE_LENGTH,
                target_col=TARGET_COL
            )
            stock_count_eval_results[model_name] = eval_res

        # 7. 保存结果
        global_eval_results[stock_count] = stock_count_eval_results
        logger.info(f"股票数量{stock_count}：所有模型评估完成，结果已保存")

    # ========================== 结果汇总与可视化（修改评估CSV保存路径） ==========================
    logger.info(f"\n=====================================")
    logger.info(f"所有股票数量梯度训练完成，开始汇总结果")
    logger.info(f"=====================================\n")

    # 整理结果为DataFrame
    result_rows = []
    for stock_count, model_results in global_eval_results.items():
        for model_name, metrics in model_results.items():
            result_rows.append({
                "股票数量": stock_count,
                "模型名称": model_name,
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"]
            })
    results_df = pd.DataFrame(result_rows)

    # 全局评估结果CSV路径（放到EVAL_RESULT_DIR文件夹）
    eval_csv_path = os.path.join(EVAL_RESULT_DIR, "global_gradient_eval_results_pytorch.csv")
    results_df.to_csv(eval_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"全局评估结果已保存至：{eval_csv_path}")
    print("\n=== 全局模型评估结果汇总 ===")
    print(results_df.round(4))

    # RMSE对比可视化（保存到LOSS_CURVES_DIR，统一图片管理）
    rmse_plot_path = os.path.join(LOSS_CURVES_DIR, "rmse_vs_stock_count_pytorch.png")
    plt.figure(figsize=(14, 6))
    for model_name in models.keys():
        subset = results_df[results_df["模型名称"] == model_name]
        plt.plot(subset["股票数量"], subset["RMSE"], marker='o', label=model_name)
    plt.title("不同股票数量下各模型RMSE对比（越低越好）")
    plt.xlabel("训练股票数量")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(rmse_plot_path)
    plt.show()
    logger.info(f"RMSE对比图保存至：{rmse_plot_path}")

    # R2对比可视化（保存到LOSS_CURVES_DIR，统一图片管理）
    r2_plot_path = os.path.join(LOSS_CURVES_DIR, "r2_vs_stock_count_pytorch.png")
    plt.figure(figsize=(14, 6))
    for model_name in models.keys():
        subset = results_df[results_df["模型名称"] == model_name]
        plt.plot(subset["股票数量"], subset["R2"], marker='s', label=model_name)
    plt.title("不同股票数量下各模型R2对比（越高越好）")
    plt.xlabel("训练股票数量")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(r2_plot_path)
    plt.show()
    logger.info(f"R2对比图保存至：{r2_plot_path}")

    # 统计显著性分析
    if 1 in global_eval_results and 1000 in global_eval_results:
        model_name = "Attention-LSTM"
        res_1 = global_eval_results[1][model_name]["per_stock_metrics"]["MAE"]
        res_1000 = global_eval_results[1000][model_name]["per_stock_metrics"]["MAE"]
        t_stat, p_value = ttest_ind(res_1, res_1000)
        logger.info(f"\n{model_name}模型：1只股票 vs 1000只股票 MAE 显著性分析")
        logger.info(f"t统计量：{t_stat:.4f}，p值：{p_value:.4f}")
        if p_value < 0.05:
            logger.info("结论：两者存在显著差异，1000只股票训练的模型泛化能力更优")
        else:
            logger.info("结论：两者无显著差异（可能受样本量或噪声影响）")
    else:
        logger.warning("无法进行1只 vs 1000只股票的显著性分析：缺少对应数据")

    logger.info(f"\n所有实验流程完成！所有结果已按文件夹分类保存至：{ROOT_RESULT_DIR}")