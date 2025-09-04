import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
Device = config.Device


def load_csv(file_path):
    """读取 CSV 文件"""
    return pd.read_csv(file_path)


def process_data(df):
    """
    读取并返回张量 [Time, 18]，
    不再 .unsqueeze(1) 让它变成 [Time, 1, 18]。
    """
    array = df.iloc[:, 1:].values  # shape: [Time, 18]
    return torch.tensor(array, dtype=torch.float32)


def create_sliding_window_samples(features, window_size, step_size, labels=None):
    """
    features.shape => [Time, 18]
    截取滑动窗口 => [num_windows, window_size, 18]
    再将其变为 [num_windows, 1, window_size, 18]  -- 不在外部再 permute
    """
    if labels is None:
        num_samples = features.shape[0]
        feature_windows = []
        for start in range(0, num_samples - window_size, step_size):
            end = start + window_size
            feature_windows.append(features[start:end])  # shape => [window_size, 18]
        feature_windows = torch.stack(feature_windows, dim=0)
        # 现在 => [num_windows, window_size, 18]

        # 如果要 [batch, C=1, time=window_size, area=18]，只需要一次操作:
        feature_windows = feature_windows.unsqueeze(1)  # => [num_windows, 1, window_size, 18]
        return feature_windows

    else:
        # 同理有 labels 时
        num_samples = features.shape[0]
        feature_windows = []
        label_windows = []
        for start in range(0, num_samples - window_size, step_size):
            end = start + window_size
            feature_windows.append(features[start:end])
            label_windows.append(labels[end])
        feature_windows = torch.stack(feature_windows, dim=0)
        label_windows = torch.stack(label_windows, dim=0)
        # => feature_windows: [num_windows, window_size, 18]
        # => label_windows:   [num_windows, ...]

        # 同样一次性加一维
        feature_windows = feature_windows.unsqueeze(1)  # => [num_windows, 1, window_size, 18]
        return feature_windows, label_windows


def normalize_per_channel(data):
    """
    对 [Time, 1, channels] 的每个 channel 独立做标准化
    """
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True)
    normalized_data = (data - mean) / (std + 1e-6)
    return normalized_data, mean, std


# PassengerFlowDataset 的构造函数加入设备选项
class PassengerFlowDataset(Dataset):
    def __init__(self, feature_windows, time_windows, label_windows, device=Device):
        super().__init__()
        self.feature_windows = feature_windows.to(device)
        self.time_windows = time_windows.to(device)
        self.label_windows = label_windows.to(device)

    def __len__(self):
        return self.feature_windows.shape[0]

    def __getitem__(self, idx):
        feature = self.feature_windows[idx]
        time_data = self.time_windows[idx]
        label = self.label_windows[idx]
        return {
            "feature": feature,  # [channels, window_size, 1]
            "time": time_data,  # [channels, window_size, 1]
            "label": label  # [1, label_dim] 或 [1, 18]
        }

def build_passenger_flow_dataloader(
    batch_size=64,
    shuffle=True,
    num_workers=0,
    device=Device,
    window_size=64,
    step_size=1,
    train_ratio=0.8
):
    """
    构建并返回 train_loader, test_loader，以及单独的 adj_tensor
    """
    # 1. 读取原始 CSV 并转换为张量 [Time, N_cols]
    passenger_flow = process_data(load_csv(config.passenger_flow_path))
    absolute_step_change_rate = process_data(load_csv(config.absolute_step_change_rate_path))
    passenger_flow_trends = process_data(load_csv(config.passenger_flow_trends_path))
    timetable = load_csv(config.timetable_path)
    timetable_features = process_data(timetable)

    # 2. 读取邻接矩阵 CSV (只读一次, 不放进 Dataset)
    graph1 = load_csv(config.graph_path[0]).values[:, 1:]
    graph2 = load_csv(config.graph_path[1]).values[:, 1:]
    graph3 = load_csv(config.graph_path[2]).values[:, 1:].astype(float)
    # => [3, 18, 18]
    adj_tensor = torch.tensor(np.stack([graph1, graph2, graph3]), dtype=torch.float32)

    # 3. 标准化 (示例: 只对 passenger_flow, passenger_flow_trends, timetable 做)
    passenger_flow_nor, _, _ = normalize_per_channel(passenger_flow)
    passenger_flow_trends_nor, _, _ = normalize_per_channel(passenger_flow_trends)
    timetable_nor, _, _ = normalize_per_channel(timetable_features)

    # 4. 不再做列拼接
    #    原先: passenger_combined_features = torch.cat([passenger_flow_trends_nor, timetable_features], dim=-1)
    #    现在我们保留它们分别： passenger_flow_trends_nor, timetable_nor

    # 5. 滑动窗口
    #    用 passenger_flow_trends_nor 当作特征 => feature_windows
    #    用 passenger_flow_nor 当作标签 => label_windows
    feature_windows, label_windows = create_sliding_window_samples(
        passenger_flow_trends_nor,  # 形状 [Time, 18]
        window_size,
        step_size,
        passenger_flow_nor         # 形状 [Time, 18]
    )
    # 同理，time_windows 使用 timetable_nor => [Time, 3]
    time_windows = create_sliding_window_samples(timetable_nor, window_size, step_size)
    # 说明：feature_windows => [num_windows, 1, window_size, 18]
    #       label_windows   => [num_windows, 18]
    #       time_windows    => [num_windows, 1, window_size, 3]

    # 6. 不再对 feature_windows, time_windows permute(0,3,1,2)，
    #    因为 create_sliding_window_samples 内部已处理好形状.

    # 7. 划分训练集/测试集
    num_samples = feature_windows.shape[0]
    train_size = int(num_samples * train_ratio)

    train_feature = feature_windows[:train_size]
    train_time = time_windows[:train_size]
    train_label = label_windows[:train_size]

    test_feature = feature_windows[train_size:]
    test_time = time_windows[train_size:]
    test_label = label_windows[train_size:]

    # 8. 分别创建 Dataset
    train_dataset = PassengerFlowDataset(
        feature_windows=train_feature,
        time_windows=train_time,
        label_windows=train_label,
        device=Device
    )
    test_dataset = PassengerFlowDataset(
        feature_windows=test_feature,
        time_windows=test_time,
        label_windows=test_label,
        device=Device
    )

    # 9. 构建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # 丢掉最后那个不足 batch_size 的小 batch
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last = True  # 丢掉最后那个不足 batch_size 的小 batch
    )

    # 10. 返回 train_loader, test_loader, 以及单独的邻接矩阵 adj_tensor
    return train_loader, test_loader, adj_tensor

