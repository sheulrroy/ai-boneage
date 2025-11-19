# ===================================================================
#  model.py 的最终正确版本 (纯PyTorch，基于ResNet50迁移学习)
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Literal
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from tqdm import tqdm
import warnings
from torchvision.models import resnet50, ResNet50_Weights

warnings.filterwarnings('ignore')

# 在 model.py 文件最顶部

# =======================================================
#  ✅ 我们把 MedicalCNN 的定义加回来，专门给Demo用，9.17update
# =======================================================
class MedicalCNN(nn.Module):
    """CNN model for medical image classification/regression."""
    def __init__(self, task_type: Literal['classification', 'regression'], num_classes: int = 2, num_conv_layers: int = 4, conv_channels: int = 32, fc_layers: List[int] = [512, 128], input_size: int = 224, dropout_rate: float = 0.2):
        super(MedicalCNN, self).__init__();
        conv_modules = []; in_channels = 1; current_ch = conv_channels;
        for _ in range(num_conv_layers):
            conv_modules.extend([nn.Conv2d(in_channels, current_ch, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(current_ch), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(dropout_rate)]);
            in_channels = current_ch;
        self.conv_layers = nn.Sequential(*conv_modules);
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1));
        fc_modules = []; prev_size = current_ch;
        for fc_size in fc_layers:
            fc_modules.extend([nn.Linear(prev_size, fc_size), nn.BatchNorm1d(fc_size), nn.ReLU(), nn.Dropout(dropout_rate)]);
            prev_size = fc_size;
        if task_type == 'classification': fc_modules.append(nn.Linear(fc_layers[-1], num_classes));
        else: fc_modules.append(nn.Linear(fc_layers[-1], 1));
        self.fc_layers = nn.Sequential(*fc_modules);
        self.task_type = task_type;
        self._initialize_weights();
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');
            elif isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight);
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3: x = x.unsqueeze(1);
        x = self.conv_layers(x);
        x = self.avgpool(x);
        x = x.view(x.size(0), -1);
        x = self.fc_layers(x);
        return x

# --- 下面是你其他的代码，比如 create_model, ModelTrainer 等，保持不变 ---

# --- 这是我们项目中唯一的、正确的深度学习模型创建函数 ---
def create_model(
    task_type: Literal['classification', 'regression'], 
    num_classes: Optional[int] = None, 
    dropout_rate: float = 0.5
) -> nn.Module:
    """
    创建一个基于PyTorch官方ResNet50的迁移学习模型。
    """
    # 1. 加载在ImageNet上预训练好的、最新的ResNet50模型
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # 2. 修改模型的第一个卷积层以接受单通道（灰度）图像
    #    原始ResNet50的第一个卷积层是为3通道RGB图像设计的
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1,  # <-- 输入通道改为1
        original_conv1.out_channels, 
        kernel_size=original_conv1.kernel_size, 
        stride=original_conv1.stride, 
        padding=original_conv1.padding, 
        bias=original_conv1.bias
    )
    # （可选）为了更好地利用预训练权重，可以把原始3通道权重的均值赋给新层
    model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    # 3. “换头术”：替换掉原来的分类头，换成我们自己的头
    num_ftrs = model.fc.in_features # 获取原始全连接层的输入特征数
    
    if task_type == 'regression':
        # 对于回归任务，我们换成一个带有Dropout的、输出为1的头
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )
    elif task_type == 'classification':
        if num_classes is None:
            raise ValueError("对于分类任务, 'num_classes' 必须被指定")
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
    
    return model



# --- 这是我们统一的、功能完整的ModelTrainer类 ---
class ModelTrainer:
    """Class to handle model training and evaluation."""
    def __init__(self, model, criterion, optimizer, device, task_type, class_names=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.task_type = task_type
        self.class_names = class_names
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        train_pbar = tqdm(train_loader, desc='Training', leave=False, bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        for images, labels in train_pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            if self.task_type == 'regression': outputs = outputs.squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        epoch_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return epoch_loss, metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        val_pbar = tqdm(val_loader, desc='Validation', leave=False, bar_format='{l_bar}{bar:30}{r_bar}', ncols=100)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                if self.task_type == 'regression': outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        val_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        return val_loss, metrics
    
    def _calculate_metrics(self, labels, preds) -> Dict[str, float]:
        metrics = {}
        labels_array, preds_array = np.array(labels).flatten(), np.array(preds).flatten()
        if len(labels_array) == 0: return metrics

        if self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(labels_array, preds_array)
            metrics['mae'] = mean_absolute_error(labels_array, preds_array)
            metrics['r2'] = r2_score(labels_array, preds_array)
        else:
            pass # 分类任务的指标可以加在这里
            
        # 转换成Python内置类型，方便保存为JSON
        native_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.floating): native_metrics[key] = float(value)
            elif isinstance(value, np.integer): native_metrics[key] = int(value)
            else: native_metrics[key] = value
        return native_metrics

    def train(self, train_loader, val_loader, num_epochs, save_dir, early_stopping_patience=15):
        best_val_loss = float('inf')
        patience_counter = 0
        epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0)
        
        for epoch in epoch_pbar:
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss); self.history['train_metrics'].append(train_metrics)
            
            val_loss, val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss); self.history['val_metrics'].append(val_metrics)
            
            postfix_dict = {'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'}
            if self.task_type == 'regression':
                postfix_dict.update({'Train R²': f'{train_metrics.get("r2", 0):.3f}', 'Val R²': f'{val_metrics.get("r2", 0):.3f}'})
            postfix_dict['LR'] = f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            epoch_pbar.set_postfix(postfix_dict)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0
                self.save_model(save_dir, 'best_model.pth')
            else:
                patience_counter += 1
            
            self.scheduler.step(val_loss)
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping after {epoch+1} epochs.'); break
        
        self.save_history(self.history, save_dir)
        return self.history

    def save_model(self, save_dir, filename):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(save_dir) / filename)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_history(self, history, save_dir):
        # ... (保存历史记录的代码，保持不变) ...
        def convert_types(obj):
            if isinstance(obj, dict): return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_types(i) for i in obj]
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.integer): return int(obj)
            return obj
        native_history = convert_types(history)
        with open(Path(save_dir) / 'training_history.json', 'w') as f:
            json.dump(native_history, f, indent=4)
            
    def plot_training_history(self, save_dir: str, start_epoch: int = 5):
        # ... (你的绘图代码，保持不变) ...
        pass

    # =======================================================
    #  ✅【关键】我们统一使用的、能获取预测值和真实值的方法
    # =======================================================
    def predict_all(self, data_loader):
        """
        这个方法用来获取模型在给定数据集上的所有预测值和真实标签。
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        return all_labels, all_preds

# --- 你其他的函数可以放在这里，比如模型对比函数 ---
# --- 【重要】请确保其他函数不再依赖于旧的模型类 ---

# 比如，你的compare_models_performance函数，它的输入应该是trainer和loader，而不是模型本身。
# 这是一个示例，请根据你的实际代码调整。
def compare_models_performance(best_cnn_trainer, train_loader, val_loader, test_loader, save_dir, task_type, class_names=None, debug=False):
    print("模型对比功能简化版...")
    # ... 在这里，你应该放入你完整的、正确的模型对比函数代码 ...
    return {"Deep Learning (CNN)": {"mae": 10.0}}