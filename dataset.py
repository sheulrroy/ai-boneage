# ===================================================================
#  请用这段完整的代码，彻底替换掉你 dataset.py 里的所有内容
# ===================================================================

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Union, Literal
from pathlib import Path
import json
from sklearn.model_selection import train_test_split # 需要导入这个

# --- 这是我们修改过的、能处理scaler的MedicalImageDataset ---
# class MedicalImageDataset(Dataset):
#     """Custom Dataset for loading medical images. (最终优化版)"""
#     def __init__(
#         self,
#         df: pd.DataFrame, # 直接接收一个处理好的DataFrame
#         image_dir: str,
#         task_type: Literal['classification', 'regression'],
#         transform=None
#     ):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.task_type = task_type
        
#         # 直接使用传入的DataFrame，它应该已经包含了标准化的'y'列
#         self.labels_df = df.copy() 
#         self.labels_df['id'] = self.labels_df['id'].astype(str)
#         self.labels_df.set_index('id', inplace=True)
        
#         self.image_files = []
#         self.labels = []
        
#         for filename in os.listdir(image_dir):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#                 img_id = os.path.splitext(filename)[0]
#                 if img_id in self.labels_df.index:
#                     self.image_files.append(filename)
#                     self.labels.append(self.labels_df.loc[img_id, 'y'])
        
#     def __len__(self) -> int:
#         return len(self.image_files)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float]]:
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         image = Image.open(img_path).convert('L')
        
#         if self.transform:
#             image = self.transform(image)
            
#         label = self.labels[idx]
        
#         if self.task_type == 'classification':
#             label = torch.tensor(label, dtype=torch.long)
#         else:
#             label = torch.tensor(label, dtype=torch.float32)
            
#         return image, label
# =======================================================
#  dataset.py -> MedicalImageDataset 的最终性能优化版
# =======================================================
from tqdm import tqdm # 我们需要tqdm来显示加载进度

class MedicalImageDataset(Dataset):
    """Custom Dataset for loading medical images. (最终性能优化版)"""
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        task_type: Literal['classification', 'regression'],
        transform=None
    ):
        self.transform = transform
        self.task_type = task_type
        
        # 【核心修改】我们不再只是记录文件名，而是直接把图片加载到内存
        self.images = []
        self.labels = []

        print(f"开始预加载 {len(df)} 张图片到内存中，请稍候...")
        # 使用tqdm来显示加载进度
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading Images"):
            img_id = str(row['id'])
            # 如果id没有后缀，我们手动加上
            if not img_id.endswith('.png'):
                img_id += '.png'
            
            img_path = os.path.join(image_dir, img_id)
            
            if os.path.exists(img_path):
                # 打开图片并转换为灰度图，然后直接存到列表里
                image = Image.open(img_path).convert('L')
                self.images.append(image)
                
                # 同时保存对应的标签
                self.labels.append(row['y'])
        
        print(f"✅ 所有图片和标签已成功加载到内存！共 {len(self.images)} 张。")

    def __len__(self) -> int:
        return len(self.images) # 现在的长度是基于内存里的图片数量
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, float]]:
        # 【核心修改】直接从内存列表中获取图片，而不是从硬盘读取
        image = self.images[idx]
        label = self.labels[idx]
        
        # 后续的transform和tensor转换不变
        if self.transform:
            image = self.transform(image)
            
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
            
        return image, label

# --- ECGDataset 和 VoiceDataset 保持不变 ---
class ECGDataset(Dataset):
    # ... (这里面的代码和你原来的一样，保持不变) ...
    def __init__(self, data_dir: str, labels_file: str, task_type: Literal['classification', 'regression'], transform=None):
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['id'] = self.labels_df['id'].astype(str)
        self.labels_df.set_index('id', inplace=True)
        self.ecg_files = []
        self.labels = []
        for filename in self.data_dir.glob('*.npy'):
            ecg_id = filename.stem
            if ecg_id in self.labels_df.index:
                self.ecg_files.append(filename)
                self.labels.append(self.labels_df.loc[ecg_id, 'y'])
    def __len__(self): return len(self.ecg_files)
    def __getitem__(self, idx):
        ecg_path = self.ecg_files[idx]
        ecg_signal = np.load(ecg_path)
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)
        if self.transform: ecg_signal = self.transform(ecg_signal)
        label = self.labels[idx]
        if self.task_type == 'classification': label = torch.tensor(label, dtype=torch.long)
        else: label = torch.tensor(label, dtype=torch.float32)
        return ecg_signal, label

class VoiceDataset(Dataset):
    # ... (这里面的代码和你原来的一样，保持不变) ...
    def __init__(self, data_dir: str, labels_file: str, task_type: Literal['classification', 'regression'], transform=None, target_length: int=5000):
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.target_length = target_length
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df['id'] = self.labels_df['id'].astype(str)
        self.labels_df.set_index('id', inplace=True)
        self.voice_files = []
        self.labels = []
        for filename in self.data_dir.glob('*.npy'):
            voice_id = filename.stem
            if voice_id in self.labels_df.index:
                self.voice_files.append(filename)
                self.labels.append(self.labels_df.loc[voice_id, 'y'])
    def __len__(self): return len(self.voice_files)
    def __getitem__(self, idx):
        voice_path = self.voice_files[idx]
        voice_signal = np.load(voice_path)
        if len(voice_signal) != self.target_length:
            original_length = len(voice_signal)
            indices = np.linspace(0, original_length - 1, self.target_length, dtype=int)
            voice_signal = voice_signal[indices]
        voice_signal = torch.tensor(voice_signal, dtype=torch.float32).unsqueeze(0)
        if self.transform: voice_signal = self.transform(voice_signal)
        label = self.labels[idx]
        if self.task_type == 'classification': label = torch.tensor(label, dtype=torch.long)
        else: label = torch.tensor(label, dtype=torch.float32)
        return voice_signal, label

# --- get_transforms, get_ecg_transforms, get_voice_transforms 保持不变 ---
def get_transforms():
    # ... (代码和你原来的一样) ...
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
    return {'train': train_transform, 'val': val_transform, 'test': val_transform}

def get_ecg_transforms():
    # ... (代码和你原来的一样) ...
    return {'train': torch.nn.Identity(), 'val': torch.nn.Identity(), 'test': torch.nn.Identity()}

def get_voice_transforms():
    # ... (代码和你原来的一样) ...
    return {'train': torch.nn.Identity(), 'val': torch.nn.Identity(), 'test': torch.nn.Identity()}

# --- 这是我们修改过的、能处理scaler的create_data_loaders ---
def create_data_loaders(
    data_dir: str,
    labels_file: str,
    task_type: Literal['classification', 'regression'],
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    random_seed: int = 42,
    scaler=None # <--- 接收scaler
) -> Dict[str, DataLoader]:
    """Create train, validation and test data loaders. (最终优化版)"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 1. 在函数内部加载并处理DataFrame
    df = pd.read_csv(labels_file)
    if 'boneage' in df.columns:
        df = df.rename(columns={'boneage': 'y'})
    
    # 2. 如果传入了scaler，就在这里直接对整个'y'列进行标准化
    if scaler and 'y' in df.columns:
        df['y'] = scaler.transform(df['y'].values.reshape(-1, 1))

    # 3. 分割处理好的DataFrame
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_seed)
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_seed)

    # 4. 获取图像变换
    transforms_dict = get_transforms()

    # 5. 用分割好的df创建Dataset实例
    train_dataset = MedicalImageDataset(df=train_df, image_dir=data_dir, task_type=task_type, transform=transforms_dict['train'])
    val_dataset = MedicalImageDataset(df=val_df, image_dir=data_dir, task_type=task_type, transform=transforms_dict['val'])
    test_dataset = MedicalImageDataset(df=test_df, image_dir=data_dir, task_type=task_type, transform=transforms_dict['test'])
    
    # 6. 创建DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

# --- create_ecg_data_loaders 和 create_voice_data_loaders 保持不变 ---
def create_ecg_data_loaders(#...
    # ... (代码和你原来的一样) ...
    data_dir: str, labels_file: str, task_type: Literal['classification', 'regression'], batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, num_workers: int = 4, random_seed: int = 42) -> Dict[str, DataLoader]:
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    dataset = ECGDataset(data_dir, labels_file, task_type)
    total_size = len(dataset); train_size = int(train_ratio * total_size); val_size = int(val_ratio * total_size); test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def create_voice_data_loaders(#...
    # ... (代码和你原来的一样) ...
    data_dir: str, labels_file: str, task_type: Literal['classification', 'regression'], batch_size: int = 32, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, num_workers: int = 4, random_seed: int = 42, target_length: int = 5000) -> Dict[str, DataLoader]:
    torch.manual_seed(random_seed); np.random.seed(random_seed)
    dataset = VoiceDataset(data_dir, labels_file, task_type, transform=get_voice_transforms()['train'], target_length=target_length)
    total_size = len(dataset); train_size = int(train_ratio * total_size); val_size = int(val_ratio * total_size); test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

# --- save_dataset_info 保持不变 ---
def save_dataset_info(#...
    # ... (代码和你原来的一样) ...
    data_loaders: Dict[str, DataLoader], task_type: Literal['classification', 'regression'], save_path: str) -> None:
    info = {'task_type': task_type, 'train_size': len(data_loaders['train'].dataset), 'val_size': len(data_loaders['val'].dataset), 'test_size': len(data_loaders['test'].dataset), 'batch_size': data_loaders['train'].batch_size}
    with open(save_path, 'w') as f: json.dump(info, f, indent=4)