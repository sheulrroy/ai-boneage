# ===================================================================
#  hyperparameter_tuning.py 的最终正确版本
#  它现在可以为任何你传入的模型创建函数进行调优
# ===================================================================

import itertools
from typing import Dict, List, Any, Tuple, Callable, Union, Literal 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
# 【关键修改】我们现在需要从model.py里导入ModelTrainer和我们唯一的create_model函数
from model import create_model, ModelTrainer 
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Universal class for performing grid search. (最终优化版)"""
    def __init__(
        self,
        model_creator: callable, # <--- 接收一个能创建模型的【函数】
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_type: Literal['classification', 'regression'],
        num_classes: int = 2,
        device: torch.device = None,
        save_dir: str = './grid_search_results'
    ):
        self.model_creator = model_creator # <--- 保存这个函数
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        
    def _create_model_and_optimizer(self, params: Dict[str, Any]) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        """Create model, criterion and optimizer with given parameters."""
        
        # 【核心修改】调用传入的model_creator来创建模型
        # 我们把模型自己可调的参数（如dropout_rate）从params里提取出来传给它
        model = self.model_creator(
            task_type=self.task_type,
            num_classes=self.num_classes,
            dropout_rate=params.get('dropout_rate', 0.5) # 从参数网格获取dropout，如果找不到就用默认值0.5
        )
        
        if self.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # 从params里获取学习率来创建优化器
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4) # 默认加上一点权重衰减
            
        return model, criterion, optimizer
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        num_epochs: int,
        early_stopping_patience: int
    ) -> Dict[str, Any]:
        """Perform grid search over hyperparameters."""
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"开始Grid Search，共 {len(param_combinations)} 种组合...")
        
        for i, params in enumerate(param_combinations):
            print(f"\n--- 正在尝试组合 {i+1}/{len(param_combinations)}:")
            print(json.dumps(params, indent=2))
            
            # 创建模型和训练器
            model, criterion, optimizer = self._create_model_and_optimizer(params)
            trainer = ModelTrainer(
                model=model, criterion=criterion, optimizer=optimizer,
                device=self.device, task_type=self.task_type
            )
            
            # 训练模型
            history = trainer.train(
                train_loader=self.train_loader, val_loader=self.val_loader,
                num_epochs=num_epochs,
                save_dir=self.save_dir / f"combination_{i+1}",
                early_stopping_patience=early_stopping_patience
            )
            
            # 获取最佳验证指标
            if history['val_loss']:
                best_val_loss = min(history['val_loss'])
                best_val_metrics = history['val_metrics'][np.argmin(history['val_loss'])]
            else:
                best_val_loss = float('inf')
                best_val_metrics = {}

            # 存储结果
            self.results.append({
                'params': params,
                'best_val_loss': best_val_loss,
                'best_val_metrics': best_val_metrics
            })
        
        if not self.results:
            print("警告：Grid Search 未产生任何结果。")
            return {}

        # 找到最佳组合
        best_result = min(self.results, key=lambda x: x['best_val_loss'])
        summary = {
            'best_combination': best_result['params'],
            'best_val_loss': best_result['best_val_loss'],
            'best_val_metrics': best_result['best_val_metrics']
        }
        
        # 保存总结
        with open(self.save_dir / 'grid_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary
    
    # ... 你原来的 plot_results, get_results_dataframe, get_best_model 等辅助函数可以保持不变 ...
    # ... 我把它们也复制过来，确保文件的完整性 ...
    def plot_results(self, metric: str = 'best_val_loss'):
        df = self.get_results_dataframe()
        param_cols = list(df.columns)
        param_cols.remove(metric) # Remove the main metric from param list
        # Remove other metrics from the param list
        other_metrics = [c for c in param_cols if c.startswith('best_val_')]
        for m in other_metrics: param_cols.remove(m)

        n_cols = len(param_cols)
        if n_cols == 0:
            print("No parameters to plot against.")
            return

        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5), squeeze=False)
        axes = axes.flatten()

        for ax, param in zip(axes, param_cols):
            sns.boxplot(data=df, x=param, y=metric, ax=ax)
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'grid_search_boxplot_results.png')
        plt.close()

    def get_results_dataframe(self) -> pd.DataFrame:
        flattened_results = []
        for result in self.results:
            flat_result = {**result['params'], 'best_val_loss': result['best_val_loss'], **{f'best_val_{k}': v for k, v in result['best_val_metrics'].items()}}
            flattened_results.append(flat_result)
        return pd.DataFrame(flattened_results)