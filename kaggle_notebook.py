import os
import math
import json
import csv
import warnings
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# ============================================================================
# EVALUATION AND METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'confusion_matrix': cm
    }
    
    if verbose and task_name:
        print(f"\n=== {task_name} Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision_avg:.4f}")
        print(f" Recall: {recall_avg:.4f}")
        print(f"F1: {f1_avg:.4f}")
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_name = "HC" if label == 0 else ("PD" if label == 1 else f"Class_{label}")
                if task_name == "PD vs DD":
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics

def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                         fold_metrics_hc, fold_metrics_pd):
    # Save HC vs PD metrics to CSV
    if fold_metrics_hc:
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.csv"
        hc_rows = []

        # Add header information
        hc_rows.append({
            'Fold': fold_idx + 1 if fold_idx is not None else 'N/A',
            'Best_Epoch': best_epoch,
            'Best_Combined_Accuracy': f"{best_val_acc:.4f}",
            'Task': 'HC vs PD'
        })

        # Create detailed metrics DataFrame
        detailed_rows = []
        for epoch_data in fold_metrics_hc:
            epoch = epoch_data['epoch']
            metrics = epoch_data['metrics']
            labels = epoch_data['labels']
            predictions = epoch_data['predictions']

            # Overall metrics
            detailed_rows.append({
                'Epoch': epoch,
                'Metric_Type': 'Overall',
                'Class': 'All',
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision_avg', 0):.4f}",
                'Recall': f"{metrics.get('recall_avg', 0):.4f}",
                'F1_Score': f"{metrics.get('f1_avg', 0):.4f}",
                'Support': len(labels)
            })

            # Per-class metrics
            if 'precision_per_class' in metrics:
                class_names = ['HC', 'PD']
                for i, class_name in enumerate(class_names):
                    if i < len(metrics['precision_per_class']):
                        detailed_rows.append({
                            'Epoch': epoch,
                            'Metric_Type': 'Per_Class',
                            'Class': class_name,
                            'Accuracy': '',
                            'Precision': f"{metrics['precision_per_class'][i]:.4f}",
                            'Recall': f"{metrics['recall_per_class'][i]:.4f}",
                            'F1_Score': f"{metrics['f1_per_class'][i]:.4f}",
                            'Support': int(metrics['support_per_class'][i])
                        })

            # Confusion matrix
            if len(labels) > 0:
                cm = confusion_matrix(labels, predictions)
                for i, row in enumerate(cm):
                    class_name = 'HC' if i == 0 else 'PD'
                    cm_row = {
                        'Epoch': epoch,
                        'Metric_Type': 'Confusion_Matrix',
                        'Class': class_name,
                        'Pred_HC': int(row[0]) if len(row) > 0 else 0,
                        'Pred_PD': int(row[1]) if len(row) > 1 else 0
                    }
                    detailed_rows.append(cm_row)

        # Save to CSV
        pd.DataFrame(detailed_rows).to_csv(hc_filename, index=False)
        print(f"✓ HC vs PD metrics saved to CSV: {hc_filename}")

    # Save PD vs DD metrics to CSV
    if fold_metrics_pd:
        pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.csv"

        # Create detailed metrics DataFrame
        detailed_rows = []
        for epoch_data in fold_metrics_pd:
            epoch = epoch_data['epoch']
            metrics = epoch_data['metrics']
            labels = epoch_data['labels']
            predictions = epoch_data['predictions']

            # Overall metrics
            detailed_rows.append({
                'Epoch': epoch,
                'Metric_Type': 'Overall',
                'Class': 'All',
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision_avg', 0):.4f}",
                'Recall': f"{metrics.get('recall_avg', 0):.4f}",
                'F1_Score': f"{metrics.get('f1_avg', 0):.4f}",
                'Support': len(labels)
            })

            # Per-class metrics
            if 'precision_per_class' in metrics:
                class_names = ['PD', 'DD']
                for i, class_name in enumerate(class_names):
                    if i < len(metrics['precision_per_class']):
                        detailed_rows.append({
                            'Epoch': epoch,
                            'Metric_Type': 'Per_Class',
                            'Class': class_name,
                            'Accuracy': '',
                            'Precision': f"{metrics['precision_per_class'][i]:.4f}",
                            'Recall': f"{metrics['recall_per_class'][i]:.4f}",
                            'F1_Score': f"{metrics['f1_per_class'][i]:.4f}",
                            'Support': int(metrics['support_per_class'][i])
                        })

            # Confusion matrix
            if len(labels) > 0:
                cm = confusion_matrix(labels, predictions)
                for i, row in enumerate(cm):
                    class_name = 'PD' if i == 0 else 'DD'
                    cm_row = {
                        'Epoch': epoch,
                        'Metric_Type': 'Confusion_Matrix',
                        'Class': class_name,
                        'Pred_PD': int(row[0]) if len(row) > 0 else 0,
                        'Pred_DD': int(row[1]) if len(row) > 1 else 0
                    }
                    detailed_rows.append(cm_row)

        # Save to CSV
        pd.DataFrame(detailed_rows).to_csv(pd_filename, index=False)
        print(f"✓ PD vs DD metrics saved to CSV: {pd_filename}")


def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir="plots"):
    
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform t-SNE dimensionality reduction
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
   
    ax1 = axes[0]
    valid_hc_pd = hc_pd_labels != -1
    if np.any(valid_hc_pd):
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        

        colors_hc_pd = ['blue' if l == 0 else 'red' for l in labels_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        if np.any(hc_mask):
            ax1.scatter(features_hc_pd[hc_mask, 0], features_hc_pd[hc_mask, 1], 
                       c='blue', label='HC', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        pd_mask = labels_hc_pd == 1
        if np.any(pd_mask):
            ax1.scatter(features_hc_pd[pd_mask, 0], features_hc_pd[pd_mask, 1], 
                       c='red', label='PD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('t-SNE: HC vs PD Classification', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1', fontsize=12)
        ax1.set_ylabel('t-SNE Component 2', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        n_hc = np.sum(hc_mask)
        n_pd = np.sum(pd_mask)
        ax1.text(0.02, 0.98, f'HC: {n_hc}\nPD: {n_pd}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No HC vs PD data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('t-SNE: HC vs PD Classification', fontsize=14, fontweight='bold')
    

    ax2 = axes[1]
    valid_pd_dd = pd_dd_labels != -1
    if np.any(valid_pd_dd):
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]
        
    
        colors_pd_dd = ['green' if l == 0 else 'orange' for l in labels_pd_dd]
        
        pd_mask = labels_pd_dd == 0
        if np.any(pd_mask):
            ax2.scatter(features_pd_dd[pd_mask, 0], features_pd_dd[pd_mask, 1], 
                       c='green', label='PD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
    
        dd_mask = labels_pd_dd == 1
        if np.any(dd_mask):
            ax2.scatter(features_pd_dd[dd_mask, 0], features_pd_dd[dd_mask, 1], 
                       c='orange', label='DD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('t-SNE: PD vs DD Classification', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1', fontsize=12)
        ax2.set_ylabel('t-SNE Component 2', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        n_pd = np.sum(pd_mask)
        n_dd = np.sum(dd_mask)
        ax2.text(0.02, 0.98, f'PD: {n_pd}\nDD: {n_dd}', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No PD vs DD data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('t-SNE: PD vs DD Classification', fontsize=14, fontweight='bold')
    

    fig.suptitle('t-SNE Visualization of Feature Space', fontsize=16, fontweight='bold', y=1.02)
    
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {output_path}")
    
   
    if np.any(valid_hc_pd):
        plt.figure(figsize=(8, 6))
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        pd_mask = labels_hc_pd == 1
        
        if np.any(hc_mask):
            plt.scatter(features_hc_pd[hc_mask, 0], features_hc_pd[hc_mask, 1], 
                       c='blue', label=f'HC (n={np.sum(hc_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask, 0], features_hc_pd[pd_mask, 1], 
                       c='red', label=f'PD (n={np.sum(pd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE: Healthy Control vs Parkinson\'s Disease', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path_hc = os.path.join(output_dir, 'tsne_hc_vs_pd.png')
        plt.savefig(output_path_hc, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"HC vs PD t-SNE plot saved to {output_path_hc}")
    

    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]
        
        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask, 0], features_pd_dd[pd_mask, 1], 
                       c='green', label=f'PD (n={np.sum(pd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask, 0], features_pd_dd[dd_mask, 1], 
                       c='orange', label=f'DD (n={np.sum(dd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE: Parkinson\'s Disease vs Differential Diagnosis', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path_pd = os.path.join(output_dir, 'tsne_pd_vs_dd.png')
        plt.savefig(output_path_pd, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PD vs DD t-SNE plot saved to {output_path_pd}")
    
    return features_2d

# ============================================================================
# Model
# ============================================================================

class TextTokenizer(nn.Module):
    
    def __init__(self, model_name='bert-base-uncased', output_dim=128, dropout=0.1):
        super().__init__()
        
        self.model_name = model_name
        self.bert = BertModel.from_pretrained(model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        input_dim = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, text_list, device):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        tokens = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        if self.training:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        output = outputs.pooler_output
        
        return self.projection(output)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0) 

class MultiheadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads
        
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, model_dim = query.size()
        seq_len_k = key.size(1)
    
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_w = F.softmax(scores, dim=-1)
        attention_w = self.dropout(attention_w)
        
        context = torch.matmul(attention_w, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.model_dim)
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output, attention_w


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)  # Residual connection
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = MultiheadAttention(model_dim, num_heads, dropout)
        self.cross_attention_2to1 = MultiheadAttention(model_dim, num_heads, dropout)
        self.self_attention_1 = MultiheadAttention(model_dim, num_heads, dropout)
        self.self_attention_2 = MultiheadAttention(model_dim, num_heads, dropout)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        # Cross attention
        channel_1_cross, _ = self.cross_attention_1to2(query=channel_1, key=channel_2, value=channel_2)
        channel_2_cross, _ = self.cross_attention_2to1(query=channel_2, key=channel_1, value=channel_1)
        
        # Self attention
        channel_1_self, _ = self.self_attention_1(query=channel_1_cross, key=channel_1_cross, value=channel_1_cross)
        channel_2_self, _ = self.self_attention_2(query=channel_2_cross, key=channel_2_cross, value=channel_2_cross)
        
        # Feed forward
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out


class Model(nn.Module):
    """
    - Level 1 (Window-level): Cross-attention between left and right wrist for each window
    - Level 2 (Task-level): Attention across windows within each task, with task embeddings
    """
    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 256,
        num_classes: int = 2,
        num_tasks: int = 10,  # Number of different tasks
        use_text: bool = True,
        text_encoder_dim: int = 128,
        fusion_method: str = 'concat',
    ):
        super().__init__()

        self.model_dim = model_dim
        self.seq_len = seq_len
        self.use_text = use_text
        self.fusion_method = fusion_method
        self.num_tasks = num_tasks

        # ===== Window-level components (Level 1) =====
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)

        self.positional_encoding = PositionalEncoding(model_dim, max_len=seq_len)

        # Window-level cross-attention layers
        self.window_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ===== Task-level components (Level 2) =====
        # Task number embeddings (learnable)
        self.task_embedding = nn.Embedding(num_tasks, model_dim * 2)

        # Task-level attention module
        self.task_attention = nn.MultiheadAttention(
            embed_dim=model_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.task_layer_norm = nn.LayerNorm(model_dim * 2)
        self.task_feed_forward = FeedForward(model_dim * 2, d_ff, dropout)

        # ===== Text encoder =====
        if use_text:
            self.text_encoder = TextTokenizer(output_dim=text_encoder_dim, dropout=dropout)

            if fusion_method == 'concat':
                fusion_dim = model_dim * 2 + text_encoder_dim
            elif fusion_method == 'attention':
                fusion_dim = model_dim * 2
                self.fusion_attention = nn.MultiheadAttention(
                    embed_dim=model_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.text_to_signal = nn.Linear(text_encoder_dim, model_dim * 2)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            fusion_dim = model_dim * 2

        # ===== Classification heads =====
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: HC vs PD
        )

        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: PD vs DD
        )

        self.dropout = nn.Dropout(dropout)

    def process_window(self, left_wrist, right_wrist):
        """
        Process a single window through window-level cross-attention.

        Args:
            left_wrist: (batch_size, seq_len, input_dim)
            right_wrist: (batch_size, seq_len, input_dim)

        Returns:
            window_embedding: (batch_size, model_dim * 2)
        """
        # Project to model dimension
        left_encoded = self.left_projection(left_wrist)
        right_encoded = self.right_projection(right_wrist)

        # Add positional encoding
        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)

        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)

        # Apply window-level cross-attention layers
        for layer in self.window_layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        # Global pooling
        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)

        # Concatenate left and right embeddings
        window_embedding = torch.cat([left_pool, right_pool], dim=1)

        return window_embedding

    def forward(self, batch_data, patient_texts=None, device=None):
        batch_size = len(batch_data)
        all_task_embeddings = []

        # Process each sample in the batch
        for sample in batch_data:
            left_windows = sample['left_windows'].to(device)  # (num_windows, seq_len, input_dim)
            right_windows = sample['right_windows'].to(device)
            task_ids = sample['task_ids'].to(device)  # (num_windows,)

            # ===== Level 1: Window-level processing =====
            # Process all windows through window-level cross-attention
            window_embeddings = self.process_window(left_windows, right_windows)  # (num_windows, model_dim * 2)

            # ===== Level 2: Task-level processing =====
            # Group windows by task and add task embeddings
            unique_tasks = torch.unique(task_ids)
            task_representations = []

            for task_id in unique_tasks:
                # Get windows for this task
                task_mask = task_ids == task_id
                task_windows = window_embeddings[task_mask]  # (num_task_windows, model_dim * 2)

                # Get task embedding
                task_emb = self.task_embedding(task_id.unsqueeze(0))  # (1, model_dim * 2)

                # Concatenate task embedding with window embeddings
                # Add task embedding to each window
                task_windows_with_emb = task_windows + task_emb  # Broadcasting

                # Average pool windows for this task
                task_rep = task_windows_with_emb.mean(dim=0, keepdim=True)  # (1, model_dim * 2)
                task_representations.append(task_rep)

            # Stack task representations
            task_sequence = torch.cat(task_representations, dim=0)  # (num_tasks, model_dim * 2)

            # Apply task-level attention
            task_attended, _ = self.task_attention(
                task_sequence.unsqueeze(0),
                task_sequence.unsqueeze(0),
                task_sequence.unsqueeze(0)
            )  # (1, num_tasks, model_dim * 2)

            task_attended = self.task_layer_norm(task_attended.squeeze(0) + task_sequence)
            task_attended = self.task_feed_forward(task_attended)  # (num_tasks, model_dim * 2)

            # Pool across tasks
            sample_embedding = task_attended.mean(dim=0)  # (model_dim * 2,)
            all_task_embeddings.append(sample_embedding)

        # Stack all sample embeddings
        fused_signal_features = torch.stack(all_task_embeddings, dim=0)  # (batch_size, model_dim * 2)

        # ===== Text fusion =====
        if self.use_text and patient_texts is not None:
            if device is None:
                device = fused_signal_features.device

            text_features = self.text_encoder(patient_texts, device)

            if self.fusion_method == 'concat':
                fused_features = torch.cat([fused_signal_features, text_features], dim=1)
            elif self.fusion_method == 'attention':
                text_transformed = self.text_to_signal(text_features).unsqueeze(1)
                signal_features = fused_signal_features.unsqueeze(1)

                fused_output, _ = self.fusion_attention(
                    query=signal_features,
                    key=text_transformed,
                    value=text_transformed
                )
                fused_features = fused_output.squeeze(1)
        else:
            fused_features = fused_signal_features

        # ===== Classification =====
        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)

        return logits_hc_vs_pd, logits_pd_vs_dd

    def get_features(self, batch_data, patient_texts=None, device=None):
        """Extract features for visualization."""
        batch_size = len(batch_data)
        all_task_embeddings = []

        # Process each sample in the batch
        for sample in batch_data:
            left_windows = sample['left_windows'].to(device)
            right_windows = sample['right_windows'].to(device)
            task_ids = sample['task_ids'].to(device)

            # Window-level processing
            window_embeddings = self.process_window(left_windows, right_windows)

            # Task-level processing
            unique_tasks = torch.unique(task_ids)
            task_representations = []

            for task_id in unique_tasks:
                task_mask = task_ids == task_id
                task_windows = window_embeddings[task_mask]
                task_emb = self.task_embedding(task_id.unsqueeze(0))
                task_windows_with_emb = task_windows + task_emb
                task_rep = task_windows_with_emb.mean(dim=0, keepdim=True)
                task_representations.append(task_rep)

            task_sequence = torch.cat(task_representations, dim=0)
            task_attended, _ = self.task_attention(
                task_sequence.unsqueeze(0),
                task_sequence.unsqueeze(0),
                task_sequence.unsqueeze(0)
            )
            task_attended = self.task_layer_norm(task_attended.squeeze(0) + task_sequence)
            task_attended = self.task_feed_forward(task_attended)
            sample_embedding = task_attended.mean(dim=0)
            all_task_embeddings.append(sample_embedding)

        fused_signal_features = torch.stack(all_task_embeddings, dim=0)

        # Text fusion
        if self.use_text and patient_texts is not None:
            if device is None:
                device = fused_signal_features.device

            text_features = self.text_encoder(patient_texts, device)

            if self.fusion_method == 'concat':
                fused_features = torch.cat([fused_signal_features, text_features], dim=1)
            elif self.fusion_method == 'attention':
                text_transformed = self.text_to_signal(text_features).unsqueeze(1)
                signal_features = fused_signal_features.unsqueeze(1)

                fused_output, _ = self.fusion_attention(
                    query=signal_features,
                    key=text_transformed,
                    value=text_transformed
                )
                fused_features = fused_output.squeeze(1)
        else:
            fused_features = fused_signal_features

        return fused_features


# ============================================================================
# Dataloader
# ============================================================================

###############Helper functions##########
def create_windows(data, window_size=256, overlap=0):
    
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))   
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])
    
    return np.array(windows) if windows else None


#down sampling 
def downsample(data, original_freq=100, target_freq=64):
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


# band pass filter
def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
    nyquist = 0.5 * original_freq
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


#prepare text 
def prepare_text(metadata, questionnaires):
    text_array = []
    
    if metadata:
        text_array.append(f"Age: {metadata.get('age', 'unknown')}")
        text_array.append(f"Gender: {metadata.get('gender', 'unknown')}")
        if metadata.get('age_at_diagnosis'):
            text_array.append(f"Age at diagnosis: {metadata.get('age_at_diagnosis')}")
        if metadata.get('disease_comment'):
            text_array.append(f"Clinical notes: {metadata.get('disease_comment')}")
    
    if questionnaires and 'item' in questionnaires:
        for item in questionnaires['item']:
            q_text = item.get('text', '')
            q_answer = item.get('answer', '')
            if q_text and q_answer:
                text_array.append(f"Q: {q_text} A: {q_answer}")
    
    return " ".join(text_array) if text_array else "No information available."


###############splitting methods################
def k_fold_split_method(data_root, full_dataset, k=5):
    patient_conditions = {}
    patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
    
    for patient_id in range(1, 470):
        patient_path = pathlib.Path(str(patients_template).format(p=patient_id))
        if patient_path.exists():
            try:
                with open(patient_path, 'r') as f:
                    condition = json.load(f).get('condition', 'Unknown')
                    patient_conditions[patient_id] = condition
            except:
                pass
            
    patient_list = []
    patient_labels = []
    for pid in sorted(patient_conditions.keys()):
        condition = patient_conditions[pid]
        if condition == 'Healthy':
            label = 0
        elif 'Parkinson' in condition:
            label = 1
        else:
            label = 2
        patient_list.append(pid)
        patient_labels.append(label)
    
    print(f"Total patients: {len(patient_list)} (HC={patient_labels.count(0)}, PD={patient_labels.count(1)}, DD={patient_labels.count(2)})")
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_datasets = []
    
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(patient_list, patient_labels)):
        train_patients = set([patient_list[i] for i in train_idx])
        test_patients = set([patient_list[i] for i in test_idx])
        
        train_mask = np.array([pid in train_patients for pid in full_dataset.patient_ids])
        test_mask = np.array([pid in test_patients for pid in full_dataset.patient_ids])
        
        train_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[train_mask],
            right_samples=full_dataset.right_samples[train_mask],
            hc_vs_pd=full_dataset.hc_vs_pd[train_mask],
            pd_vs_dd=full_dataset.pd_vs_dd[train_mask],
            patient_texts=[full_dataset.patient_texts[i] for i, m in enumerate(train_mask) if m],
            patient_ids=full_dataset.patient_ids[train_mask]
        )
        
        test_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[test_mask],
            right_samples=full_dataset.right_samples[test_mask],
            hc_vs_pd=full_dataset.hc_vs_pd[test_mask],
            pd_vs_dd=full_dataset.pd_vs_dd[test_mask],
            patient_texts=[full_dataset.patient_texts[i] for i, m in enumerate(test_mask) if m],
            patient_ids=full_dataset.patient_ids[test_mask]
        )
        
        # Print fold info
        train_hc = np.sum(train_dataset.hc_vs_pd == 0)
        train_pd = np.sum((train_dataset.hc_vs_pd == 1) & (train_dataset.pd_vs_dd == 0))
        train_dd = np.sum(train_dataset.pd_vs_dd == 1)
        test_hc = np.sum(test_dataset.hc_vs_pd == 0)
        test_pd = np.sum((test_dataset.hc_vs_pd == 1) & (test_dataset.pd_vs_dd == 0))
        test_dd = np.sum(test_dataset.pd_vs_dd == 1)
        
        print(f"\nFold {fold_id+1}/{k}:")
        print(f"  Train: {len(train_dataset)} samples (HC={train_hc}, PD={train_pd}, DD={train_dd})")
        print(f"  Test:  {len(test_dataset)} samples (HC={test_hc}, PD={test_pd}, DD={test_dd})")
        
        fold_datasets.append((train_dataset, test_dataset))
    
    return fold_datasets


def patient_level_split_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, 
                               patient_texts, patient_ids, split_ratio=0.85):
    """
    Split data at patient level using stratified sampling to maintain class balance.
    """
    # Get unique patients and their labels
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    
    for pid in unique_patients:
        # Get the label for this patient (all samples from same patient have same label)
        patient_mask = patient_ids == pid
        hc_vs_pd_label = hc_vs_pd[patient_mask][0]
        pd_vs_dd_label = pd_vs_dd[patient_mask][0]
        
        if hc_vs_pd_label == 0:
            label = 0  # Healthy
        elif hc_vs_pd_label == 1 and pd_vs_dd_label == 0:
            label = 1  # Parkinson's
        else:
            label = 2  # Other disorders
        
        patient_labels.append(label)
    
    patient_labels = np.array(patient_labels)
    
    train_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=(1 - split_ratio),
        stratify=patient_labels,
        random_state=42
    )
    
    train_patients = set(train_patients)
    test_patients = set(test_patients)
    
    train_mask = np.array([pid in train_patients for pid in patient_ids])
    test_mask = np.array([pid in test_patients for pid in patient_ids])
    
    train_data = {
        'left': left_samples[train_mask],
        'right': right_samples[train_mask],
        'hc_vs_pd': hc_vs_pd[train_mask],
        'pd_vs_dd': pd_vs_dd[train_mask],
        'texts': [patient_texts[i] for i, m in enumerate(train_mask) if m],
        'patient_ids': patient_ids[train_mask]
    }
    
    test_data = {
        'left': left_samples[test_mask],
        'right': right_samples[test_mask],
        'hc_vs_pd': hc_vs_pd[test_mask],
        'pd_vs_dd': pd_vs_dd[test_mask],
        'texts': [patient_texts[i] for i, m in enumerate(test_mask) if m],
        'patient_ids': patient_ids[test_mask]
    }
    
    # Print split info
    print(f"\nPatient-level split:")
    print(f"  Train: {len(train_patients)} patients, {len(train_data['left'])} samples")
    print(f"  Test: {len(test_patients)} patients, {len(test_data['left'])} samples")
    
    return train_data, test_data


def task_wise_split_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, 
                           patient_texts, task_names, patient_ids=None, train_tasks=None):
    train_indices = []
    test_indices = []
    
    for idx, task in enumerate(task_names):
        if task in train_tasks:
            train_indices.append(idx)
        else:
            test_indices.append(idx)
    
    train_data = {
        'left': left_samples[train_indices],
        'right': right_samples[train_indices],
        'hc_vs_pd': hc_vs_pd[train_indices],
        'pd_vs_dd': pd_vs_dd[train_indices],
        'texts': [patient_texts[i] for i in train_indices],
        'patient_ids': patient_ids[train_indices] if patient_ids is not None else None
    }
    
    test_data = {
        'left': left_samples[test_indices],
        'right': right_samples[test_indices],
        'hc_vs_pd': hc_vs_pd[test_indices],
        'pd_vs_dd': pd_vs_dd[test_indices],
        'texts': [patient_texts[i] for i in test_indices],
        'patient_ids': patient_ids[test_indices] if patient_ids is not None else None
    }
    
    return train_data, test_data

class ParkinsonsDataLoader(Dataset):
    """
    Hierarchical data loader that groups windows by patient and task.

    Each sample represents one patient with all their windows grouped by task.
    """
    def __init__(self, data_root: str = None, window_size: int = 256,
                 left_samples=None, right_samples=None,
                 hc_vs_pd=None, pd_vs_dd=None, patient_texts=None,
                 patient_ids=None, task_names=None,
                 apply_dowsampling=True,
                 apply_bandpass_filter=True, apply_prepare_text=True):

        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.apply_prepare_text = apply_prepare_text

        # Load data from files or use provided data
        if data_root is not None:
            # Initialize empty arrays
            self.left_samples = []
            self.right_samples = []
            self.hc_vs_pd = []
            self.pd_vs_dd = []
            self.patient_texts = []
            self.patient_ids = []
            self.task_names = []
            self.window_size = window_size
            self.data_root = data_root

            # Set up paths
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            self.questionnaires_template = pathlib.Path(data_root) / "questionnaire" / "questionnaire_response_{p:03d}.json"

            # Tasks
            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            self.wrists = ["LeftWrist", "RightWrist"]

            self.patient_ids_list = list(range(1, 470))
            print(f"Dataset: {len(self.patient_ids_list)} patients (001-469)")

            # Load data from files
            self._load_data()
        else:
            self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            self.hc_vs_pd = np.array(hc_vs_pd) if not isinstance(hc_vs_pd, np.ndarray) else hc_vs_pd
            self.pd_vs_dd = np.array(pd_vs_dd) if not isinstance(pd_vs_dd, np.ndarray) else pd_vs_dd
            self.patient_texts = list(patient_texts) if not isinstance(patient_texts, list) else patient_texts
            self.patient_ids = np.array(patient_ids) if not isinstance(patient_ids, np.ndarray) else patient_ids
            self.task_names = np.array(task_names) if not isinstance(task_names, np.ndarray) else task_names

        # Task name to ID mapping
        self.task_to_id = {
            "CrossArms": 0, "DrinkGlas": 1, "Entrainment": 2, "HoldWeight": 3,
            "LiftHold": 4, "PointFinger": 5, "Relaxed": 6, "StretchHold": 7,
            "TouchIndex": 8, "TouchNose": 9
        }

        # Group data by patient
        self._group_by_patient()

    def _load_data(self):
        """Load data from the file system."""
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patients"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            questionnaire_path = pathlib.Path(str(self.questionnaires_template).format(p=patient_id))

            if not patient_path.exists():
                continue

            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)

                condition = metadata.get('condition', '')
                questionnaire = {}
                try:
                    with open(questionnaire_path, 'r') as f:
                        questionnaire = json.load(f)
                except:
                    pass

                if self.apply_prepare_text:
                    per_patient_text = prepare_text(metadata, questionnaire)
                else:
                    per_patient_text = ""

                if condition == 'Healthy':
                    hc_vs_pd_label = 0  # Healthy
                    pd_vs_dd_label = -1  # Not applicable for PD vs DD
                    overlap = 0.70
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1
                    pd_vs_dd_label = 0   # Parkinson's for PD vs DD
                    overlap = 0
                else:
                    hc_vs_pd_label = -1  # Not applicable for HC vs PD
                    pd_vs_dd_label = 1   # Other disorders
                    overlap = 0.65

                patient_left_samples = []
                patient_right_samples = []
                patient_sample_texts = []
                patient_task_names = []

                for task in self.tasks:
                    left_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="LeftWrist"))
                    right_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="RightWrist"))

                    if not (left_path.exists() and right_path.exists()):
                        continue

                    try:
                        left_data = np.loadtxt(left_path, delimiter=",")
                        right_data = np.loadtxt(right_path, delimiter=",")

                        if left_data.shape[1] > 6:
                            left_data = left_data[:, :6]  # Take first 6 channels
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :]  # Skip first 0.5 sec

                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]

                        # Downsample
                        if self.apply_dowsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)

                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)

                        if left_data is None or right_data is None:
                            continue

                        # Create windows
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)

                        if left_windows is not None and right_windows is not None:
                            min_windows = min(len(left_windows), len(right_windows))

                            for i in range(min_windows):
                                patient_left_samples.append(left_windows[i])
                                patient_right_samples.append(right_windows[i])
                                patient_sample_texts.append(per_patient_text)
                                patient_task_names.append(task)

                    except Exception as e:
                        print(f"Error loading data for patient {patient_id}, task {task}: {e}")
                        continue

                # Add all samples from this patient
                if len(patient_left_samples) > 0:
                    n_samples = len(patient_left_samples)

                    for i in range(n_samples):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.patient_texts.append(patient_sample_texts[i])
                        self.hc_vs_pd.append(hc_vs_pd_label)
                        self.pd_vs_dd.append(pd_vs_dd_label)
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])

            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue

        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd = np.array(self.hc_vs_pd)
        self.pd_vs_dd = np.array(self.pd_vs_dd)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)

    def _group_by_patient(self):
        """Group windows by patient and task."""
        self.patient_data = {}
        unique_patients = np.unique(self.patient_ids)

        for patient_id in unique_patients:
            # Get all windows for this patient
            patient_mask = self.patient_ids == patient_id

            patient_left = self.left_samples[patient_mask]
            patient_right = self.right_samples[patient_mask]
            patient_tasks = self.task_names[patient_mask]
            patient_hc_pd = self.hc_vs_pd[patient_mask][0]  # Same for all windows
            patient_pd_dd = self.pd_vs_dd[patient_mask][0]
            patient_text = self.patient_texts[np.where(patient_mask)[0][0]]  # Get first text

            # Convert task names to IDs
            task_ids = np.array([self.task_to_id[task] for task in patient_tasks])

            self.patient_data[patient_id] = {
                'left_windows': patient_left,
                'right_windows': patient_right,
                'task_ids': task_ids,
                'hc_vs_pd': patient_hc_pd,
                'pd_vs_dd': patient_pd_dd,
                'patient_text': patient_text
            }

        self.patient_list = list(self.patient_data.keys())

    def get_train_test_split(self, split_type=1, **kwargs):
        """Get train/test split maintaining patient grouping."""
        if split_type == 1:
            # Patient-level split
            split_ratio = kwargs.get('split_ratio', 0.85)

            # Get unique patients and their labels
            patient_labels = []
            for pid in self.patient_list:
                hc_vs_pd_label = self.patient_data[pid]['hc_vs_pd']
                pd_vs_dd_label = self.patient_data[pid]['pd_vs_dd']

                if hc_vs_pd_label == 0:
                    label = 0  # Healthy
                elif hc_vs_pd_label == 1 and pd_vs_dd_label == 0:
                    label = 1  # Parkinson's
                else:
                    label = 2  # Other disorders
                patient_labels.append(label)

            patient_labels = np.array(patient_labels)

            train_patients, test_patients = train_test_split(
                self.patient_list,
                test_size=(1 - split_ratio),
                stratify=patient_labels,
                random_state=42
            )

            train_patients = set(train_patients)
            test_patients = set(test_patients)

            # Create train/test datasets
            train_data = self._create_split_dataset(train_patients)
            test_data = self._create_split_dataset(test_patients)

            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=train_data['left'],
                right_samples=train_data['right'],
                hc_vs_pd=train_data['hc_vs_pd'],
                pd_vs_dd=train_data['pd_vs_dd'],
                patient_texts=train_data['texts'],
                patient_ids=train_data['patient_ids'],
                task_names=train_data['task_names']
            )

            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=test_data['left'],
                right_samples=test_data['right'],
                hc_vs_pd=test_data['hc_vs_pd'],
                pd_vs_dd=test_data['pd_vs_dd'],
                patient_texts=test_data['texts'],
                patient_ids=test_data['patient_ids'],
                task_names=test_data['task_names']
            )

            print(f"\nHierarchical patient-level split:")
            print(f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples")
            print(f"  Test: {len(test_patients)} patients, {len(test_dataset)} samples")

            return train_dataset, test_dataset

        elif split_type == 3:
            # K-fold split (patient-level)
            k = kwargs.get('k', 5)

            patient_labels = []
            for pid in self.patient_list:
                hc_vs_pd_label = self.patient_data[pid]['hc_vs_pd']
                pd_vs_dd_label = self.patient_data[pid]['pd_vs_dd']

                if hc_vs_pd_label == 0:
                    label = 0
                elif hc_vs_pd_label == 1 and pd_vs_dd_label == 0:
                    label = 1
                else:
                    label = 2
                patient_labels.append(label)

            patient_labels = np.array(patient_labels)

            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            fold_datasets = []

            for fold_id, (train_idx, test_idx) in enumerate(skf.split(self.patient_list, patient_labels)):
                train_patients = set([self.patient_list[i] for i in train_idx])
                test_patients = set([self.patient_list[i] for i in test_idx])

                train_data = self._create_split_dataset(train_patients)
                test_data = self._create_split_dataset(test_patients)

                train_dataset = ParkinsonsDataLoader(
                    data_root=None,
                    left_samples=train_data['left'],
                    right_samples=train_data['right'],
                    hc_vs_pd=train_data['hc_vs_pd'],
                    pd_vs_dd=train_data['pd_vs_dd'],
                    patient_texts=train_data['texts'],
                    patient_ids=train_data['patient_ids'],
                    task_names=train_data['task_names']
                )

                test_dataset = ParkinsonsDataLoader(
                    data_root=None,
                    left_samples=test_data['left'],
                    right_samples=test_data['right'],
                    hc_vs_pd=test_data['hc_vs_pd'],
                    pd_vs_dd=test_data['pd_vs_dd'],
                    patient_texts=test_data['texts'],
                    patient_ids=test_data['patient_ids'],
                    task_names=test_data['task_names']
                )

                print(f"\nHierarchical Fold {fold_id+1}/{k}:")
                print(f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples")
                print(f"  Test: {len(test_patients)} patients, {len(test_dataset)} samples")

                fold_datasets.append((train_dataset, test_dataset))

            return fold_datasets

        else:
            raise ValueError(f"Invalid split_type: {split_type}. Use 1 (patient-level) or 3 (k-fold)")

    def _create_split_dataset(self, patient_set):
        """Create dataset from a set of patients."""
        left_list = []
        right_list = []
        hc_vs_pd_list = []
        pd_vs_dd_list = []
        texts_list = []
        patient_ids_list = []
        task_names_list = []

        for pid in patient_set:
            data = self.patient_data[pid]
            num_windows = len(data['left_windows'])

            # Add all windows for this patient
            left_list.extend(data['left_windows'])
            right_list.extend(data['right_windows'])
            hc_vs_pd_list.extend([data['hc_vs_pd']] * num_windows)
            pd_vs_dd_list.extend([data['pd_vs_dd']] * num_windows)
            texts_list.extend([data['patient_text']] * num_windows)
            patient_ids_list.extend([pid] * num_windows)

            # Convert task IDs back to names
            id_to_task = {v: k for k, v in self.task_to_id.items()}
            task_names_list.extend([id_to_task[tid] for tid in data['task_ids']])

        return {
            'left': np.array(left_list),
            'right': np.array(right_list),
            'hc_vs_pd': np.array(hc_vs_pd_list),
            'pd_vs_dd': np.array(pd_vs_dd_list),
            'texts': texts_list,
            'patient_ids': np.array(patient_ids_list),
            'task_names': np.array(task_names_list)
        }

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        """
        Returns all windows for a single patient, grouped by task.

        Returns:
            dict with:
                'left_windows': (num_windows, seq_len, input_dim)
                'right_windows': (num_windows, seq_len, input_dim)
                'task_ids': (num_windows,)
                'hc_vs_pd': scalar
                'pd_vs_dd': scalar
                'patient_text': string
        """
        patient_id = self.patient_list[idx]
        data = self.patient_data[patient_id]

        return {
            'left_windows': torch.FloatTensor(data['left_windows']),
            'right_windows': torch.FloatTensor(data['right_windows']),
            'task_ids': torch.LongTensor(data['task_ids']),
            'hc_vs_pd': torch.LongTensor([data['hc_vs_pd']]),
            'pd_vs_dd': torch.LongTensor([data['pd_vs_dd']]),
            'patient_text': data['patient_text']
        }
# ============================================================================
# Trainer
# ============================================================================

def extract_features(model, dataloader, device, use_text):
    model.eval()
    all_features = []
    all_hc_pd_labels = []
    all_pd_dd_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            left_sample, right_sample, hc_pd, pd_dd, patient_text = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            
            text_input = patient_text if use_text else None
            
            features = model.get_features(left_sample, right_sample, text_input, device)
            
            all_features.append(features.cpu().numpy())
            all_hc_pd_labels.append(hc_pd.numpy())
            all_pd_dd_labels.append(pd_dd.numpy())
    
    all_features = np.vstack(all_features)
    all_hc_pd_labels = np.concatenate(all_hc_pd_labels)
    all_pd_dd_labels = np.concatenate(all_pd_dd_labels)
    
    return all_features, all_hc_pd_labels, all_pd_dd_labels

def collate_fn(batch):
    """
    Custom collate function for hierarchical data loader.
    Since each patient has different number of windows, we return a list.
    """
    batch_data = []
    hc_vs_pd_labels = []
    pd_vs_dd_labels = []
    patient_texts = []

    for sample in batch:
        batch_data.append({
            'left_windows': sample['left_windows'],
            'right_windows': sample['right_windows'],
            'task_ids': sample['task_ids']
        })
        hc_vs_pd_labels.append(sample['hc_vs_pd'])
        pd_vs_dd_labels.append(sample['pd_vs_dd'])
        patient_texts.append(sample['patient_text'])

    # Stack labels
    hc_vs_pd_labels = torch.stack(hc_vs_pd_labels).squeeze()
    pd_vs_dd_labels = torch.stack(pd_vs_dd_labels).squeeze()

    return batch_data, hc_vs_pd_labels, pd_vs_dd_labels, patient_texts


def train__single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device, use_text):
    """Train hierarchical model for one epoch"""
    model.train()
    train_loss = 0.0
    hc_pd_train_pred, hc_pd_train_labels = [], []
    pd_dd_train_pred, pd_dd_train_labels = [], []

    for batch_data, hc_pd, pd_dd, patient_texts in tqdm(dataloader, desc="Training"):
        hc_pd = hc_pd.to(device)
        pd_dd = pd_dd.to(device)

        optimizer.zero_grad()
        text_input = patient_texts if use_text else None
        hc_pd_logits, pd_dd_logits = model(batch_data, text_input, device)

        total_loss = 0
        loss_count = 0

        # HC vs PD loss
        valid_hc_pd_mask = (hc_pd != -1)
        if valid_hc_pd_mask.any():
            valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
            valid_labels_hc = hc_pd[valid_hc_pd_mask]
            loss_hc = criterion_hc(valid_logits_hc, valid_labels_hc)
            total_loss += loss_hc
            loss_count += 1

            preds_hc = torch.argmax(valid_logits_hc, dim=1)
            hc_pd_train_pred.extend(preds_hc.cpu().numpy())
            hc_pd_train_labels.extend(valid_labels_hc.cpu().numpy())

        # PD vs DD loss
        valid_pd_dd_mask = (pd_dd != -1)
        if valid_pd_dd_mask.any():
            valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
            valid_labels_pd = pd_dd[valid_pd_dd_mask]
            loss_pd = criterion_pd(valid_logits_pd, valid_labels_pd)
            total_loss += loss_pd
            loss_count += 1

            preds_pd = torch.argmax(valid_logits_pd, dim=1)
            pd_dd_train_pred.extend(preds_pd.cpu().numpy())
            pd_dd_train_labels.extend(valid_labels_pd.cpu().numpy())

        # Backward pass
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += avg_loss.item()

    train_loss /= len(dataloader)

    # Calculate training metrics
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred,
                                        "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred,
                                        "Training PD vs DD", verbose=False)

    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device, use_text):
    """Validate hierarchical model for one epoch"""
    model.eval()
    val_loss = 0.0
    hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs = [], [], []
    pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = [], [], []

    with torch.no_grad():
        for batch_data, hc_pd, pd_dd, patient_texts in tqdm(dataloader, desc="Validation"):
            hc_pd = hc_pd.to(device)
            pd_dd = pd_dd.to(device)

            text_input = patient_texts if use_text else None
            hc_pd_logits, pd_dd_logits = model(batch_data, text_input, device)

            total_loss = 0
            loss_count = 0

            # HC vs PD loss
            valid_hc_pd_mask = (hc_pd != -1)
            if valid_hc_pd_mask.any():
                valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                valid_labels_hc = hc_pd[valid_hc_pd_mask]
                loss_hc = criterion_hc(valid_logits_hc, valid_labels_hc)
                total_loss += loss_hc
                loss_count += 1

                preds_hc = torch.argmax(valid_logits_hc, dim=1)
                probs_hc = F.softmax(valid_logits_hc, dim=1)[:, 1]
                hc_pd_val_pred.extend(preds_hc.cpu().numpy())
                hc_pd_val_labels.extend(valid_labels_hc.cpu().numpy())
                hc_pd_val_probs.extend(probs_hc.cpu().numpy())

            # PD vs DD loss
            valid_pd_dd_mask = (pd_dd != -1)
            if valid_pd_dd_mask.any():
                valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                valid_labels_pd = pd_dd[valid_pd_dd_mask]
                loss_pd = criterion_pd(valid_logits_pd, valid_labels_pd)
                total_loss += loss_pd
                loss_count += 1

                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                probs_pd = F.softmax(valid_logits_pd, dim=1)[:, 1]
                pd_dd_val_pred.extend(preds_pd.cpu().numpy())
                pd_dd_val_labels.extend(valid_labels_pd.cpu().numpy())
                pd_dd_val_probs.extend(probs_pd.cpu().numpy())

            if loss_count > 0:
                avg_loss = total_loss / loss_count
                val_loss += avg_loss.item()

    val_loss /= len(dataloader)

    return (val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs,
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs)


def extract_features(model, dataloader, device, use_text):
    """Extract features from hierarchical model"""
    model.eval()
    all_features = []
    all_hc_pd_labels = []
    all_pd_dd_labels = []

    with torch.no_grad():
        for batch_data, hc_pd, pd_dd, patient_texts in tqdm(dataloader, desc="Extracting features"):
            text_input = patient_texts if use_text else None

            features = model.get_features(batch_data, text_input, device)

            all_features.append(features.cpu().numpy())
            all_hc_pd_labels.append(hc_pd.numpy())
            all_pd_dd_labels.append(pd_dd.numpy())

    all_features = np.vstack(all_features)
    all_hc_pd_labels = np.concatenate(all_hc_pd_labels)
    all_pd_dd_labels = np.concatenate(all_pd_dd_labels)

    return all_features, all_hc_pd_labels, all_pd_dd_labels


def train_model(config):
    """Train the hierarchical attention model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("TRAINING HIERARCHICAL DUAL-CHANNEL TRANSFORMER")
    print("="*70)

    os.makedirs("metrics", exist_ok=True)

    # Load dataset using ParkinsonsDataLoader
    full_dataset = ParkinsonsDataLoader(
        config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter'],
        apply_prepare_text=config.get('apply_prepare_text', False)
    )

    split_type = config.get('split_type', 3)

    if split_type == 3:
        fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config['num_folds'])
        num_folds = len(fold_datasets)
    else:
        train_dataset, val_dataset = full_dataset.get_train_test_split(
            split_type=split_type,
            split_ratio=config.get('split_ratio', 0.85)
        )
        fold_datasets = [(train_dataset, val_dataset)]
        num_folds = 1

    all_fold_results = []

    for fold_idx in range(num_folds):

        if num_folds > 1:
            print(f"\n{'='*70}")
            print(f"Starting Fold {fold_idx+1}/{num_folds}")
            print(f"{'='*70}")

        train_dataset, val_dataset = fold_datasets[fold_idx]

        # Use custom collate function for variable-length sequences
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn
        )

        # Model - Use Model
        model = Model(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            seq_len=config['seq_len'],
            num_classes=config['num_classes'],
            num_tasks=config.get('num_tasks', 10),
            use_text=config.get('use_text', False)
        ).to(device)

        print(f"\nModel: Model")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()

        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        fold_features = None
        fold_hc_pd_labels = None
        fold_pd_dd_labels = None

        fold_metrics_hc = []
        fold_metrics_pd = []

        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None

        for epoch in range(config['num_epochs']):

            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

            #############Training phase###########
            train_loss, train_metrics_hc, train_metrics_pd = train__single_epoch(
                model, train_loader, hc_pd_loss, pd_dd_loss, optimizer,
                device, config.get('use_text', False)
            )

            ###########Validation phase############
            val_results = validate_single_epoch(
                model, val_loader, hc_pd_loss, pd_dd_loss,
                device, config.get('use_text', False)
            )
            val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results

            print("\n" + "="*60)
            val_metrics_hc = calculate_metrics(
                hc_pd_val_labels, hc_pd_val_pred,
                f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation HC vs PD",
                verbose=True
            )
            val_metrics_pd = calculate_metrics(
                pd_dd_val_labels, pd_dd_val_pred,
                f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation PD vs DD",
                verbose=True
            )
            print("="*60)

            if hc_pd_val_labels:
                fold_metrics_hc.append({
                    'epoch': epoch + 1,
                    'predictions': hc_pd_val_pred.copy(),
                    'labels': hc_pd_val_labels.copy(),
                    'metrics': val_metrics_hc
                })

            if pd_dd_val_labels:
                fold_metrics_pd.append({
                    'epoch': epoch + 1,
                    'predictions': pd_dd_val_pred.copy(),
                    'labels': pd_dd_val_labels.copy(),
                    'metrics': val_metrics_pd
                })

            val_acc_hc = val_metrics_hc.get('accuracy', 0)
            val_acc_pd = val_metrics_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2

            train_acc_hc = train_metrics_hc.get('accuracy', 0)
            train_acc_pd = train_metrics_pd.get('accuracy', 0)

            scheduler.step(val_loss)

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc_hc'].append(train_acc_hc)
            history['train_acc_pd'].append(train_acc_pd)
            history['val_loss'].append(val_loss)
            history['val_acc_hc'].append(val_acc_hc)
            history['val_acc_pd'].append(val_acc_pd)
            history['val_acc_combined'].append(val_acc_combined)

            print(f"\n{'Fold ' + str(fold_idx+1) + ', ' if num_folds > 1 else ''}Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")

            # Save best model and store probabilities for ROC
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1

                # Store best predictions and probabilities for ROC curves
                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)

                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)

                model_save_name = f'hierarchical_best_model{"_fold_" + str(fold_idx+1) if num_folds > 1 else ""}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx if num_folds > 1 else None,
                    'epoch': epoch,
                    'val_acc_combined': val_acc_combined,
                    'val_acc_hc': val_acc_hc,
                    'val_acc_pd': val_acc_pd,
                    'config': config
                }, model_save_name)
                print(f"✓ New best model saved: {model_save_name}")

        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""

            if fold_metrics_hc and fold_metrics_pd:
                # Use CSV format instead of text
                save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                                    fold_metrics_hc, fold_metrics_pd)

        fold_features, fold_hc_pd_labels, fold_pd_dd_labels = extract_features(
            model, val_loader, device, config.get('use_text', False)
        )

        fold_result = {
            'best_val_accuracy': best_val_acc,
            'history': history,
            'features': fold_features,
            'hc_pd_labels': fold_hc_pd_labels,
            'pd_dd_labels': fold_pd_dd_labels
        }
        all_fold_results.append(fold_result)

        if config.get('create_plots', True):
            plot_dir = f"plots/hierarchical_{'fold_' + str(fold_idx+1) if num_folds > 1 else 'single_run'}"
            os.makedirs(plot_dir, exist_ok=True)

            plot_loss(history, f"{plot_dir}/loss.png")

            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs,
                              f"{plot_dir}/roc_hc_vs_pd.png")

            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs,
                              f"{plot_dir}/roc_pd_vs_dd.png")

            if fold_features is not None:
                plot_tsne(fold_features, fold_hc_pd_labels, fold_pd_dd_labels, output_dir=plot_dir)

    return all_fold_results




def main():
    """Main function for training hierarchical attention model"""

    config = {
        # Data settings
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'apply_prepare_text': False,

        # Split settings
        'split_type': 3,  # 1=patient-level, 3=k-fold
        'split_ratio': 0.85,
        'num_folds': 5,

        # Model architecture
        'input_dim': 6,
        'model_dim': 64,
        'num_heads': 8,
        'num_layers': 3,  # Number of window-level cross-attention layers
        'd_ff': 256,
        'dropout': 0.2,
        'seq_len': 256,
        'num_classes': 2,
        'num_tasks': 10,  # Number of tasks for hierarchical attention
        'use_text': False,

        # Training settings
        'batch_size': 8,  # Smaller batch size for hierarchical model (patient-level batching)
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,

        # Output settings
        'save_metrics': True,  # Metrics will be saved as CSV
        'create_plots': True,
    }
    results = train_model(config)

    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    for i, fold_result in enumerate(results):
        print(f"Fold {i+1}: Best Validation Accuracy = {fold_result['best_val_accuracy']:.4f}")

    avg_accuracy = np.mean([r['best_val_accuracy'] for r in results])
    std_accuracy = np.std([r['best_val_accuracy'] for r in results])
    print(f"\nAverage Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
