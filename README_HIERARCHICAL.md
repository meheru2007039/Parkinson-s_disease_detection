# Hierarchical Attention for Parkinson's Disease Detection

This document describes the hierarchical attention architecture implemented for Parkinson's disease detection using dual-channel (left and right wrist) accelerometer data.

## Overview

The hierarchical attention model processes accelerometer data at two levels:

1. **Window-level (Level 1)**: Cross-attention between left and right wrist samples within each time window
2. **Task-level (Level 2)**: Attention across multiple windows, grouped by task, with learnable task embeddings

## Architecture

### Level 1: Window-Level Processing

Each window contains 256 time steps of 6-channel accelerometer data from both wrists:
- **Input**: Left wrist (256, 6) and Right wrist (256, 6)
- **Processing**:
  - Linear projection to model dimension
  - Positional encoding
  - Multiple layers of cross-attention between left and right wrists
  - Global average pooling
- **Output**: Window embedding (model_dim * 2)

### Level 2: Task-Level Processing

Windows are grouped by the 10 different tasks performed:
- **Tasks**: CrossArms, DrinkGlas, Entrainment, HoldWeight, LiftHold, PointFinger, Relaxed, StretchHold, TouchIndex, TouchNose
- **Processing**:
  - Window embeddings are grouped by task
  - Task number embeddings (0-9) are added to make the model task-aware
  - Task-level attention is applied across windows within each task
  - Task representations are pooled
- **Output**: Patient-level embedding

### Classification Heads

Two binary classification tasks:
1. **HC vs PD**: Healthy Control vs Parkinson's Disease
2. **PD vs DD**: Parkinson's Disease vs Differential Diagnosis

## Key Components

### 1. HierarchicalDualChannelTransformer (Model)

Located in `kaggle_notebook.py` (lines 647-940)

Key features:
- Window-level cross-attention layers (existing CrossAttention module)
- Task embedding layer (nn.Embedding with 10 tasks)
- Task-level attention module (MultiheadAttention)
- Dual classification heads

### 2. HierarchicalParkinsonsDataLoader (Data Loader)

Located in `kaggle_notebook.py` (lines 1453-1719)

Key features:
- Groups windows by patient and task
- Maps task names to IDs (0-9)
- Returns patient-level samples with all windows and task information
- Supports patient-level and k-fold splitting

### 3. Training Functions

New hierarchical training functions:
- `hierarchical_collate_fn()`: Custom collate function for variable-length sequences
- `train_hierarchical_single_epoch()`: Training loop for hierarchical model
- `validate_hierarchical_single_epoch()`: Validation loop
- `extract_hierarchical_features()`: Feature extraction for t-SNE
- `train_hierarchical_model()`: Main training function

### 4. Metrics Saving (CSV Format)

New CSV saving functions:
- `save_metrics_to_csv()`: Saves detailed metrics to CSV format
- `save_fold_metric_csv()`: Saves fold-wise metrics to CSV files

**CSV Output Structure**:
- `metrics/hc_vs_pd_metrics_fold_X.csv`: HC vs PD metrics for each fold
- `metrics/pd_vs_dd_metrics_fold_X.csv`: PD vs DD metrics for each fold

Each CSV contains:
- Overall accuracy, precision, recall, F1-score per epoch
- Per-class metrics
- Confusion matrices

## Usage

### Running the Hierarchical Model

```python
# The hierarchical model is now the default
# Simply run the notebook:
if __name__ == "__main__":
    results = main_hierarchical()
```

### Configuration

Edit the config dictionary in `main_hierarchical()`:

```python
config = {
    # Data settings
    'data_root': "/path/to/dataset",
    'apply_downsampling': True,
    'apply_bandpass_filter': True,
    'apply_prepare_text': False,

    # Split settings
    'split_type': 3,  # 3=k-fold, 1=patient-level
    'num_folds': 5,

    # Model architecture
    'input_dim': 6,
    'model_dim': 64,
    'num_heads': 8,
    'num_layers': 3,  # Window-level cross-attention layers
    'd_ff': 256,
    'dropout': 0.2,
    'seq_len': 256,
    'num_tasks': 10,  # Number of tasks
    'use_text': False,

    # Training settings
    'batch_size': 8,  # Patient-level batching
    'learning_rate': 0.0005,
    'weight_decay': 0.01,
    'num_epochs': 100,

    # Output settings
    'save_metrics': True,  # Saves as CSV
    'create_plots': True,
}
```

### Running the Original Baseline Model

To run the original non-hierarchical model:

```python
if __name__ == "__main__":
    results = main()  # Uses original DualChannelTransformer
```

## Data Flow

```
Raw Data (100Hz, 6 channels per wrist)
    ↓
Preprocessing (downsample, filter, window creation)
    ↓
Patient-level grouping (all windows for a patient)
    ↓
LEVEL 1: Window-level Cross-Attention
    - Process each window pair (left + right wrist)
    - Output: Window embeddings
    ↓
LEVEL 2: Task-level Attention
    - Group windows by task
    - Add task embeddings (task 0-9)
    - Apply task-level attention
    - Pool across tasks
    ↓
Patient-level embedding
    ↓
Classification Heads (HC vs PD, PD vs DD)
    ↓
Predictions + Metrics (saved as CSV)
```

## Model Improvements

The hierarchical attention architecture provides several advantages:

1. **Task Awareness**: The model explicitly learns task-specific patterns through task embeddings
2. **Better Context**: Window-level and task-level processing captures both local and global patterns
3. **Patient-level Processing**: Processes all data from a patient together, maintaining temporal and task relationships
4. **Interpretability**: Attention weights at both levels can be visualized to understand what the model learns

## Output Files

### Model Checkpoints
- `hierarchical_best_model_fold_X.pth`: Best model for each fold

### Metrics (CSV format)
- `metrics/hc_vs_pd_metrics_fold_X.csv`: Detailed metrics for HC vs PD task
- `metrics/pd_vs_dd_metrics_fold_X.csv`: Detailed metrics for PD vs DD task

### Plots
- `plots/hierarchical_fold_X/loss.png`: Training and validation loss curves
- `plots/hierarchical_fold_X/roc_hc_vs_pd.png`: ROC curve for HC vs PD
- `plots/hierarchical_fold_X/roc_pd_vs_dd.png`: ROC curve for PD vs DD
- `plots/hierarchical_fold_X/tsne_visualization.png`: t-SNE visualization of learned features

## Comparison with Baseline

| Aspect | Baseline Model | Hierarchical Model |
|--------|---------------|-------------------|
| Processing Level | Window-level only | Window + Task levels |
| Task Awareness | No | Yes (task embeddings) |
| Batch Size | 64 (window-level) | 8 (patient-level) |
| Data Organization | Individual windows | Grouped by patient & task |
| Attention Levels | 1 level (cross-attention) | 2 levels (window + task) |
| Metrics Format | .txt files | .csv files |

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- scipy
- transformers (for BERT, if using text)
- matplotlib
- tqdm

## Notes

1. **Batch Size**: The hierarchical model uses patient-level batching (batch_size=8 means 8 patients per batch), which is different from the baseline window-level batching.

2. **Memory**: The hierarchical model may require more memory since it processes all windows for a patient simultaneously. Adjust batch_size if you encounter memory issues.

3. **Task-wise Split**: The hierarchical data loader supports patient-level and k-fold splits. Task-wise splitting can be added if needed.

4. **CSV Metrics**: Metrics are now saved in CSV format for easier analysis in tools like Excel, Python pandas, or R.

## Citation

If you use this code, please cite the original PADS dataset:

```
PADS: Parkinson's Disease Smartwatch Dataset
Version 1.0.0
```

## License

This implementation is provided for research and educational purposes.
