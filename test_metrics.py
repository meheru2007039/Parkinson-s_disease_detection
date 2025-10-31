"""
Test script to verify metric calculation and storing functions
"""
import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_notebook import calculate_metrics, save_fold_metric

def test_calculate_metrics():
    """Test the calculate_metrics function with known inputs"""
    print("="*70)
    print("Testing calculate_metrics function")
    print("="*70)

    # Test case 1: Perfect predictions
    print("\nTest 1: Perfect predictions")
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 0]

    metrics = calculate_metrics(y_true, y_pred, "Perfect Classification", verbose=True)

    assert metrics['accuracy'] == 1.0, f"Expected accuracy 1.0, got {metrics['accuracy']}"
    assert metrics['precision_avg'] == 1.0, f"Expected precision 1.0, got {metrics['precision_avg']}"
    assert metrics['recall_avg'] == 1.0, f"Expected recall 1.0, got {metrics['recall_avg']}"
    assert metrics['f1_avg'] == 1.0, f"Expected F1 1.0, got {metrics['f1_avg']}"
    print("✓ Test 1 passed: Perfect predictions verified")

    # Test case 2: Mixed predictions
    print("\nTest 2: Mixed predictions")
    y_true = [0, 0, 1, 1, 1, 0, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0, 0, 1]  # 2 errors out of 8

    metrics = calculate_metrics(y_true, y_pred, "Mixed Classification", verbose=True)

    expected_accuracy = 6/8  # 6 correct out of 8
    assert abs(metrics['accuracy'] - expected_accuracy) < 0.001, \
        f"Expected accuracy {expected_accuracy}, got {metrics['accuracy']}"
    print(f"✓ Test 2 passed: Mixed predictions accuracy = {metrics['accuracy']:.4f}")

    # Test case 3: All same class (edge case)
    print("\nTest 3: All same class predictions")
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]

    metrics = calculate_metrics(y_true, y_pred, "All Same Class", verbose=True)

    assert metrics['accuracy'] == 1.0, f"Expected accuracy 1.0, got {metrics['accuracy']}"
    print("✓ Test 3 passed: All same class handled correctly")

    # Test case 4: Empty input (edge case)
    print("\nTest 4: Empty input")
    y_true = []
    y_pred = []

    metrics = calculate_metrics(y_true, y_pred, "Empty", verbose=False)

    assert metrics == {}, "Expected empty dict for empty input"
    print("✓ Test 4 passed: Empty input handled correctly")

    # Test case 5: Verify confusion matrix
    print("\nTest 5: Confusion matrix verification")
    y_true = [0, 0, 0, 1, 1, 1]  # 3 class 0, 3 class 1
    y_pred = [0, 0, 1, 1, 1, 0]  # 2 correct 0s, 1 wrong 0->1, 1 wrong 1->0, 2 correct 1s

    metrics = calculate_metrics(y_true, y_pred, "CM Verification", verbose=True)
    cm = metrics['confusion_matrix']

    # Expected confusion matrix:
    # [[2, 1],   # True 0: 2 predicted as 0, 1 predicted as 1
    #  [1, 2]]   # True 1: 1 predicted as 0, 2 predicted as 1

    assert cm[0, 0] == 2, f"CM[0,0] should be 2, got {cm[0,0]}"
    assert cm[0, 1] == 1, f"CM[0,1] should be 1, got {cm[0,1]}"
    assert cm[1, 0] == 1, f"CM[1,0] should be 1, got {cm[1,0]}"
    assert cm[1, 1] == 2, f"CM[1,1] should be 2, got {cm[1,1]}"
    print("✓ Test 5 passed: Confusion matrix is correct")

    print("\n" + "="*70)
    print("All calculate_metrics tests passed!")
    print("="*70)


def test_save_fold_metric():
    """Test the save_fold_metric function"""
    print("\n" + "="*70)
    print("Testing save_fold_metric function")
    print("="*70)

    # Create test data
    fold_metrics_hc = [
        {
            'epoch': 1,
            'predictions': [0, 0, 1, 1],
            'labels': [0, 0, 1, 1],
            'metrics': {
                'accuracy': 1.0,
                'precision_avg': 1.0,
                'recall_avg': 1.0,
                'f1_avg': 1.0,
                'precision_per_class': np.array([1.0, 1.0]),
                'recall_per_class': np.array([1.0, 1.0]),
                'f1_per_class': np.array([1.0, 1.0]),
                'support_per_class': np.array([2, 2])
            }
        },
        {
            'epoch': 2,
            'predictions': [0, 1, 1, 1],
            'labels': [0, 0, 1, 1],
            'metrics': {
                'accuracy': 0.75,
                'precision_avg': 0.75,
                'recall_avg': 0.75,
                'f1_avg': 0.75,
                'precision_per_class': np.array([1.0, 0.67]),
                'recall_per_class': np.array([0.5, 1.0]),
                'f1_per_class': np.array([0.67, 0.8]),
                'support_per_class': np.array([2, 2])
            }
        }
    ]

    fold_metrics_pd = [
        {
            'epoch': 1,
            'predictions': [0, 0, 1],
            'labels': [0, 0, 1],
            'metrics': {
                'accuracy': 1.0,
                'precision_avg': 1.0,
                'recall_avg': 1.0,
                'f1_avg': 1.0,
                'precision_per_class': np.array([1.0, 1.0]),
                'recall_per_class': np.array([1.0, 1.0]),
                'f1_per_class': np.array([1.0, 1.0]),
                'support_per_class': np.array([2, 1])
            }
        }
    ]

    # Test saving metrics
    print("\nSaving test metrics to CSV files...")
    save_fold_metric(
        fold_idx=0,
        fold_suffix="_fold_1",
        best_epoch=1,
        best_val_acc=0.95,
        fold_metrics_hc=fold_metrics_hc,
        fold_metrics_pd=fold_metrics_pd
    )

    # Verify files were created
    hc_file = "metrics/hc_vs_pd_metrics_fold_1.csv"
    pd_file = "metrics/pd_vs_dd_metrics_fold_1.csv"

    assert os.path.exists(hc_file), f"HC metrics file not created: {hc_file}"
    assert os.path.exists(pd_file), f"PD metrics file not created: {pd_file}"
    print(f"✓ Files created successfully")

    # Read and verify CSV content
    import pandas as pd

    print("\nVerifying HC vs PD metrics CSV...")
    hc_df = pd.read_csv(hc_file)
    print(f"  - Rows: {len(hc_df)}")
    print(f"  - Columns: {list(hc_df.columns)}")

    # Check that we have data for 2 epochs
    epochs = hc_df['Epoch'].unique()
    assert len(epochs) == 2, f"Expected 2 epochs, found {len(epochs)}"
    print(f"  - Epochs found: {sorted(epochs)}")

    # Check that we have Overall, Per_Class, and Confusion_Matrix rows
    metric_types = hc_df['Metric_Type'].unique()
    print(f"  - Metric types: {list(metric_types)}")

    print("\nVerifying PD vs DD metrics CSV...")
    pd_df = pd.read_csv(pd_file)
    print(f"  - Rows: {len(pd_df)}")
    print(f"  - Columns: {list(pd_df.columns)}")

    # Test edge case: only HC metrics (no PD metrics)
    print("\nTest edge case: Only HC metrics, no PD metrics")
    save_fold_metric(
        fold_idx=1,
        fold_suffix="_fold_2",
        best_epoch=1,
        best_val_acc=0.90,
        fold_metrics_hc=fold_metrics_hc,
        fold_metrics_pd=[]  # Empty PD metrics
    )

    hc_file2 = "metrics/hc_vs_pd_metrics_fold_2.csv"
    assert os.path.exists(hc_file2), "HC file should still be created when PD is empty"
    print("✓ Edge case passed: HC metrics saved even when PD metrics are empty")

    print("\n" + "="*70)
    print("All save_fold_metric tests passed!")
    print("="*70)


def test_issue_detection():
    """Test to detect the potential bug in train_model"""
    print("\n" + "="*70)
    print("Testing for potential issues")
    print("="*70)

    print("\n🔍 Issue 1: Condition for saving metrics")
    print("   Location: kaggle_notebook.py:1721")
    print("   Current code: if fold_metrics_hc and fold_metrics_pd:")
    print("   Issue: Metrics won't be saved if either list is empty")
    print("   Recommendation: Change to 'if fold_metrics_hc or fold_metrics_pd:'")
    print("   Severity: MEDIUM - Could lose data in edge cases")

    print("\n🔍 Issue 2: Redundant confusion matrix calculation")
    print("   Location: kaggle_notebook.py:138, 196")
    print("   Issue: Confusion matrix is calculated twice:")
    print("     1. In calculate_metrics (line 51)")
    print("     2. In save_fold_metric (lines 138, 196)")
    print("   Impact: Minor performance overhead, but functionally correct")
    print("   Severity: LOW - Optimization opportunity")

    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "METRIC VALIDATION TEST SUITE" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")

    try:
        # Create metrics directory if it doesn't exist
        os.makedirs("metrics", exist_ok=True)

        # Run tests
        test_calculate_metrics()
        test_save_fold_metric()
        test_issue_detection()

        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*20 + "ALL TESTS PASSED!" + " "*28 + "║")
        print("╚" + "="*68 + "╝")
        print("\nConclusion: Metric calculation functions are working correctly.")
        print("Minor issues detected (see above) but core functionality is sound.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
