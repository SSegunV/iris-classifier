import pytest

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train import train_and_evaluate


def test_accuracy_reasonable():
    """Test that accuracy is within a realistic range."""
    accuracy, cMatrix, _ = train_and_evaluate(test_size=0.2, random_state=42)
    assert 0.8 <= accuracy <= 1.0, f"Accuracy too low: {accuracy}"

def test_confusion_matrix_shape():
    """Ensure confusion matrix is 3x3 for the 3 iris classes."""
    _, cMatrix, _ = train_and_evaluate(test_size=0.2, random_state=42)
    assert cMatrix.shape == (3, 3), f"Confusion matrix shape unexpected: {cMatrix.shape}"

def test_confusion_matrix_non_negative():
    """All values in confusion matrix should be non-negative integers."""
    _, cMatrix, _ = train_and_evaluate(test_size=0.2, random_state=42)
    assert (cMatrix >= 0).all(), "Confusion matrix contains negative values"
    assert issubclass(cMatrix.dtype.type, (int, float)), "Matrix has wrong type"

def test_repeatable_results():
    """Ensure the same random state gives the same accuracy."""
    acc1, _, _ = train_and_evaluate(test_size=0.2, random_state=42)
    acc2, _, _ = train_and_evaluate(test_size=0.2, random_state=42)
    assert acc1 == acc2, "Accuracy not repeatable with same random state"
