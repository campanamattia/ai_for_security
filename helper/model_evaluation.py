from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
import numpy as np

def evaluate_classification_model(y_true, y_pred, model_name="Model"):
    """
    Print simplified but comprehensive model evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str, optional
        Name of the model for display purposes
    """
    console = Console()
    
    # Calculate core metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted')
    }
    
    # Create metrics table
    table = Table(show_header=True, header_style="bold white", box=None)
    table.add_column("Metric", style="white")
    table.add_column("Score", justify="right")
    
    # Add metrics rows with color coding
    for metric, value in metrics.items():
        color = "green" if value > 0.8 else "yellow" if value > 0.6 else "red"
        table.add_row(
            metric,
            f"[{color}]{value:.5f}[/{color}]"
        )
    
    # Calculate confusion matrix summary
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    correct = cm.diagonal().sum()
    
    # Create confusion matrix table
    cm_table = Table(show_header=True, header_style="bold white", box=None)
    cm_table.add_column("Predicted \\ Actual", style="white")
    for i in range(cm.shape[1]):
        cm_table.add_column(f"Class {i}", justify="right")
    
    for i in range(cm.shape[0]):
        row = [f"Class {i}"] + [str(cm[i, j]) for j in range(cm.shape[1])]
        cm_table.add_row(*row)
    
    # Create a combined panel with two columns
    combined_table = Table.grid(expand=True)
    combined_table.add_column(justify="center", ratio=1)
    combined_table.add_column(justify="center", ratio=1)
    
    combined_table.add_row(table, cm_table)
    
    # Create and display panel
    panel = Panel(
        combined_table,
        title=f"[bold]{model_name} - Performance Metrics and Confusion Matrix[/bold]",
        border_style="white"
    )
    
    console.print("\n", panel, "\n")

# Example usage:
"""
# Binary classification
evaluate_classification_model(y_test, y_pred, "Random Forest")

# Multiclass classification
evaluate_classification_model(y_test, y_pred, "XGBoost Classifier")
"""