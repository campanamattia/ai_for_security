from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.jupyter import print as jupyter_print
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import numpy as np
import pandas as pd
from IPython import get_ipython

# Check if we're in a Jupyter notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython
            return False
        else:
            return False
    except NameError:  # Probably standard Python interpreter
        return False

# Use appropriate print function
def rich_print(obj):
    if is_notebook():
        jupyter_print(obj)
    else:
        console = Console()
        console.print(obj)

def print_overall_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Print overall metrics in a nicely formatted panel.
    Works for both binary and multiclass classification.
    """
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average='macro'),
        "Weighted Precision": precision_score(y_true, y_pred, average='weighted'),
        "Macro Recall": recall_score(y_true, y_pred, average='macro'),
        "Weighted Recall": recall_score(y_true, y_pred, average='weighted'),
        "Macro F1": f1_score(y_true, y_pred, average='macro'),
        "Weighted F1": f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        # Check if binary or multiclass
        if len(y_pred_proba.shape) == 2:  # If 2D array
            if y_pred_proba.shape[1] == 2:  # Binary
                metrics["ROC AUC"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multiclass
                metrics["ROC AUC (OvR)"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    # Create table for metrics
    table = Table(show_header=False, box=None)
    for metric, value in metrics.items():
        color = "green" if value > 0.9 else "yellow" if value > 0.7 else "red"
        table.add_row(
            f"{metric}:",
            f"[{color}]{value:.5f}[/{color}]"
        )
    
    # Wrap in a panel
    panel = Panel(
        table,
        title=f"[bold blue]{model_name} - Overall Metrics[/bold blue]",
        border_style="blue"
    )
    rich_print(panel)

def print_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Print confusion matrix with percentages and summary statistics.
    Works for both binary and multiclass classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percentages = cm / cm.sum() * 100
    
    # Get number of classes
    n_classes = cm.shape[0]
    class_labels = np.unique(y_true)
    
    # Create main confusion matrix table
    table = Table(
        title=f"[bold blue]{model_name} - Confusion Matrix[/bold blue]",
        show_header=True,
        header_style="bold magenta"
    )
    
    # Add columns
    table.add_column("Predicted ↓ Actual →", justify="center", style="cyan")
    for i in range(n_classes):
        table.add_column(f"Class {class_labels[i]}", justify="center")
    
    # Add rows with counts and percentages
    for i in range(n_classes):
        row_values = [f"Class {class_labels[i]}"]
        for j in range(n_classes):
            row_values.append(
                f"{cm[i,j]:,}\n[dim]({cm_percentages[i,j]:.2f}%)[/dim]"
            )
        table.add_row(*row_values)
    
    # Calculate and add summary statistics
    total = cm.sum()
    correct = cm.diagonal().sum()
    incorrect = total - correct
    
    summary = Table(show_header=False, box=None)
    summary.add_row("Total Samples:", f"{total:,}")
    summary.add_row(
        "Correct Predictions:", 
        f"[green]{correct:,}[/green] ({correct/total*100:.2f}%)"
    )
    summary.add_row(
        "Incorrect Predictions:", 
        f"[red]{incorrect:,}[/red] ({incorrect/total*100:.2f}%)"
    )
    
    # Print everything
    rich_print("\n")
    rich_print(table)
    rich_print("\n")
    rich_print(summary)
    
    # Print detailed classification report using pandas for better notebook display
    report = classification_report(y_true, y_pred, digits=5, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    rich_print("\nDetailed Classification Report:")
    rich_print(report_df)

def print_feature_importance(feature_importance_df, top_n=10, model_name="Model"):
    """
    Print feature importance in a nicely formatted table.
    """
    table = Table(
        title=f"[bold blue]{model_name} - Top {top_n} Important Features[/bold blue]",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Feature", justify="left")
    table.add_column("Importance", justify="right")
    table.add_column("% of Total", justify="right")
    
    # Calculate total importance
    total_importance = feature_importance_df['importance'].sum()
    
    # Add rows for top features
    for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
        percentage = (row['importance'] / total_importance) * 100
        table.add_row(
            str(i),
            row['feature'],
            f"{row['importance']:.5f}",
            f"{percentage:.2f}%"
        )
    
    rich_print("\n")
    rich_print(table)

def evaluate_model(y_true, y_pred, y_pred_proba=None, feature_importance_df=None, model_name="Model"):
    """
    Comprehensive model evaluation printing all metrics and visualizations.
    Works for both binary and multiclass classification.
    """
    print_overall_metrics(y_true, y_pred, y_pred_proba, model_name)
    print_confusion_matrix(y_true, y_pred, model_name)
    
    if feature_importance_df is not None:
        print_feature_importance(feature_importance_df, model_name=model_name)

# Example usage
"""
# For binary classification
evaluate_model(
    y_test, 
    y_pred, 
    y_pred_proba,
    feature_importance_df, 
    "Binary Classifier"
)

# For multiclass
evaluate_model(
    y_test, 
    y_pred_multi, 
    y_pred_proba_multi,
    feature_importance_df, 
    "Multiclass Classifier"
)
"""