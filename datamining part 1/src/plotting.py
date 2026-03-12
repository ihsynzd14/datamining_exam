import matplotlib.pyplot as plt
import seaborn as sns
import os

IMAGE_DIR = os.path.join(os.path.dirname(__file__), '../report/images')

def save_plot(fig, filename):
    """
    Save a matplotlib figure to the report/images directory.
    Useful for including in the LaTeX report.
    """
    filepath = os.path.join(IMAGE_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {filepath}")

def setup_style():
    """
    Setup standard plotting styles for consistent output in the report.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14
    })
