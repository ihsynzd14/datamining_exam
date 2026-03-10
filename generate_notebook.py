import json

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# --- Task 1 Markdown and Code ---

add_md("""# Task 1: Data Understanding & Preparation

This notebook does the first step of our project. We will load the data, understand what it means, find missing values, and prepare it for the next steps. All comments are written in simple English.""")

add_md("""## 1. Setup and Load Data
First, we load the required libraries. We also load our dataset from the `dataset` folder.""")

add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# We use the plotting functions we created in src/plotting.py
import sys
import os
sys.path.append(os.path.abspath('../src'))
from plotting import save_plot, setup_style

setup_style()

# Load the dataset
df = pd.read_csv('../dataset/DM1_game_dataset.csv')
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
df.head(3)""")

add_md("""## 2. Data Semantics
Here we look at the data types and see if there are any missing values. This helps us understand what information we have.""")

add_code("""# Check data types and missing values
df.info()

# Count how many missing values we have in each column
missing_counts = df.isnull().sum()
missing_counts[missing_counts > 0]""")

add_md("""### Fixing Missing Values
We found missing values in `ComAgeRec`, `LanguageEase`, `Family`, `ImagePath`, and `Description`.
*   **Family**: Has too many missing values (more than 15000). We will drop this column because it is not useful.
*   **ComAgeRec & LanguageEase**: We will fill the missing values with the median (the middle value) to keep the data shape.
*   **ImagePath & Description**: Very few missing values. We will just drop these rows.
*   **GoodPlayers**: This column is stored as text (object) but it's a list. We will drop it for now or keep it as text, as it's hard to use directly in models.
*   **ImagePath, Description, Name**: We don't need these text/URL columns for machine learning. We will drop them to clean our data.""")

add_code("""# Make a copy of the original data to clean it
clean_df = df.copy()

# 1. Drop columns we do not need for machine learning
cols_to_drop = ['Family', 'ImagePath', 'Description', 'Name', 'GoodPlayers']
clean_df = clean_df.drop(columns=cols_to_drop)

# 2. Fill missing values with median for numeric columns
clean_df['ComAgeRec'] = clean_df['ComAgeRec'].fillna(clean_df['ComAgeRec'].median())
clean_df['LanguageEase'] = clean_df['LanguageEase'].fillna(clean_df['LanguageEase'].median())

# Check again to make sure no missing values are left
print("Missing values after cleaning:")
print(clean_df.isnull().sum().sum())""")

add_md("""## 3. Statistics and Distributions
Now we look at the basic numbers (like mean, min, max) for our numeric variables. Then we plot them to see their shape.""")

add_code("""# Show basic statistics
clean_df.describe().T""")

add_md("""Let's draw histograms for some important variables like `YearPublished`, `GameWeight`, and `NumOwned` to see if there are outliers (strange extreme values).""")

add_code("""# Plot distributions of important variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot Year Published
sns.histplot(clean_df['YearPublished'], bins=50, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Year Published')
axes[0].set_xlim(1900, 2025) # Zoom in to ignore very old years for a better view

# Plot Game Weight (Complexity)
sns.histplot(clean_df['GameWeight'], bins=30, ax=axes[1], color='lightgreen')
axes[1].set_title('Distribution of Game Weight')

# Plot Number of Owners
sns.histplot(clean_df['NumOwned'], bins=50, ax=axes[2], color='salmon')
axes[2].set_title('Distribution of Number Owned')
axes[2].set_yscale('log') # Use log scale because some games are owned by too many people

plt.tight_layout()
save_plot(fig, 'variable_distributions.png')
plt.show()""")

add_md("""### Analysis of Distributions
1.  **Year Published**: Most games were published after 2000. It shows a big increase in recent years.
2.  **Game Weight**: Most games have a weight (complexity) between 1 and 3. The shape is a bit skewed to the right.
3.  **NumOwned**: We used a log scale because a few games are owned by a very large number of people. This means there are some outliers.

Next, let's look at the target variable we will use for Classification: **Rating**.""")

add_code("""# Plot the target variable
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='Rating', data=clean_df, order=['Low', 'Medium', 'High'], palette='Set2', ax=ax)
ax.set_title('Distribution of Target Variable: Rating')

save_plot(fig, 'rating_distribution.png')
plt.show()""")

add_md("""## 4. Pairwise Correlations
We want to see if some variables are highly correlated (very similar to each other). If two variables give the same information, we can remove one. This makes our models simpler and faster.""")

add_code("""# Calculate correlation matrix for numeric columns only
numeric_cols = clean_df.select_vars = clean_df.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

# Draw the heatmap
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, ax=ax)
ax.set_title('Correlation Matrix of Numeric Variables')

save_plot(fig, 'correlation_matrix.png')
plt.show()""")

add_md("""### Removing Highly Correlated Variables
From the map, we see that:
*   `ComMinPlaytime` and `ComMaxPlaytime` are highly correlated with `MfgPlaytime`. We can keep just `MfgPlaytime`.
*   `NumWant`, `NumWish`, `NumUserRatings` are highly correlated with `NumOwned`. We can keep `NumOwned` to represent popularity.
*   `GameWeight` and `ComWeight` are very similar. We can keep `GameWeight`.

Let's remove the extra columns to finish our data preparation.""")

add_code("""# Drop highly correlated columns
highly_correlated_cols = ['ComMinPlaytime', 'ComMaxPlaytime', 'NumWant', 'NumWish', 'NumUserRatings', 'NumComments', 'ComWeight']
clean_df = clean_df.drop(columns=highly_correlated_cols)

print(f"Final dataset has {clean_df.shape[0]} rows and {clean_df.shape[1]} columns.")
clean_df.head()""")

add_md("""## 5. Save the Clean Data
We save the cleaned dataset to the `processed` folder so we can use it in Task 2 (Clustering) and Task 3 (Classification & Regression).""")

add_code("""import os
os.makedirs('../dataset/processed', exist_ok=True)

# Save to CSV
clean_path = '../dataset/processed/DM1_game_dataset_clean.csv'
clean_df.to_csv(clean_path, index=False)
print(f"Data successfully saved to {clean_path}")""")

# Write notebook to file
with open('datamining part 1/notebooks/01_data_understanding.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

